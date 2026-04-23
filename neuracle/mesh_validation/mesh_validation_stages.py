"""
Mesh validation 闃舵鎵ц閫昏緫銆?
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.spatial import cKDTree

from neuracle.atlas import get_standardized_roi_path
from neuracle.mesh_validation.mesh_validation_schema import (
    RUN_STATE_COMPLETED,
    RUN_STATE_FAILED,
    RUN_STATE_RUNNING,
    RunStateTracker,
    forward_result_path,
    forward_run_path,
    inverse_result_path,
    inverse_run_path,
    mesh_result_path,
    mesh_run_path,
    preset_paths_for,
    read_json,
    replay_result_path,
    replay_run_path,
    wait_for_run_completion,
    write_json,
)
from neuracle.mesh_validation.mesh_validation_workspace import (
    MeshGenerationError,
    SubjectConfig,
    WorkspaceBuildResult,
    WorkspacePreparationError,
    load_existing_workspace,
    prepare_workspace,
    resolve_workspace_eeg_cap,
)
from neuracle.ti_optimization import init_optimization, run_optimization, setup_goal
from neuracle.ti_simulation import (
    calculate_ti,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)
from neuracle.utils.constants import EEG10_20_EXTENDED_SPM12
from neuracle.utils.ti_export import export_ti_to_nifti
from simnibs.mesh_tools import mesh_io
from simnibs.optimization.tes_flex_optimization.measures import ROC
from simnibs.utils import transformations
from simnibs.utils.mesh_element_properties import ElementTags

LOGGER = logging.getLogger("mesh_validation")
INVERSE_PRESET_ORDER = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
FORWARD_THRESHOLDS = {
    "M1": {"mean_median": 0.05, "tail": 0.08, "hotspot": 0.10, "nrmse": 0.08},
    "M2": {"mean_median": 0.08, "tail": 0.10, "hotspot": 0.15, "nrmse": 0.12},
    "M3": {"mean_median": 0.12, "tail": 0.15, "hotspot": 0.20, "nrmse": 0.12},
}
INVERSE_THRESHOLDS = {"M1": 0.05, "M2": 0.08, "M3": 0.12, "M4": 0.12}
FORWARD_HOTSPOT_THRESHOLDS_ABS = [0.10, 0.15, 0.20, 0.25, 0.30]


def sort_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    瀵圭粨鏋滆鍋氱ǔ瀹氭帓搴忋€?

    Parameters
    ----------
    rows : list[dict[str, Any]]
        缁撴灉琛屽垪琛ㄣ€?

    Returns
    -------
    list[dict[str, Any]]
        鎺掑簭鍚庣殑缁撴灉琛屽垪琛ㄣ€?
    """
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("subject_id", "")),
            str(row.get("case_name", "")),
            int(row.get("seed", -1)) if row.get("seed") is not None else -1,
            str(row.get("preset", "")),
            str(row.get("status", "")),
        ),
    )


def _resolve_failure_stage(default_failure_stage: str, exc: Exception) -> str:
    """
    瑙ｆ瀽澶辫触闃舵鍚嶇О銆?

    Parameters
    ----------
    default_failure_stage : str
        榛樿澶辫触闃舵鍚嶇О銆?
    exc : Exception
        鎹曡幏鍒扮殑寮傚父銆?

    Returns
    -------
    str
        failure_stage銆?
    """
    if isinstance(exc, MeshGenerationError):
        return "mesh_generation_failed"
    if isinstance(exc, WorkspacePreparationError):
        return "workspace_prepare_failed"
    return default_failure_stage


def _execute_stage_task(
    stage: str,
    result_path: Path,
    run_path: Path,
    command: str,
    payload: dict[str, Any],
    default_failure_stage: str,
    runner: Any,
) -> dict[str, Any]:
    """
    鎵ц鍗曚釜闃舵浠诲姟骞剁淮鎶?result/run 鏂囦欢銆?

    Parameters
    ----------
    stage : str
        闃舵鍚嶇О銆?
    result_path : Path
        result.json 璺緞銆?
    run_path : Path
        run.json 璺緞銆?
    command : str
        褰撳墠鍛戒护琛屻€?
    payload : dict[str, Any]
        缁撴灉鍩虹瀛楁銆?
    default_failure_stage : str
        榛樿澶辫触闃舵鍚嶇О銆?
    runner : Any
        鐪熸鎵ц閫昏緫鐨勫洖璋冦€?

    Returns
    -------
    dict[str, Any]
        result.json 瀵硅薄銆?
    """
    tracker = RunStateTracker(run_path=run_path, stage=stage, command=command, metadata=payload)
    tracker.start()
    try:
        result = runner()
    except Exception as exc:  # noqa: BLE001
        failure = {
            **payload,
            "status": "failed",
            "failure_stage": _resolve_failure_stage(default_failure_stage, exc),
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }
        write_json(result_path, failure)
        tracker.finish(RUN_STATE_FAILED, error=str(exc))
        LOGGER.exception("%s 鎵ц澶辫触: %s", stage, payload)
        return failure
    write_json(result_path, result)
    tracker.finish(RUN_STATE_COMPLETED)
    return result


def collect_parallel_rows(task_args: list[tuple[Any, ...]], worker_fn: Any, max_workers: int) -> list[dict[str, Any]]:
    """
    浣跨敤 spawn 杩涚▼姹犳墽琛?worker銆?

    Parameters
    ----------
    task_args : list[tuple[Any, ...]]
        姣忎釜 worker 鐨勫弬鏁板厓缁勩€?
    worker_fn : Any
        妯″潡绾?worker 鍏ュ彛銆?
    max_workers : int
        鏈€澶у苟琛岃繘绋嬫暟銆?

    Returns
    -------
    list[dict[str, Any]]
        鑱氬悎鍚庣殑缁撴灉琛屻€?
    """
    if not task_args:
        return []
    if max_workers <= 1 or len(task_args) == 1:
        rows: list[dict[str, Any]] = []
        for args in task_args:
            rows.extend(worker_fn(*args))
        return sort_result_rows(rows)
    rows: list[dict[str, Any]] = []
    spawn_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(max_workers, len(task_args)), mp_context=spawn_context) as executor:
        futures = [executor.submit(worker_fn, *args) for args in task_args]
        for future in as_completed(futures):
            rows.extend(future.result())
    return sort_result_rows(rows)


def _prepare_stage_artifacts_dir(stage_dir: Path) -> Path:
    """
    閲嶅缓闃舵宸ヤ欢鐩綍銆?

    Parameters
    ----------
    stage_dir : Path
        闃舵鐩綍銆?

    Returns
    -------
    Path
        宸ヤ欢鐩綍銆?
    """
    stage_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = stage_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return artifacts_dir


def _load_completed_stage_result(
    run_path: Path,
    result_path: Path,
    description: str,
    wait_on_running: bool,
) -> dict[str, Any]:
    """
    璇诲彇宸插畬鎴愮殑闃舵缁撴灉銆?

    Parameters
    ----------
    run_path : Path
        run.json 璺緞銆?
    result_path : Path
        result.json 璺緞銆?
    description : str
        鎻忚堪瀛楃涓层€?
    wait_on_running : bool
        閬囧埌 running 鏄惁绛夊緟銆?

    Returns
    -------
    dict[str, Any]
        result.json 鍐呭銆?
    """
    run_payload = read_json(run_path)
    if run_payload is None:
        raise FileNotFoundError(f"缂哄皯杩愯鐘舵€? {description} -> {run_path}")
    state = str(run_payload.get("state"))
    if state == RUN_STATE_RUNNING:
        if not wait_on_running:
            raise RuntimeError(f"鍓嶅簭浠诲姟浠嶅湪杩愯: {description}")
        run_payload = wait_for_run_completion(run_path, description)
        state = str(run_payload.get("state"))
    if state != RUN_STATE_COMPLETED:
        raise RuntimeError(f"鍓嶅簭浠诲姟鏈畬鎴? {description} -> {state}")
    result_payload = read_json(result_path)
    if result_payload is None:
        raise FileNotFoundError(f"缺少结果文件: {description} -> {result_path}")
    return result_payload


def _ensure_m0_workspace_ready(
    subject: SubjectConfig,
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult],
) -> WorkspaceBuildResult:
    """
    纭繚 M0 workspace 宸插畬鎴愪笖鍙敤銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    workspace_cache : dict[tuple[str, str], WorkspaceBuildResult]
        workspace 缂撳瓨銆?

    Returns
    -------
    WorkspaceBuildResult
        M0 workspace銆?
    """
    paths = preset_paths_for(work_root, subject.id, "M0")
    _load_completed_stage_result(
        run_path=mesh_run_path(paths),
        result_path=mesh_result_path(paths),
        description=f"M0 mesh {subject.id}",
        wait_on_running=True,
    )
    return ensure_workspace(
        workspace_cache=workspace_cache,
        subject=subject,
        preset="M0",
        work_root=work_root,
        preset_ini_paths=preset_ini_paths,
        debug_mesh=debug_mesh,
    )


def _load_inverse_result(
    subject: SubjectConfig,
    preset: str,
    case_name: str,
    seed: int,
    work_root: Path,
    wait_on_running: bool,
) -> dict[str, Any]:
    """
    鍔犺浇 inverse 姝ｅ紡缁撴灉銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    case_name : str
        case 鍚嶇О銆?
    seed : int
        闅忔満绉嶅瓙銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    wait_on_running : bool
        閬囧埌 running 鏄惁绛夊緟銆?

    Returns
    -------
    dict[str, Any]
        inverse 缁撴灉銆?
    """
    paths = preset_paths_for(work_root, subject.id, preset)
    return _load_completed_stage_result(
        run_path=inverse_run_path(paths, case_name, seed),
        result_path=inverse_result_path(paths, case_name, seed),
        description=f"{preset} inverse {subject.id}/{case_name}/{seed}",
        wait_on_running=wait_on_running,
    )


def _load_replay_result(
    subject: SubjectConfig,
    preset: str,
    case_name: str,
    seed: int,
    work_root: Path,
    wait_on_running: bool,
) -> dict[str, Any]:
    """
    鍔犺浇 replay 姝ｅ紡缁撴灉銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    case_name : str
        case 鍚嶇О銆?
    seed : int
        闅忔満绉嶅瓙銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    wait_on_running : bool
        閬囧埌 running 鏄惁绛夊緟銆?

    Returns
    -------
    dict[str, Any]
        replay 缁撴灉銆?
    """
    paths = preset_paths_for(work_root, subject.id, preset)
    return _load_completed_stage_result(
        run_path=replay_run_path(paths, case_name, seed),
        result_path=replay_result_path(paths, case_name, seed),
        description=f"{preset} replay {subject.id}/{case_name}/{seed}",
        wait_on_running=wait_on_running,
    )


def _run_mesh_preset_worker(
    subject: SubjectConfig,
    preset: str,
    preset_ini_paths: dict[str, Path],
    work_root: Path,
    debug_mesh: bool,
    command: str,
) -> list[dict[str, Any]]:
    """
    鎵ц鍗曚釜 `(subject, preset)` 鐨?mesh 闃舵銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        mesh 缁撴灉琛屻€?
    """
    payload = {"subject_id": subject.id, "preset": preset}
    paths = preset_paths_for(work_root, subject.id, preset)
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}

    def runner() -> dict[str, Any]:
        start = time.perf_counter()
        workspace = ensure_workspace(
            workspace_cache=workspace_cache,
            subject=subject,
            preset=preset,
            work_root=work_root,
            preset_ini_paths=preset_ini_paths,
            debug_mesh=debug_mesh,
        )
        elapsed = time.perf_counter() - start
        stats = summarize_mesh(workspace.mesh_file)
        return {
            **payload,
            "status": "ok",
            "elapsed_seconds": elapsed,
            "workspace_dir": str(workspace.workspace_dir),
            "mesh_file": str(workspace.mesh_file),
            "final_labels_path": str(workspace.final_tissues_path),
            **stats,
        }

    result = _execute_stage_task(
        stage="mesh",
        result_path=mesh_result_path(paths),
        run_path=mesh_run_path(paths),
        command=command,
        payload=payload,
        default_failure_stage="mesh_generation_failed",
        runner=runner,
    )
    return [result]


def build_mesh_variants(
    subjects: list[SubjectConfig],
    presets: list[str],
    preset_ini_paths: dict[str, Path],
    work_root: Path,
    debug_mesh: bool,
    preset_workers: int,
    command: str,
) -> list[dict[str, Any]]:
    """
    鏋勫缓 workspace 骞剁粺璁?mesh 鎸囨爣銆?

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 鍒楄〃銆?
    presets : list[str]
        preset 鍒楄〃銆?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    preset_workers : int
        preset 骞惰杩涚▼鏁般€?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        mesh 缁撴灉琛屻€?
    """
    task_args = [(subject, preset, preset_ini_paths, work_root, debug_mesh, command) for subject in subjects for preset in presets]
    return collect_parallel_rows(task_args, _run_mesh_preset_worker, preset_workers)


def build_spec_mask(spec: dict[str, Any], reference_img: nib.Nifti1Image, subject_dir: str, cache_dir: Path, cache_key: str) -> np.ndarray:
    """
    鏋勫缓 ROI 鎺╃爜銆?

    Parameters
    ----------
    spec : dict[str, Any]
        ROI 閰嶇疆銆?
    reference_img : nib.Nifti1Image
        鍙傝€冨浘鍍忋€?
    subject_dir : str
        subject 鐩綍銆?
    cache_dir : Path
        缂撳瓨鐩綍銆?
    cache_key : str
        缂撳瓨閿€?

    Returns
    -------
    np.ndarray
        甯冨皵鎺╃爜銆?
    """
    spec = materialize_roi_spec(spec)
    spec_type = spec.get("type", "sphere")
    if spec_type == "sphere":
        ijk = np.indices(reference_img.shape, dtype=float).reshape(3, -1).T
        xyz = nib.affines.apply_affine(reference_img.affine, ijk)
        center = np.asarray(spec["center"], dtype=float)
        radius = float(spec["radius"])
        return (np.linalg.norm(xyz - center, axis=1) <= radius).reshape(reference_img.shape)
    if spec_type != "mask":
        raise ValueError(f"涓嶆敮鎸佺殑 ROI 绫诲瀷: {spec_type}")
    mask_img = nib.load(spec["path"])
    if spec.get("space", "subject") == "mni":
        cache_dir.mkdir(parents=True, exist_ok=True)
        warped_path = cache_dir / f"{cache_key}_mni2subject.nii.gz"
        if not warped_path.exists():
            transformations.warp_volume(
                spec["path"],
                subject_dir,
                str(warped_path),
                transformation_direction="mni2subject",
                transformation_type="nonl",
                reference=str(Path(subject_dir) / "T1.nii.gz"),
                order=0,
                binary=True,
            )
        mask_img = nib.load(str(warped_path))
    if mask_img.shape != reference_img.shape or not np.allclose(mask_img.affine, reference_img.affine):
        mask_img = resample_from_to(mask_img, reference_img, order=0)
    return np.asarray(mask_img.get_fdata()) == spec.get("value", 1)


def materialize_roi_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    Materialize atlas ROI specs into mask ROI specs.
    """
    if spec.get("type", "sphere") != "atlas":
        return spec
    return {
        "type": "mask",
        "path": str(get_standardized_roi_path(spec["atlas_name"], spec["areas"])),
        "space": spec.get("space", "mni"),
        "value": spec.get("value", 1),
    }


def compute_gm_threshold_metrics(
    ti_data: np.ndarray,
    gm_mask: np.ndarray,
    thresholds_abs: list[float],
) -> list[dict[str, float]]:
    """
    Compute current-case GM threshold counts for each absolute threshold.
    """
    gm_voxels = int(np.count_nonzero(gm_mask))
    metrics: list[dict[str, float]] = []
    for threshold in thresholds_abs:
        active_voxels = int(np.count_nonzero(gm_mask & (ti_data >= float(threshold))))
        metrics.append(
            {
                "threshold_abs": float(threshold),
                "gm_voxels": gm_voxels,
                "current_gm_threshold_voxels": active_voxels,
                "current_gm_threshold_fraction": float(active_voxels / gm_voxels) if gm_voxels else math.nan,
            }
        )
    return metrics


def load_volume_data(volume_path: Path, reference_img: nib.Nifti1Image) -> np.ndarray:
    """
    璇诲彇骞跺榻愪綋绉暟鎹€?

    Parameters
    ----------
    volume_path : Path
        浣撶Н璺緞銆?
    reference_img : nib.Nifti1Image
        鍙傝€冨浘鍍忋€?

    Returns
    -------
    np.ndarray
        瀵归綈鍚庣殑浣撶Н鏁版嵁銆?
    """
    image = nib.squeeze_image(nib.load(str(volume_path)))
    reference_img = nib.squeeze_image(reference_img)
    if image.shape != reference_img.shape or not np.allclose(image.affine, reference_img.affine):
        image = resample_from_to(image, reference_img, order=0)
    return np.asarray(image.get_fdata())



def compute_forward_case_metrics(
    ti_volume_path: Path,
    final_labels_path: Path,
    case: dict[str, Any],
) -> dict[str, Any]:
    """
    计算 forward case 的 GM 指标。

    Parameters
    ----------
    ti_volume_path : Path
        TI 体积路径。
    final_labels_path : Path
        最终组织标签路径。
    case : dict[str, Any]
        forward case 配置。

    Returns
    -------
    dict[str, Any]
        case 级指标结果。
    """
    reference_img = nib.squeeze_image(nib.load(str(ti_volume_path)))
    ti_data = load_volume_data(ti_volume_path, reference_img)
    current_labels = load_volume_data(final_labels_path, reference_img)
    gm_mask = current_labels == ElementTags.GM
    return {
        "gm_peak_value": float(np.max(ti_data[gm_mask])) if np.any(gm_mask) else math.nan,
        "gm_threshold_metrics": compute_gm_threshold_metrics(
            ti_data=ti_data,
            gm_mask=gm_mask,
            thresholds_abs=[float(value) for value in case.get("gm_thresholds_abs", FORWARD_HOTSPOT_THRESHOLDS_ABS)],
        ),
    }


def _run_forward_preset_worker(
    subject: SubjectConfig,
    preset: str,
    forward_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    command: str,
) -> list[dict[str, Any]]:
    """
    鎵ц鍗曚釜 `(subject, preset)` 鐨?forward 闃舵銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    forward_cases : list[dict[str, Any]]
        forward case 鍒楄〃銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        forward 缁撴灉琛屻€?
    """
    rows: list[dict[str, Any]] = []
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}
    paths = preset_paths_for(work_root, subject.id, preset)
    for case in forward_cases:
        payload = {"subject_id": subject.id, "case_name": case["name"], "preset": preset}
        stage_dir = paths.forward_root / case["name"]

        def runner() -> dict[str, Any]:
            start = time.perf_counter()
            workspace = ensure_workspace(
                workspace_cache=workspace_cache,
                subject=subject,
                preset=preset,
                work_root=work_root,
                preset_ini_paths=preset_ini_paths,
                debug_mesh=debug_mesh,
            )
            output_dir = _prepare_stage_artifacts_dir(stage_dir)
            session = setup_session(
                subject_dir=str(workspace.workspace_dir),
                output_dir=str(output_dir),
                msh_file_path=str(workspace.mesh_file),
                eeg_cap=resolve_workspace_eeg_cap(workspace, case["eeg_cap"]),
            )
            setup_electrode_pair1(
                session,
                case["pair1"],
                case["current1"],
                case.get("electrode_shape", "ellipse"),
                case.get("electrode_dimensions", [40, 40]),
                float(case.get("electrode_thickness", 2.0)),
            )
            setup_electrode_pair2(
                session,
                case["pair2"],
                case["current2"],
                case.get("electrode_shape", "ellipse"),
                case.get("electrode_dimensions", [40, 40]),
                float(case.get("electrode_thickness", 2.0)),
            )
            mesh1_path, mesh2_path = run_tdcs_simulation(session, str(workspace.workspace_dir), str(output_dir), int(case.get("n_workers", 1)))
            ti_mesh_path = Path(calculate_ti(mesh1_path, mesh2_path, str(output_dir)))
            ti_nifti_path = Path(export_ti_to_nifti(str(ti_mesh_path), str(output_dir), subject.reference_t1, "max_TI", f"{subject.id}_{case['name']}"))
            metrics = compute_forward_case_metrics(
                ti_volume_path=ti_nifti_path,
                final_labels_path=workspace.final_tissues_path,
                case=case,
            )
            elapsed = time.perf_counter() - start
            return {
                **payload,
                "status": "ok",
                "elapsed_seconds": elapsed,
                "ti_mesh_path": str(ti_mesh_path),
                "ti_volume_path": str(ti_nifti_path),
                "final_labels_path": str(workspace.final_tissues_path),
                **metrics,
            }

        rows.append(
            _execute_stage_task(
                stage="forward",
                result_path=forward_result_path(paths, case["name"]),
                run_path=forward_run_path(paths, case["name"]),
                command=command,
                payload=payload,
                default_failure_stage="forward_execution_failed",
                runner=runner,
            )
        )
    return rows


def run_forward_validation(
    subjects: list[SubjectConfig],
    presets: list[str],
    forward_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    preset_workers: int,
    command: str,
) -> list[dict[str, Any]]:
    """
    杩愯 forward TI 楠岃瘉銆?

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 鍒楄〃銆?
    presets : list[str]
        preset 鍒楄〃銆?
    forward_cases : list[dict[str, Any]]
        forward case 鍒楄〃銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    preset_workers : int
        preset 骞惰杩涚▼鏁般€?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        forward 缁撴灉琛屻€?
    """
    task_args = [(subject, preset, forward_cases, work_root, preset_ini_paths, debug_mesh, command) for subject in subjects for preset in presets]
    return collect_parallel_rows(task_args, _run_forward_preset_worker, preset_workers)


def read_hdf5_scalar(dataset: Any) -> Any:
    """
    璇诲彇 HDF5 鏍囬噺銆?

    Parameters
    ----------
    dataset : Any
        HDF5 鏁版嵁闆嗐€?

    Returns
    -------
    Any
        鏍囬噺鍊笺€?
    """
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def parse_inverse_summary(summary_path: Path) -> dict[str, Any]:
    """
    瑙ｆ瀽浼樺寲 summary.hdf5銆?

    Parameters
    ----------
    summary_path : Path
        summary 璺緞銆?

    Returns
    -------
    dict[str, Any]
        鎽樿缁撴灉銆?
    """
    result: dict[str, Any] = {}
    with h5py.File(summary_path, "r") as handle:
        result["optimizer"] = read_hdf5_scalar(handle["optimizer/optimizer"])
        result["fopt"] = float(read_hdf5_scalar(handle["optimizer/fopt"]))
        result["n_test"] = int(read_hdf5_scalar(handle["optimizer/n_test"]))
        result["n_sim"] = int(read_hdf5_scalar(handle["optimizer/n_sim"]))
    return result


def load_mapping(mapping_path: Path) -> list[dict[str, Any]]:
    """
    璇诲彇 electrode mapping銆?

    Parameters
    ----------
    mapping_path : Path
        mapping 璺緞銆?

    Returns
    -------
    list[dict[str, Any]]
        瑙勮寖鍖栧悗鐨?mapping 鏉＄洰銆?
    """
    with mapping_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    entries = []
    for optimized, mapped, label, distance, index_pair in zip(
        raw["optimized_positions"],
        raw["mapped_positions"],
        raw["mapped_labels"],
        raw["distances"],
        raw["channel_array_indices"],
        strict=True,
    ):
        entries.append(
            {
                "channel": int(index_pair[0]),
                "array": int(index_pair[1]),
                "optimized_position": np.asarray(optimized, dtype=float),
                "mapped_position": np.asarray(mapped, dtype=float),
                "mapped_label": label,
                "mapping_distance_mm": float(distance),
            }
        )
    entries.sort(key=lambda item: (item["channel"], item["array"]))
    return entries


def add_inverse_roi(opt: Any, spec: dict[str, Any], mesh_file: str, default_difference: bool = False) -> None:
    """
    鍚戜紭鍖栧璞℃坊鍔?ROI銆?

    Parameters
    ----------
    opt : Any
        浼樺寲瀵硅薄銆?
    spec : dict[str, Any]
        ROI 閰嶇疆銆?
    mesh_file : str
        mesh 璺緞銆?
    default_difference : bool, optional
        鏄惁榛樿浣滀负 difference ROI銆?

    Returns
    -------
    None
    """
    spec = materialize_roi_spec(spec)
    roi = opt.add_roi()
    roi.method = "volume"
    roi.mesh = mesh_file
    roi.subpath = opt.subpath
    roi.tissues = [ElementTags.WM, ElementTags.GM]
    if spec.get("type", "sphere") == "mask":
        roi.mask_path = spec["path"]
        roi.mask_space = spec.get("space", "subject")
        roi.mask_value = spec.get("value", 1)
        if default_difference:
            roi.mask_operator = ["difference"]
    else:
        roi.roi_sphere_center = spec["center"]
        roi.roi_sphere_radius = float(spec["radius"])
        roi.roi_sphere_center_space = spec.get("space", "subject")
        if default_difference:
            roi.roi_sphere_operator = ["difference"]


def configure_inverse_case(opt: Any, case: dict[str, Any], mesh_file: str, net_file: str) -> None:
    """
    閰嶇疆 inverse case銆?

    Parameters
    ----------
    opt : Any
        浼樺寲瀵硅薄銆?
    case : dict[str, Any]
        inverse case 閰嶇疆銆?
    mesh_file : str
        mesh 璺緞銆?
    net_file : str
        鐢垫瀬缃戞枃浠惰矾寰勩€?

    Returns
    -------
    None
    """
    setup_goal(
        opt=opt,
        goal=case["goal"],
        focality_threshold=case.get("threshold"),
        net_electrode_file=net_file,
        optimizer=case.get("optimizer", "differential_evolution"),
        track_focality=True,
        detailed_results=True,
    )
    pair1 = opt.add_electrode_layout("ElectrodeArrayPair")
    pair1.center = case.get("electrode_pair1_center", [[0, 0]])
    pair1.radius = case.get("electrode_radius", [10])
    pair1.current = case.get("electrode_current1", [0.002, -0.002])
    pair2 = opt.add_electrode_layout("ElectrodeArrayPair")
    pair2.center = case.get("electrode_pair2_center", [[0, 0]])
    pair2.radius = case.get("electrode_radius", [10])
    pair2.current = case.get("electrode_current2", [0.002, -0.002])
    add_inverse_roi(opt, case["roi"], mesh_file=mesh_file, default_difference=False)
    if case["goal"] == "focality":
        add_inverse_roi(opt, case.get("non_roi", case["roi"]), mesh_file=mesh_file, default_difference=True)


def compute_goal_from_volume(
    ti_data: np.ndarray,
    reference_img: nib.Nifti1Image,
    subject_dir: str,
    case: dict[str, Any],
    cache_dir: Path,
) -> dict[str, float]:
    """
    鏍规嵁 TI 浣撶Н閲嶇畻鐩爣鍊笺€?

    Parameters
    ----------
    ti_data : np.ndarray
        TI 鏁版嵁銆?
    reference_img : nib.Nifti1Image
        鍙傝€冨浘鍍忋€?
    subject_dir : str
        subject 鐩綍銆?
    case : dict[str, Any]
        inverse case 閰嶇疆銆?
    cache_dir : Path
        缂撳瓨鐩綍銆?

    Returns
    -------
    dict[str, float]
        replay 鎸囨爣缁撴灉銆?
    """
    roi_mask = build_spec_mask(case["roi"], reference_img, subject_dir, cache_dir, f"{case['name']}_inverse_roi")
    roi_values = ti_data[roi_mask]
    result = {"replay_roi_mean": float(np.mean(roi_values)), "replay_roi_p999": float(np.percentile(roi_values, 99.9))}
    if case["goal"] == "mean":
        result["replay_goal"] = float(-np.mean(roi_values))
        return result
    if case["goal"] == "max":
        result["replay_goal"] = float(-np.percentile(roi_values, 99.9))
        return result
    non_roi_mask = build_spec_mask(case.get("non_roi", case["roi"]), reference_img, subject_dir, cache_dir, f"{case['name']}_inverse_non_roi")
    non_roi_mask = np.logical_and(non_roi_mask, np.logical_not(roi_mask))
    non_roi_values = ti_data[non_roi_mask]
    roc_value = float(ROC(roi_values, non_roi_values, case.get("threshold", [0.1, 0.2]), focal=True))
    result["replay_non_roi_mean"] = float(np.mean(non_roi_values))
    result["replay_roc"] = roc_value
    result["replay_goal"] = float(-100 * (math.sqrt(2) - roc_value))
    return result


def group_labels_by_channel(entries: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    浠庢槧灏勭粨鏋滄仮澶嶄袱瀵圭數鏋佹爣绛俱€?

    Parameters
    ----------
    entries : list[dict[str, Any]]
        鏄犲皠鏉＄洰銆?

    Returns
    -------
    tuple[list[str], list[str]]
        涓ゅ鐢垫瀬鏍囩銆?
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry["channel"], []).append(entry)
    pair1 = [item["mapped_label"] for item in sorted(grouped.get(0, []), key=lambda item: item["array"])]
    pair2 = [item["mapped_label"] for item in sorted(grouped.get(1, []), key=lambda item: item["array"])]
    return pair1, pair2


def compute_mapping_drift(baseline_entries: list[dict[str, Any]], candidate_entries: list[dict[str, Any]]) -> dict[str, float | bool]:
    """
    璁＄畻鏄犲皠婕傜Щ鎸囨爣銆?

    Parameters
    ----------
    baseline_entries : list[dict[str, Any]]
        M0 鍩虹嚎鏄犲皠銆?
    candidate_entries : list[dict[str, Any]]
        褰撳墠鏄犲皠銆?

    Returns
    -------
    dict[str, float | bool]
        婕傜Щ鎸囨爣銆?
    """
    if len(baseline_entries) != len(candidate_entries):
        return {"label_consistent": False}
    optimized_drifts = []
    mapped_drifts = []
    label_consistent = True
    for baseline, candidate in zip(baseline_entries, candidate_entries, strict=True):
        optimized_drifts.append(float(np.linalg.norm(candidate["optimized_position"] - baseline["optimized_position"])))
        mapped_drifts.append(float(np.linalg.norm(candidate["mapped_position"] - baseline["mapped_position"])))
        label_consistent = label_consistent and baseline["mapped_label"] == candidate["mapped_label"]
    return {
        "label_consistent": label_consistent,
        "optimized_center_drift_mean_mm": float(np.mean(optimized_drifts)),
        "optimized_center_drift_max_mm": float(np.max(optimized_drifts)),
        "mapped_center_drift_mean_mm": float(np.mean(mapped_drifts)),
        "mapped_center_drift_max_mm": float(np.max(mapped_drifts)),
    }


def run_replay_on_m0(
    subject: SubjectConfig,
    preset: str,
    case: dict[str, Any],
    seed: int,
    mapped_entries: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult],
) -> Path:
    """
    鍦?M0 workspace 涓婂洖鏀炬槧灏勫悗鐨勭數鏋併€?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        鏉ユ簮 preset銆?
    case : dict[str, Any]
        inverse case 閰嶇疆銆?
    seed : int
        闅忔満绉嶅瓙銆?
    mapped_entries : list[dict[str, Any]]
        鏄犲皠缁撴灉銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    workspace_cache : dict[tuple[str, str], WorkspaceBuildResult]
        workspace 缂撳瓨銆?

    Returns
    -------
    Path
        replay TI NIfTI 璺緞銆?
    """
    candidate_paths = preset_paths_for(work_root, subject.id, preset)
    replay_dir = _prepare_stage_artifacts_dir(candidate_paths.replay_root / case["name"] / "seeds" / str(seed))
    pair1, pair2 = group_labels_by_channel(mapped_entries)
    baseline_workspace = ensure_workspace(
        workspace_cache=workspace_cache,
        subject=subject,
        preset="M0",
        work_root=work_root,
        preset_ini_paths=preset_ini_paths,
        debug_mesh=debug_mesh,
    )
    session = setup_session(
        str(baseline_workspace.workspace_dir),
        str(replay_dir),
        str(baseline_workspace.mesh_file),
        eeg_cap=resolve_workspace_eeg_cap(baseline_workspace, case.get("net_electrode_file", EEG10_20_EXTENDED_SPM12)),
    )
    setup_electrode_pair1(session, pair1, case.get("electrode_current1", [0.002, -0.002]))
    setup_electrode_pair2(session, pair2, case.get("electrode_current2", [0.002, -0.002]))
    mesh1_path, mesh2_path = run_tdcs_simulation(session, str(baseline_workspace.workspace_dir), str(replay_dir), int(case.get("n_workers", 1)))
    ti_mesh_path = Path(calculate_ti(mesh1_path, mesh2_path, str(replay_dir)))
    return Path(export_ti_to_nifti(str(ti_mesh_path), str(replay_dir), subject.reference_t1, "max_TI", f"{subject.id}_{case['name']}_seed_{seed}_replay"))


def _run_inverse_preset_worker(
    subject: SubjectConfig,
    preset: str,
    inverse_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    command: str,
) -> list[dict[str, Any]]:
    """
    鎵ц鍗曚釜 `(subject, preset)` 鐨?inverse 闃舵銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    inverse_cases : list[dict[str, Any]]
        inverse case 鍒楄〃銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        inverse 缁撴灉琛屻€?
    """
    rows: list[dict[str, Any]] = []
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}
    paths = preset_paths_for(work_root, subject.id, preset)
    for case in inverse_cases:
        for seed in case.get("seeds", []):
            payload = {"subject_id": subject.id, "case_name": case["name"], "preset": preset, "seed": seed}
            stage_dir = paths.inverse_root / case["name"] / "seeds" / str(seed)

            def runner() -> dict[str, Any]:
                start = time.perf_counter()
                workspace = ensure_workspace(
                    workspace_cache=workspace_cache,
                    subject=subject,
                    preset=preset,
                    work_root=work_root,
                    preset_ini_paths=preset_ini_paths,
                    debug_mesh=debug_mesh,
                )
                output_dir = _prepare_stage_artifacts_dir(stage_dir)
                mesh_file = str(workspace.mesh_file)
                opt = init_optimization(str(workspace.workspace_dir), str(output_dir), mesh_file)
                opt.seed = int(seed)
                configure_inverse_case(
                    opt,
                    case,
                    mesh_file,
                    resolve_workspace_eeg_cap(workspace, case.get("net_electrode_file", EEG10_20_EXTENDED_SPM12)),
                )
                result_dir = Path(run_optimization(opt, int(case.get("n_workers", 1))))
                summary = parse_inverse_summary(result_dir / "detailed_results" / "summary.hdf5")
                mapping_entries = load_mapping(result_dir / "electrode_mapping.json")
                mapped_mesh = result_dir / "mapped_electrodes_simulation" / f"{subject.id}_tes_mapped_opt_head_mesh.msh"
                mapped_ti_volume = Path(export_ti_to_nifti(str(mapped_mesh), str(result_dir), subject.reference_t1, "max_TI", f"{subject.id}_{case['name']}_seed_{seed}_mapped"))
                elapsed = time.perf_counter() - start
                return {
                    **payload,
                    "status": "ok",
                    "elapsed_seconds": elapsed,
                    "optimizer": summary["optimizer"],
                    "optimizer_fopt": summary["fopt"],
                    "optimizer_n_test": summary["n_test"],
                    "optimizer_n_sim": summary["n_sim"],
                    "mapped_mesh_path": str(mapped_mesh),
                    "mapped_ti_volume_path": str(mapped_ti_volume),
                    "electrode_mapping_path": str(result_dir / "electrode_mapping.json"),
                    "mapped_labels": "|".join(entry["mapped_label"] for entry in mapping_entries),
                    "mapping_distance_mean_mm": float(np.mean([entry["mapping_distance_mm"] for entry in mapping_entries])),
                }

            rows.append(
                _execute_stage_task(
                    stage="inverse",
                    result_path=inverse_result_path(paths, case["name"], int(seed)),
                    run_path=inverse_run_path(paths, case["name"], int(seed)),
                    command=command,
                    payload=payload,
                    default_failure_stage="inverse_execution_failed",
                    runner=runner,
                )
            )
    return rows


def run_inverse_validation(
    subjects: list[SubjectConfig],
    presets: list[str],
    inverse_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    preset_workers: int,
    command: str,
) -> list[dict[str, Any]]:
    """
    杩愯 inverse 楠岃瘉銆?

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 鍒楄〃銆?
    presets : list[str]
        preset 鍒楄〃銆?
    inverse_cases : list[dict[str, Any]]
        inverse case 鍒楄〃銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    preset_workers : int
        preset 骞惰杩涚▼鏁般€?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        inverse 缁撴灉琛屻€?
    """
    active_presets = [preset for preset in presets if preset in INVERSE_PRESET_ORDER]
    task_args = [(subject, preset, inverse_cases, work_root, preset_ini_paths, debug_mesh, command) for subject in subjects for preset in active_presets]
    return collect_parallel_rows(task_args, _run_inverse_preset_worker, preset_workers)


def _build_replay_success_result(
    subject: SubjectConfig,
    preset: str,
    case: dict[str, Any],
    seed: int,
    elapsed_seconds: float,
    replay_ti_volume: Path,
    replay_metrics: dict[str, float],
    inverse_result: dict[str, Any],
    baseline_replay: dict[str, Any] | None,
    baseline_inverse: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    鏋勯€?replay 鎴愬姛缁撴灉銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    case : dict[str, Any]
        inverse case 閰嶇疆銆?
    seed : int
        闅忔満绉嶅瓙銆?
    elapsed_seconds : float
        鎵ц鑰楁椂銆?
    replay_ti_volume : Path
        replay TI 璺緞銆?
    replay_metrics : dict[str, float]
        replay 鎸囨爣銆?
    inverse_result : dict[str, Any]
        褰撳墠 preset inverse 缁撴灉銆?
    baseline_replay : dict[str, Any] or None
        M0 replay 鍩虹嚎缁撴灉銆?
    baseline_inverse : dict[str, Any] or None
        M0 inverse 鍩虹嚎缁撴灉銆?

    Returns
    -------
    dict[str, Any]
        replay 缁撴灉銆?
    """
    payload = {
        "subject_id": subject.id,
        "case_name": case["name"],
        "preset": preset,
        "seed": seed,
        "status": "ok",
        "elapsed_seconds": elapsed_seconds,
        "replay_ti_volume_path": str(replay_ti_volume),
    }
    payload["source_inverse_result_path"] = str(inverse_result.get("result_path", ""))
    payload.update(replay_metrics)
    if preset == "M0":
        payload.update(
            {
                "comparison_status": "baseline_reference",
                "comparison_reason": None,
                "goal_gap_on_m0": 0.0,
                "label_consistent": True,
                "optimized_center_drift_mean_mm": 0.0,
                "optimized_center_drift_max_mm": 0.0,
                "mapped_center_drift_mean_mm": 0.0,
                "mapped_center_drift_max_mm": 0.0,
                "inverse_pass": True,
            }
        )
        return payload
    if baseline_replay is None or baseline_inverse is None:
        payload.update(
            {
                "comparison_status": "baseline_unavailable",
                "comparison_reason": "M0 replay baseline is unavailable",
                "goal_gap_on_m0": None,
                "label_consistent": None,
                "optimized_center_drift_mean_mm": None,
                "optimized_center_drift_max_mm": None,
                "mapped_center_drift_mean_mm": None,
                "mapped_center_drift_max_mm": None,
                "inverse_pass": None,
            }
        )
        return payload
    baseline_mapping = load_mapping(Path(str(baseline_inverse["electrode_mapping_path"])))
    current_mapping = load_mapping(Path(str(inverse_result["electrode_mapping_path"])))
    payload.update(
        {
            "comparison_status": "ok",
            "comparison_reason": None,
            "goal_gap_on_m0": relative_error(float(replay_metrics["replay_goal"]), float(baseline_replay["replay_goal"])),
            **compute_mapping_drift(baseline_mapping, current_mapping),
        }
    )
    if preset in INVERSE_THRESHOLDS and payload["goal_gap_on_m0"] is not None:
        payload["inverse_pass"] = bool(payload["goal_gap_on_m0"] <= INVERSE_THRESHOLDS[preset])
    else:
        payload["inverse_pass"] = None
    return payload


def _run_single_replay_result(
    subject: SubjectConfig,
    preset: str,
    case: dict[str, Any],
    seed: int,
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    command: str,
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult],
    wait_on_inverse: bool,
) -> dict[str, Any]:
    """
    鎵ц鍗曚釜 replay 缁撴灉鐢熸垚銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    case : dict[str, Any]
        inverse case 閰嶇疆銆?
    seed : int
        闅忔満绉嶅瓙銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    command : str
        褰撳墠鍛戒护琛屻€?
    workspace_cache : dict[tuple[str, str], WorkspaceBuildResult]
        workspace 缂撳瓨銆?
    wait_on_inverse : bool
        褰撳墠 preset inverse 鏄惁鍏佽绛夊緟銆?

    Returns
    -------
    dict[str, Any]
        replay 缁撴灉銆?
    """
    paths = preset_paths_for(work_root, subject.id, preset)
    payload = {"subject_id": subject.id, "case_name": case["name"], "preset": preset, "seed": seed}

    def runner() -> dict[str, Any]:
        start = time.perf_counter()
        baseline_workspace = _ensure_m0_workspace_ready(
            subject=subject,
            work_root=work_root,
            preset_ini_paths=preset_ini_paths,
            debug_mesh=debug_mesh,
            workspace_cache=workspace_cache,
        )
        inverse_result = _load_inverse_result(
            subject=subject,
            preset=preset,
            case_name=case["name"],
            seed=seed,
            work_root=work_root,
            wait_on_running=wait_on_inverse,
        )
        inverse_result["result_path"] = str(inverse_result_path(paths, case["name"], seed))
        mapped_entries = load_mapping(Path(str(inverse_result["electrode_mapping_path"])))
        replay_ti_volume = run_replay_on_m0(
            subject=subject,
            preset=preset,
            case=case,
            seed=seed,
            mapped_entries=mapped_entries,
            work_root=work_root,
            preset_ini_paths=preset_ini_paths,
            debug_mesh=debug_mesh,
            workspace_cache=workspace_cache,
        )
        reference_img = nib.load(subject.reference_t1)
        replay_metrics = compute_goal_from_volume(
            load_volume_data(replay_ti_volume, reference_img),
            reference_img,
            str(baseline_workspace.workspace_dir),
            case,
            replay_ti_volume.parent / "_mask_cache",
        )
        baseline_replay: dict[str, Any] | None = None
        baseline_inverse: dict[str, Any] | None = None
        if preset != "M0":
            baseline_replay = _load_replay_result(
                subject=subject,
                preset="M0",
                case_name=case["name"],
                seed=seed,
                work_root=work_root,
                wait_on_running=True,
            )
            baseline_inverse = _load_inverse_result(
                subject=subject,
                preset="M0",
                case_name=case["name"],
                seed=seed,
                work_root=work_root,
                wait_on_running=True,
            )
        elapsed = time.perf_counter() - start
        return _build_replay_success_result(
            subject=subject,
            preset=preset,
            case=case,
            seed=seed,
            elapsed_seconds=elapsed,
            replay_ti_volume=replay_ti_volume,
            replay_metrics=replay_metrics,
            inverse_result=inverse_result,
            baseline_replay=baseline_replay,
            baseline_inverse=baseline_inverse,
        )

    return _execute_stage_task(
        stage="replay",
        result_path=replay_result_path(paths, case["name"], seed),
        run_path=replay_run_path(paths, case["name"], seed),
        command=command,
        payload=payload,
        default_failure_stage="replay_execution_failed",
        runner=runner,
    )


def _ensure_m0_replay_baseline(
    subject: SubjectConfig,
    case: dict[str, Any],
    seed: int,
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    command: str,
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult],
) -> dict[str, Any]:
    """
    纭繚 M0 replay 鍩虹嚎瀛樺湪銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    case : dict[str, Any]
        inverse case 閰嶇疆銆?
    seed : int
        闅忔満绉嶅瓙銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    command : str
        褰撳墠鍛戒护琛屻€?
    workspace_cache : dict[tuple[str, str], WorkspaceBuildResult]
        workspace 缂撳瓨銆?

    Returns
    -------
    dict[str, Any]
        M0 replay 缁撴灉銆?
    """
    _ensure_m0_workspace_ready(
        subject=subject,
        work_root=work_root,
        preset_ini_paths=preset_ini_paths,
        debug_mesh=debug_mesh,
        workspace_cache=workspace_cache,
    )
    _load_inverse_result(
        subject=subject,
        preset="M0",
        case_name=case["name"],
        seed=seed,
        work_root=work_root,
        wait_on_running=True,
    )
    paths = preset_paths_for(work_root, subject.id, "M0")
    run_path = replay_run_path(paths, case["name"], seed)
    result_path = replay_result_path(paths, case["name"], seed)
    run_payload = read_json(run_path)
    if run_payload is not None:
        state = str(run_payload.get("state"))
        if state == RUN_STATE_RUNNING:
            return _load_replay_result(subject, "M0", case["name"], seed, work_root, wait_on_running=True)
        if state == RUN_STATE_COMPLETED:
            return _load_replay_result(subject, "M0", case["name"], seed, work_root, wait_on_running=False)
    if result_path.exists() and run_payload is None:
        raise RuntimeError(f"妫€娴嬪埌鏃犵姸鎬?replay 缁撴灉锛屾嫆缁濈户缁? {result_path}")
    return _run_single_replay_result(
        subject=subject,
        preset="M0",
        case=case,
        seed=seed,
        work_root=work_root,
        preset_ini_paths=preset_ini_paths,
        debug_mesh=debug_mesh,
        command=command,
        workspace_cache=workspace_cache,
        wait_on_inverse=True,
    )


def _run_replay_preset_worker(
    subject: SubjectConfig,
    preset: str,
    inverse_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    command: str,
) -> list[dict[str, Any]]:
    """
    鎵ц鍗曚釜 `(subject, preset)` 鐨?replay 闃舵銆?

    Parameters
    ----------
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    inverse_cases : list[dict[str, Any]]
        inverse case 鍒楄〃銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        replay 缁撴灉琛屻€?
    """
    rows: list[dict[str, Any]] = []
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}
    for case in inverse_cases:
        for seed in case.get("seeds", []):
            rows.append(
                _run_single_replay_result(
                    subject=subject,
                    preset=preset,
                    case=case,
                    seed=int(seed),
                    work_root=work_root,
                    preset_ini_paths=preset_ini_paths,
                    debug_mesh=debug_mesh,
                    command=command,
                    workspace_cache=workspace_cache,
                    wait_on_inverse=False,
                )
            )
    return rows


def run_replay_validation(
    subjects: list[SubjectConfig],
    presets: list[str],
    inverse_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    preset_workers: int,
    command: str,
) -> list[dict[str, Any]]:
    """
    杩愯 replay 楠岃瘉銆?

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 鍒楄〃銆?
    presets : list[str]
        preset 鍒楄〃銆?
    inverse_cases : list[dict[str, Any]]
        inverse case 鍒楄〃銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?
    preset_workers : int
        preset 骞惰杩涚▼鏁般€?
    command : str
        褰撳墠鍛戒护琛屻€?

    Returns
    -------
    list[dict[str, Any]]
        replay 缁撴灉琛屻€?
    """
    baseline_rows: list[dict[str, Any]] = []
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}
    for subject in subjects:
        _ensure_m0_workspace_ready(
            subject=subject,
            work_root=work_root,
            preset_ini_paths=preset_ini_paths,
            debug_mesh=debug_mesh,
            workspace_cache=workspace_cache,
        )
        for case in inverse_cases:
            for seed in case.get("seeds", []):
                baseline_result = _ensure_m0_replay_baseline(
                    subject=subject,
                    case=case,
                    seed=int(seed),
                    work_root=work_root,
                    preset_ini_paths=preset_ini_paths,
                    debug_mesh=debug_mesh,
                    command=command,
                    workspace_cache=workspace_cache,
                )
                if "M0" in presets:
                    baseline_rows.append(baseline_result)
    candidate_presets = [preset for preset in presets if preset != "M0"]
    task_args = [(subject, preset, inverse_cases, work_root, preset_ini_paths, debug_mesh, command) for subject in subjects for preset in candidate_presets]
    parallel_rows = collect_parallel_rows(task_args, _run_replay_preset_worker, preset_workers)
    return sort_result_rows(baseline_rows + parallel_rows)


def ensure_workspace(
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult],
    subject: SubjectConfig,
    preset: str,
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
) -> WorkspaceBuildResult:
    """
    鑾峰彇鎴栧噯澶?workspace銆?

    Parameters
    ----------
    workspace_cache : dict[tuple[str, str], WorkspaceBuildResult]
        workspace 缂撳瓨銆?
    subject : SubjectConfig
        subject 閰嶇疆銆?
    preset : str
        preset 鍚嶇О銆?
    work_root : Path
        宸ヤ綔鏍圭洰褰曘€?
    preset_ini_paths : dict[str, Path]
        preset ini 璺緞鏄犲皠銆?
    debug_mesh : bool
        鏄惁淇濈暀 mesh 璋冭瘯杈撳嚭銆?

    Returns
    -------
    WorkspaceBuildResult
        workspace 缁撴灉銆?
    """
    key = (subject.id, preset)
    if key not in workspace_cache:
        try:
            workspace_cache[key] = load_existing_workspace(subject, preset, work_root)
        except WorkspacePreparationError:
            workspace_cache[key] = prepare_workspace(subject, preset, work_root, preset_ini_paths[preset], debug_mesh)
    return workspace_cache[key]


def relative_error(value: float, baseline: float) -> float:
    """
    璁＄畻鐩稿璇樊銆?

    Parameters
    ----------
    value : float
        褰撳墠鍊笺€?
    baseline : float
        鍩虹嚎鍊笺€?

    Returns
    -------
    float
        鐩稿璇樊銆?
    """
    if baseline in (0, 0.0) or math.isnan(baseline):
        return math.nan
    return float(abs(value - baseline) / abs(baseline))


def summarize_mesh(mesh_file: Path) -> dict[str, float]:
    """
    缁熻 mesh 瑙勬ā銆?

    Parameters
    ----------
    mesh_file : Path
        mesh 璺緞銆?

    Returns
    -------
    dict[str, float]
        mesh 缁熻缁撴灉銆?
    """
    mesh = mesh_io.read_msh(str(mesh_file))
    tetra_mask = mesh.elm.get_tetrahedra()
    tetra_tags = mesh.elm.tag1[tetra_mask]
    volumes = mesh.elements_volumes_and_areas()[tetra_mask]
    stats: dict[str, float] = {
        "node_count": float(mesh.nodes.nr),
        "tetra_count": float(np.sum(tetra_mask)),
        "mesh_file": str(mesh_file),
    }
    for tag in np.unique(tetra_tags):
        stats[f"tag_volume_{int(tag)}"] = float(np.sum(volumes[tetra_tags == tag]))
    return stats


def scalp_surface_points(mesh_file: Path) -> np.ndarray:
    """
    鎻愬彇澶寸毊澶栬〃闈㈢偣闆嗐€?

    Parameters
    ----------
    mesh_file : Path
        mesh 璺緞銆?

    Returns
    -------
    np.ndarray
        琛ㄩ潰鐐瑰潗鏍囥€?
    """
    mesh = mesh_io.read_msh(str(mesh_file))
    _, vertices_out, _, _ = mesh.partition_skin_surface(label_skin=ElementTags.SCALP_TH_SURFACE)
    return mesh.nodes.node_coord[vertices_out - 1]


def bidirectional_surface_distance(reference_points: np.ndarray, candidate_points: np.ndarray) -> dict[str, float]:
    """
    璁＄畻鍙屽悜琛ㄩ潰璺濈銆?

    Parameters
    ----------
    reference_points : np.ndarray
        鍙傝€冪偣闆嗐€?
    candidate_points : np.ndarray
        鍊欓€夌偣闆嗐€?

    Returns
    -------
    dict[str, float]
        mean distance 鍜?95% Hausdorff銆?
    """
    ref_tree = cKDTree(reference_points)
    cand_tree = cKDTree(candidate_points)
    dist_ref = ref_tree.query(candidate_points)[0]
    dist_cand = cand_tree.query(reference_points)[0]
    merged = np.concatenate([dist_ref, dist_cand])
    return {
        "scalp_mean_distance_mm": float(np.mean(merged)),
        "scalp_hausdorff95_mm": float(np.percentile(merged, 95)),
    }
