"""
Mesh validation 阶段执行逻辑。
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

from neuracle.mesh_validation.mesh_validation_workspace import (
    MeshGenerationError,
    SubjectConfig,
    WorkspaceBuildResult,
    WorkspacePreparationError,
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


def sort_result_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    对结果行做稳定排序。

    Parameters
    ----------
    rows : list[dict[str, Any]]
        结果行列表。

    Returns
    -------
    list[dict[str, Any]]
        排序后的结果行列表。
    """
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("subject_id", "")),
            str(row.get("case_name", "")),
            int(row.get("seed", -1)) if row.get("seed") is not None else -1,
            str(row.get("preset", "")),
            str(row.get("roi_name", "")),
            str(row.get("status", "")),
        ),
    )


def run_with_failure_capture(default_failure_stage: str, payload: dict[str, Any], fn: Any) -> list[dict[str, Any]]:
    """
    执行任务并将异常转换为失败结果行。

    Parameters
    ----------
    default_failure_stage : str
        默认失败阶段名称。
    payload : dict[str, Any]
        失败行需要保留的基础字段。
    fn : Any
        执行函数。

    Returns
    -------
    list[dict[str, Any]]
        执行结果行列表。
    """
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        failure_stage = default_failure_stage
        if isinstance(exc, MeshGenerationError):
            failure_stage = "mesh_generation_failed"
        elif isinstance(exc, WorkspacePreparationError):
            failure_stage = "workspace_prepare_failed"
        LOGGER.exception("%s 执行失败: %s", failure_stage, payload)
        return [
            {
                **payload,
                "status": "failed",
                "failure_stage": failure_stage,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }
        ]
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def collect_parallel_rows(task_args: list[tuple[Any, ...]], worker_fn: Any, max_workers: int) -> list[dict[str, Any]]:
    """
    使用 spawn 进程池执行 worker。

    Parameters
    ----------
    task_args : list[tuple[Any, ...]]
        每个 worker 的参数元组。
    worker_fn : Any
        模块级 worker 入口。
    max_workers : int
        最大并行进程数。

    Returns
    -------
    list[dict[str, Any]]
        聚合后的结果行。
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


def _run_mesh_preset_worker(
    subject: SubjectConfig,
    preset: str,
    preset_ini_paths: dict[str, Path],
    work_root: Path,
    debug_mesh: bool,
) -> list[dict[str, Any]]:
    """
    执行单个 `(subject, preset)` 的 mesh 阶段。

    Parameters
    ----------
    subject : SubjectConfig
        subject 配置。
    preset : str
        preset 名称。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    work_root : Path
        工作根目录。
    debug_mesh : bool
        是否保留 mesh 调试输出。

    Returns
    -------
    list[dict[str, Any]]
        mesh 结果行。
    """
    payload = {"subject_id": subject.id, "preset": preset}
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

    return run_with_failure_capture("mesh_generation_failed", payload, runner)


def build_mesh_variants(
    subjects: list[SubjectConfig],
    presets: list[str],
    preset_ini_paths: dict[str, Path],
    work_root: Path,
    debug_mesh: bool,
    preset_workers: int,
) -> list[dict[str, Any]]:
    """
    构建 workspace 并统计 mesh 指标。

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 列表。
    presets : list[str]
        preset 列表。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    work_root : Path
        工作根目录。
    debug_mesh : bool
        是否保留 mesh 调试输出。
    preset_workers : int
        preset 并行进程数。

    Returns
    -------
    list[dict[str, Any]]
        mesh 结果行。
    """
    task_args = [(subject, preset, preset_ini_paths, work_root, debug_mesh) for subject in subjects for preset in presets]
    return collect_parallel_rows(task_args, _run_mesh_preset_worker, preset_workers)


def build_spec_mask(spec: dict[str, Any], reference_img: nib.Nifti1Image, subject_dir: str, cache_dir: Path, cache_key: str) -> np.ndarray:
    """
    构建 ROI 掩码。

    Parameters
    ----------
    spec : dict[str, Any]
        ROI 配置。
    reference_img : nib.Nifti1Image
        参考图像。
    subject_dir : str
        subject 目录。
    cache_dir : Path
        缓存目录。
    cache_key : str
        缓存键。

    Returns
    -------
    np.ndarray
        布尔掩码。
    """
    spec_type = spec.get("type", "sphere")
    if spec_type == "sphere":
        ijk = np.indices(reference_img.shape, dtype=float).reshape(3, -1).T
        xyz = nib.affines.apply_affine(reference_img.affine, ijk)
        center = np.asarray(spec["center"], dtype=float)
        radius = float(spec["radius"])
        return (np.linalg.norm(xyz - center, axis=1) <= radius).reshape(reference_img.shape)
    if spec_type != "mask":
        raise ValueError(f"不支持的 ROI 类型: {spec_type}")
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


def load_volume_data(volume_path: Path, reference_img: nib.Nifti1Image) -> np.ndarray:
    """
    读取并对齐体积数据。

    Parameters
    ----------
    volume_path : Path
        体积路径。
    reference_img : nib.Nifti1Image
        参考图像。

    Returns
    -------
    np.ndarray
        对齐后的体积数据。
    """
    image = nib.squeeze_image(nib.load(str(volume_path)))
    reference_img = nib.squeeze_image(reference_img)
    if image.shape != reference_img.shape or not np.allclose(image.affine, reference_img.affine):
        image = resample_from_to(image, reference_img, order=0)
    return np.asarray(image.get_fdata())


def roi_statistics(values: np.ndarray) -> dict[str, float]:
    """
    计算 ROI 统计量。

    Parameters
    ----------
    values : np.ndarray
        ROI 内体素值。

    Returns
    -------
    dict[str, float]
        ROI 统计结果。
    """
    if values.size == 0:
        return {"roi_mean": math.nan, "roi_median": math.nan, "roi_p95": math.nan, "roi_p99": math.nan, "roi_max": math.nan}
    return {
        "roi_mean": float(np.mean(values)),
        "roi_median": float(np.median(values)),
        "roi_p95": float(np.percentile(values, 95)),
        "roi_p99": float(np.percentile(values, 99)),
        "roi_max": float(np.max(values)),
    }


def compute_forward_case_metrics(
    ti_volume_path: Path,
    subject: SubjectConfig,
    subject_dir: str,
    case: dict[str, Any],
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """
    计算正向 case 的 ROI 指标。

    Parameters
    ----------
    ti_volume_path : Path
        TI 体积路径。
    subject : SubjectConfig
        subject 配置。
    subject_dir : str
        当前 workspace 目录。
    case : dict[str, Any]
        forward case 配置。
    cache_dir : Path
        掩码缓存目录。

    Returns
    -------
    list[dict[str, Any]]
        指标结果行。
    """
    reference_img = nib.load(subject.reference_t1)
    ti_data = load_volume_data(ti_volume_path, reference_img)
    hotspot_mask = build_spec_mask(case["hotspot_roi"], reference_img, subject_dir, cache_dir, f"{subject.id}_{case['name']}_hotspot")
    hotspot_value = float(np.max(ti_data[hotspot_mask])) if np.any(hotspot_mask) else math.nan
    rows: list[dict[str, Any]] = []
    for roi_index, roi_spec in enumerate(case["roi_specs"]):
        roi_name = roi_spec.get("name", f"roi_{roi_index}")
        roi_mask = build_spec_mask(roi_spec, reference_img, subject_dir, cache_dir, f"{subject.id}_{case['name']}_{roi_name}")
        stats = roi_statistics(ti_data[roi_mask])
        rows.append({"subject_id": subject.id, "case_name": case["name"], "roi_name": roi_name, "hotspot_value": hotspot_value, **stats})
    return rows


def _run_forward_preset_worker(
    subject: SubjectConfig,
    preset: str,
    forward_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
) -> list[dict[str, Any]]:
    """
    执行单个 `(subject, preset)` 的 forward 阶段。

    Parameters
    ----------
    subject : SubjectConfig
        subject 配置。
    preset : str
        preset 名称。
    forward_cases : list[dict[str, Any]]
        forward case 列表。
    work_root : Path
        工作根目录。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。

    Returns
    -------
    list[dict[str, Any]]
        forward 结果行。
    """
    rows: list[dict[str, Any]] = []
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}
    for case in forward_cases:
        payload = {"subject_id": subject.id, "case_name": case["name"], "preset": preset}

        def runner() -> list[dict[str, Any]]:
            start = time.perf_counter()
            workspace = ensure_workspace(
                workspace_cache=workspace_cache,
                subject=subject,
                preset=preset,
                work_root=work_root,
                preset_ini_paths=preset_ini_paths,
                debug_mesh=debug_mesh,
            )
            output_dir = workspace.paths.forward_dir / case["name"]
            reset_output_dir(output_dir)
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
            metric_rows = compute_forward_case_metrics(
                ti_volume_path=ti_nifti_path,
                subject=subject,
                subject_dir=str(workspace.workspace_dir),
                case=case,
                cache_dir=output_dir / "_mask_cache",
            )
            elapsed = time.perf_counter() - start
            for row in metric_rows:
                row.update(
                    {
                        **payload,
                        "status": "ok",
                        "elapsed_seconds": elapsed,
                        "ti_mesh_path": str(ti_mesh_path),
                        "ti_volume_path": str(ti_nifti_path),
                        "final_labels_path": str(workspace.final_tissues_path),
                    }
                )
            return metric_rows

        rows.extend(run_with_failure_capture("forward_execution_failed", payload, runner))
    return rows


def run_forward_validation(
    subjects: list[SubjectConfig],
    presets: list[str],
    forward_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    preset_workers: int,
) -> list[dict[str, Any]]:
    """
    运行正向 TI 验证。

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 列表。
    presets : list[str]
        preset 列表。
    forward_cases : list[dict[str, Any]]
        forward case 列表。
    work_root : Path
        工作根目录。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。
    preset_workers : int
        preset 并行进程数。

    Returns
    -------
    list[dict[str, Any]]
        forward 结果行。
    """
    task_args = [(subject, preset, forward_cases, work_root, preset_ini_paths, debug_mesh) for subject in subjects for preset in presets]
    return collect_parallel_rows(task_args, _run_forward_preset_worker, preset_workers)


def read_hdf5_scalar(dataset: Any) -> Any:
    """
    读取 HDF5 标量。

    Parameters
    ----------
    dataset : Any
        HDF5 数据集。

    Returns
    -------
    Any
        标量值。
    """
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def parse_inverse_summary(summary_path: Path) -> dict[str, Any]:
    """
    解析优化 summary.hdf5。

    Parameters
    ----------
    summary_path : Path
        summary 路径。

    Returns
    -------
    dict[str, Any]
        摘要结果。
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
    读取 electrode mapping。

    Parameters
    ----------
    mapping_path : Path
        mapping 路径。

    Returns
    -------
    list[dict[str, Any]]
        规范化后的 mapping 条目。
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
    向优化对象添加 ROI。

    Parameters
    ----------
    opt : Any
        优化对象。
    spec : dict[str, Any]
        ROI 配置。
    mesh_file : str
        mesh 路径。
    default_difference : bool, optional
        是否默认作为 difference ROI。

    Returns
    -------
    None
    """
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
    配置 inverse case。

    Parameters
    ----------
    opt : Any
        优化对象。
    case : dict[str, Any]
        inverse case 配置。
    mesh_file : str
        mesh 路径。
    net_file : str
        电极网文件路径。

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
    if case["goal"] in {"focality", "focality_inv"}:
        add_inverse_roi(opt, case.get("non_roi", case["roi"]), mesh_file=mesh_file, default_difference=True)


def compute_goal_from_volume(
    ti_data: np.ndarray,
    reference_img: nib.Nifti1Image,
    subject_dir: str,
    case: dict[str, Any],
    cache_dir: Path,
) -> dict[str, float]:
    """
    根据 TI 体积重算目标值。

    Parameters
    ----------
    ti_data : np.ndarray
        TI 数据。
    reference_img : nib.Nifti1Image
        参考图像。
    subject_dir : str
        subject 目录。
    case : dict[str, Any]
        inverse case 配置。
    cache_dir : Path
        缓存目录。

    Returns
    -------
    dict[str, float]
        replay 指标结果。
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
    if case["goal"] in {"focality", "focality_inv"}:
        non_roi_mask = np.logical_and(non_roi_mask, np.logical_not(roi_mask))
    non_roi_values = ti_data[non_roi_mask]
    roc_value = float(ROC(roi_values, non_roi_values, case.get("threshold", [0.1, 0.2]), focal=case["goal"] == "focality"))
    result["replay_non_roi_mean"] = float(np.mean(non_roi_values))
    result["replay_roc"] = roc_value
    result["replay_goal"] = float(-100 * (math.sqrt(2) - roc_value)) if case["goal"] == "focality" else float(-100 * roc_value)
    return result


def group_labels_by_channel(entries: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """
    从映射结果恢复两对电极标签。

    Parameters
    ----------
    entries : list[dict[str, Any]]
        映射条目。

    Returns
    -------
    tuple[list[str], list[str]]
        两对电极标签。
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(entry["channel"], []).append(entry)
    pair1 = [item["mapped_label"] for item in sorted(grouped.get(0, []), key=lambda item: item["array"])]
    pair2 = [item["mapped_label"] for item in sorted(grouped.get(1, []), key=lambda item: item["array"])]
    return pair1, pair2


def compute_mapping_drift(baseline_entries: list[dict[str, Any]], candidate_entries: list[dict[str, Any]]) -> dict[str, float | bool]:
    """
    计算映射漂移指标。

    Parameters
    ----------
    baseline_entries : list[dict[str, Any]]
        M0 基线映射。
    candidate_entries : list[dict[str, Any]]
        当前映射。

    Returns
    -------
    dict[str, float | bool]
        漂移指标。
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
    在 M0 workspace 上回放映射后的电极。

    Parameters
    ----------
    subject : SubjectConfig
        subject 配置。
    preset : str
        来源 preset。
    case : dict[str, Any]
        inverse case 配置。
    seed : int
        随机种子。
    mapped_entries : list[dict[str, Any]]
        映射结果。
    work_root : Path
        工作根目录。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。
    workspace_cache : dict[tuple[str, str], WorkspaceBuildResult]
        workspace 缓存。

    Returns
    -------
    Path
        replay TI NIfTI 路径。
    """
    replay_dir = work_root / subject.id / preset / "inverse" / case["name"] / f"seed_{seed}" / "replay_on_m0"
    reset_output_dir(replay_dir)
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
) -> list[dict[str, Any]]:
    """
    执行单个 `(subject, preset)` 的 inverse 阶段。

    Parameters
    ----------
    subject : SubjectConfig
        subject 配置。
    preset : str
        preset 名称。
    inverse_cases : list[dict[str, Any]]
        inverse case 列表。
    work_root : Path
        工作根目录。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。

    Returns
    -------
    list[dict[str, Any]]
        inverse 结果行。
    """
    rows: list[dict[str, Any]] = []
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}
    for case in inverse_cases:
        for seed in case.get("seeds", []):
            payload = {"subject_id": subject.id, "case_name": case["name"], "preset": preset, "seed": seed}

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
                output_dir = workspace.paths.inverse_dir / case["name"] / f"seed_{seed}"
                reset_output_dir(output_dir)
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

            rows.extend(run_with_failure_capture("inverse_execution_failed", payload, runner))
    return rows


def run_inverse_validation(
    subjects: list[SubjectConfig],
    presets: list[str],
    inverse_cases: list[dict[str, Any]],
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
    preset_workers: int,
) -> list[dict[str, Any]]:
    """
    运行 inverse 验证。

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 列表。
    presets : list[str]
        preset 列表。
    inverse_cases : list[dict[str, Any]]
        inverse case 列表。
    work_root : Path
        工作根目录。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。
    preset_workers : int
        preset 并行进程数。

    Returns
    -------
    list[dict[str, Any]]
        inverse 结果行。
    """
    active_presets = [preset for preset in presets if preset in INVERSE_PRESET_ORDER]
    task_args = [(subject, preset, inverse_cases, work_root, preset_ini_paths, debug_mesh) for subject in subjects for preset in active_presets]
    return collect_parallel_rows(task_args, _run_inverse_preset_worker, preset_workers)


def ensure_workspace(
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult],
    subject: SubjectConfig,
    preset: str,
    work_root: Path,
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
) -> WorkspaceBuildResult:
    """
    获取或准备 workspace。

    Parameters
    ----------
    workspace_cache : dict[tuple[str, str], WorkspaceBuildResult]
        workspace 缓存。
    subject : SubjectConfig
        subject 配置。
    preset : str
        preset 名称。
    work_root : Path
        工作根目录。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。

    Returns
    -------
    WorkspaceBuildResult
        workspace 结果。
    """
    key = (subject.id, preset)
    if key not in workspace_cache:
        workspace_cache[key] = prepare_workspace(subject, preset, work_root, preset_ini_paths[preset], debug_mesh)
    return workspace_cache[key]


def relative_error(value: float, baseline: float) -> float:
    """
    计算相对误差。

    Parameters
    ----------
    value : float
        当前值。
    baseline : float
        基线值。

    Returns
    -------
    float
        相对误差。
    """
    if baseline in (0, 0.0) or math.isnan(baseline):
        return math.nan
    return float(abs(value - baseline) / abs(baseline))


def summarize_mesh(mesh_file: Path) -> dict[str, float]:
    """
    统计 mesh 规模。

    Parameters
    ----------
    mesh_file : Path
        mesh 路径。

    Returns
    -------
    dict[str, float]
        mesh 统计结果。
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
    提取头皮外表面点集。

    Parameters
    ----------
    mesh_file : Path
        mesh 路径。

    Returns
    -------
    np.ndarray
        表面点坐标。
    """
    mesh = mesh_io.read_msh(str(mesh_file))
    _, vertices_out, _, _ = mesh.partition_skin_surface(label_skin=ElementTags.SCALP_TH_SURFACE)
    return mesh.nodes.node_coord[vertices_out - 1]


def bidirectional_surface_distance(reference_points: np.ndarray, candidate_points: np.ndarray) -> dict[str, float]:
    """
    计算双向表面距离。

    Parameters
    ----------
    reference_points : np.ndarray
        参考点集。
    candidate_points : np.ndarray
        候选点集。

    Returns
    -------
    dict[str, float]
        mean distance 和 95% Hausdorff。
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


def reset_output_dir(path: Path) -> None:
    """
    重建阶段输出目录。

    Parameters
    ----------
    path : Path
        输出目录。

    Returns
    -------
    None
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
