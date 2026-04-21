"""
Mesh validation 报告聚合逻辑。
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuracle.mesh_validation.mesh_validation_stages import (
    FORWARD_THRESHOLDS,
    INVERSE_THRESHOLDS,
    bidirectional_surface_distance,
    compute_goal_from_volume,
    compute_mapping_drift,
    ensure_workspace,
    load_mapping,
    load_volume_data,
    relative_error,
    run_replay_on_m0,
    scalp_surface_points,
)
from neuracle.mesh_validation.mesh_validation_workspace import PRESET_ORDER, SubjectConfig, WorkspaceBuildResult
from simnibs.utils.mesh_element_properties import ElementTags

LOGGER = logging.getLogger("mesh_validation")


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """
    写出 CSV。

    Parameters
    ----------
    path : Path
        输出路径。
    rows : list[dict[str, Any]]
        结果行列表。

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    """
    写出 JSON。

    Parameters
    ----------
    path : Path
        输出路径。
    payload : Any
        输出内容。

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path, default: Any) -> Any:
    """
    读取 JSON，不存在时返回默认值。

    Parameters
    ----------
    path : Path
        JSON 路径。
    default : Any
        默认值。

    Returns
    -------
    Any
        JSON 内容或默认值。
    """
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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


def annotate_mesh_rows(mesh_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    为 mesh 结果补充与 M0 的表面距离比较。

    Parameters
    ----------
    mesh_rows : list[dict[str, Any]]
        mesh 结果行。

    Returns
    -------
    list[dict[str, Any]]
        补充后的 mesh 结果行。
    """
    baseline_points_by_subject: dict[str, np.ndarray] = {}
    for row in mesh_rows:
        if row.get("status") != "ok" or row.get("preset") != "M0":
            continue
        baseline_points_by_subject[str(row["subject_id"])] = scalp_surface_points(Path(row["mesh_file"]))
    for row in mesh_rows:
        if row.get("status") != "ok":
            continue
        baseline_points = baseline_points_by_subject.get(str(row["subject_id"]))
        if baseline_points is None:
            row["scalp_mean_distance_mm"] = None
            row["scalp_hausdorff95_mm"] = None
            continue
        current_points = scalp_surface_points(Path(row["mesh_file"]))
        row.update(bidirectional_surface_distance(baseline_points, current_points))
    return sort_rows(mesh_rows)


def mark_forward_baseline_unavailable(row: dict[str, Any]) -> None:
    """
    标记正向 comparison 不可用。

    Parameters
    ----------
    row : dict[str, Any]
        结果行。

    Returns
    -------
    None
    """
    row.update(
        {
            "comparison_status": "baseline_unavailable",
            "comparison_reason": "M0 baseline is unavailable for this subject/case/roi",
            "gm_pearson_r": None,
            "gm_nrmse": None,
            "baseline_hotspot_value": None,
            "roi_mean_rel_error": None,
            "roi_median_rel_error": None,
            "roi_p95_rel_error": None,
            "roi_p99_rel_error": None,
            "roi_max_rel_error": None,
            "hotspot_rel_error": None,
            "forward_pass": None,
        }
    )


def evaluate_forward_pass(row: dict[str, Any], preset: str) -> bool | None:
    """
    评估正向是否通过。

    Parameters
    ----------
    row : dict[str, Any]
        结果行。
    preset : str
        preset 名称。

    Returns
    -------
    bool or None
        是否通过，无法比较时返回 None。
    """
    if row.get("comparison_status") != "ok":
        return None
    if preset == "M0":
        return True
    if preset not in FORWARD_THRESHOLDS:
        return None
    threshold = FORWARD_THRESHOLDS[preset]
    checks = [
        row.get("roi_mean_rel_error", 0.0) <= threshold["mean_median"],
        row.get("roi_median_rel_error", 0.0) <= threshold["mean_median"],
        max(row.get("roi_p95_rel_error", 0.0), row.get("roi_p99_rel_error", 0.0), row.get("roi_max_rel_error", 0.0)) <= threshold["tail"],
        row.get("hotspot_rel_error", 0.0) <= threshold["hotspot"],
        row.get("gm_pearson_r", 1.0) >= 0.95,
        row.get("gm_nrmse", 0.0) <= threshold["nrmse"],
    ]
    return bool(all(checks))


def annotate_forward_rows(forward_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    为 forward 结果补充与 M0 的 comparison。

    Parameters
    ----------
    forward_rows : list[dict[str, Any]]
        forward 结果行。

    Returns
    -------
    list[dict[str, Any]]
        补充后的 forward 结果行。
    """
    baseline_by_key = {
        (str(row["subject_id"]), str(row["case_name"]), str(row["roi_name"])): row
        for row in forward_rows
        if row.get("status") == "ok" and row.get("preset") == "M0"
    }
    gm_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for row in forward_rows:
        if row.get("status") != "ok":
            continue
        if row.get("preset") == "M0":
            row["comparison_status"] = "baseline_reference"
            row["comparison_reason"] = None
            row["gm_pearson_r"] = 1.0
            row["gm_nrmse"] = 0.0
            row["baseline_hotspot_value"] = row.get("hotspot_value")
            row["roi_mean_rel_error"] = 0.0
            row["roi_median_rel_error"] = 0.0
            row["roi_p95_rel_error"] = 0.0
            row["roi_p99_rel_error"] = 0.0
            row["roi_max_rel_error"] = 0.0
            row["hotspot_rel_error"] = 0.0
            row["forward_pass"] = True
            continue
        baseline = baseline_by_key.get((str(row["subject_id"]), str(row["case_name"]), str(row["roi_name"])))
        if baseline is None:
            mark_forward_baseline_unavailable(row)
            continue
        gm_key = (str(baseline["ti_volume_path"]), str(row["ti_volume_path"]))
        if gm_key not in gm_cache:
            reference_img = nib.load(str(Path(row["ti_volume_path"])))
            current_ti = load_volume_data(Path(row["ti_volume_path"]), reference_img)
            baseline_ti = load_volume_data(Path(baseline["ti_volume_path"]), reference_img)
            current_labels = load_volume_data(Path(row["final_labels_path"]), reference_img)
            baseline_labels = load_volume_data(Path(baseline["final_labels_path"]), reference_img)
            gm_mask = (current_labels == ElementTags.GM) & (baseline_labels == ElementTags.GM)
            gm_cache[gm_key] = (current_ti[gm_mask], baseline_ti[gm_mask])
        current_gm, baseline_gm = gm_cache[gm_key]
        if current_gm.size > 1 and np.std(current_gm) > 0 and np.std(baseline_gm) > 0:
            pearson_r = float(np.corrcoef(current_gm, baseline_gm)[0, 1])
        else:
            pearson_r = math.nan
        if baseline_gm.size and np.linalg.norm(baseline_gm) > 0:
            nrmse = float(np.linalg.norm(current_gm - baseline_gm) / np.linalg.norm(baseline_gm))
        else:
            nrmse = math.nan
        row.update(
            {
                "comparison_status": "ok",
                "comparison_reason": None,
                "gm_pearson_r": pearson_r,
                "gm_nrmse": nrmse,
                "baseline_hotspot_value": baseline["hotspot_value"],
                "roi_mean_rel_error": relative_error(row["roi_mean"], baseline["roi_mean"]),
                "roi_median_rel_error": relative_error(row["roi_median"], baseline["roi_median"]),
                "roi_p95_rel_error": relative_error(row["roi_p95"], baseline["roi_p95"]),
                "roi_p99_rel_error": relative_error(row["roi_p99"], baseline["roi_p99"]),
                "roi_max_rel_error": relative_error(row["roi_max"], baseline["roi_max"]),
                "hotspot_rel_error": relative_error(row["hotspot_value"], baseline["hotspot_value"]),
            }
        )
        row["forward_pass"] = evaluate_forward_pass(row, str(row["preset"]))
    return sort_rows(forward_rows)


def mark_inverse_baseline_unavailable(row: dict[str, Any], reason: str = "M0 baseline is unavailable for this subject/case/seed") -> None:
    """
    标记 inverse comparison 不可用。

    Parameters
    ----------
    row : dict[str, Any]
        结果行。
    reason : str
        原因说明。

    Returns
    -------
    None
    """
    row.update(
        {
            "comparison_status": "baseline_unavailable",
            "comparison_reason": reason,
            "replay_ti_volume_path": None,
            "replay_roi_mean": None,
            "replay_roi_p999": None,
            "replay_non_roi_mean": None,
            "replay_roc": None,
            "replay_goal": None,
            "goal_gap_on_m0": None,
            "label_consistent": None,
            "optimized_center_drift_mean_mm": None,
            "optimized_center_drift_max_mm": None,
            "mapped_center_drift_mean_mm": None,
            "mapped_center_drift_max_mm": None,
            "inverse_pass": None,
        }
    )


def mark_inverse_comparison_failed(row: dict[str, Any], reason: str) -> None:
    """
    标记 inverse comparison 计算失败。

    Parameters
    ----------
    row : dict[str, Any]
        结果行。
    reason : str
        失败原因。

    Returns
    -------
    None
    """
    mark_inverse_baseline_unavailable(row, reason)
    row["comparison_status"] = "comparison_failed"


def _prepare_inverse_replay_metrics(
    inverse_rows: list[dict[str, Any]],
    work_root: Path,
    subjects: list[SubjectConfig],
    inverse_cases: list[dict[str, Any]],
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
) -> dict[str, list[dict[str, Any]]]:
    """
    在 report 阶段串行执行 inverse replay。

    Parameters
    ----------
    inverse_rows : list[dict[str, Any]]
        inverse 结果行。
    work_root : Path
        工作根目录。
    subjects : list[SubjectConfig]
        subject 配置列表。
    inverse_cases : list[dict[str, Any]]
        inverse case 配置列表。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        electrode mapping 缓存。
    """
    subject_by_id = {subject.id: subject for subject in subjects}
    case_by_name = {case["name"]: case for case in inverse_cases}
    workspace_cache: dict[tuple[str, str], WorkspaceBuildResult] = {}
    mapping_cache: dict[str, list[dict[str, Any]]] = {}
    for row in inverse_rows:
        if row.get("status") != "ok":
            continue
        subject = subject_by_id.get(str(row["subject_id"]))
        case = case_by_name.get(str(row["case_name"]))
        if subject is None or case is None:
            mark_inverse_comparison_failed(row, "missing subject or inverse case definition for report replay")
            continue
        mapping_path = str(row["electrode_mapping_path"])
        try:
            if mapping_path not in mapping_cache:
                mapping_cache[mapping_path] = load_mapping(Path(mapping_path))
            mapped_entries = mapping_cache[mapping_path]
            replay_ti_volume = run_replay_on_m0(
                subject=subject,
                preset=str(row["preset"]),
                case=case,
                seed=int(row["seed"]),
                mapped_entries=mapped_entries,
                work_root=work_root,
                preset_ini_paths=preset_ini_paths,
                debug_mesh=debug_mesh,
                workspace_cache=workspace_cache,
            )
            reference_img = nib.load(subject.reference_t1)
            baseline_workspace = ensure_workspace(
                workspace_cache=workspace_cache,
                subject=subject,
                preset="M0",
                work_root=work_root,
                preset_ini_paths=preset_ini_paths,
                debug_mesh=debug_mesh,
            )
            replay_metrics = compute_goal_from_volume(
                load_volume_data(replay_ti_volume, reference_img),
                reference_img,
                str(baseline_workspace.workspace_dir),
                case,
                Path(replay_ti_volume).parent / "_mask_cache",
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("inverse replay failed: %s", row)
            mark_inverse_comparison_failed(row, f"report replay failed: {exc}")
            continue
        row.update({"replay_ti_volume_path": str(replay_ti_volume), **replay_metrics})
    return mapping_cache


def annotate_inverse_rows(
    inverse_rows: list[dict[str, Any]],
    work_root: Path,
    subjects: list[SubjectConfig],
    inverse_cases: list[dict[str, Any]],
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
) -> list[dict[str, Any]]:
    """
    为 inverse 结果补充 replay 和与 M0 的 comparison。

    Parameters
    ----------
    inverse_rows : list[dict[str, Any]]
        inverse 结果行。
    work_root : Path
        工作根目录。
    subjects : list[SubjectConfig]
        subject 配置列表。
    inverse_cases : list[dict[str, Any]]
        inverse case 配置列表。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。

    Returns
    -------
    list[dict[str, Any]]
        补充后的 inverse 结果行。
    """
    mapping_cache = _prepare_inverse_replay_metrics(inverse_rows, work_root, subjects, inverse_cases, preset_ini_paths, debug_mesh)
    baseline_by_key = {
        (str(row["subject_id"]), str(row["case_name"]), int(row["seed"])): row
        for row in inverse_rows
        if row.get("status") == "ok" and row.get("preset") == "M0"
    }
    for row in inverse_rows:
        if row.get("status") != "ok":
            continue
        if row.get("comparison_status") == "comparison_failed":
            continue
        if row.get("preset") == "M0":
            row["comparison_status"] = "baseline_reference"
            row["comparison_reason"] = None
            row["goal_gap_on_m0"] = 0.0
            row["label_consistent"] = True
            row["optimized_center_drift_mean_mm"] = 0.0
            row["optimized_center_drift_max_mm"] = 0.0
            row["mapped_center_drift_mean_mm"] = 0.0
            row["mapped_center_drift_max_mm"] = 0.0
            row["inverse_pass"] = True
            continue
        key = (str(row["subject_id"]), str(row["case_name"]), int(row["seed"]))
        baseline = baseline_by_key.get(key)
        if baseline is None or baseline.get("comparison_status") == "comparison_failed" or baseline.get("replay_goal") is None:
            mark_inverse_baseline_unavailable(row)
            continue
        baseline_mapping_path = str(baseline["electrode_mapping_path"])
        current_mapping_path = str(row["electrode_mapping_path"])
        if baseline_mapping_path not in mapping_cache:
            mapping_cache[baseline_mapping_path] = load_mapping(Path(baseline_mapping_path))
        if current_mapping_path not in mapping_cache:
            mapping_cache[current_mapping_path] = load_mapping(Path(current_mapping_path))
        row["comparison_status"] = "ok"
        row["comparison_reason"] = None
        row["goal_gap_on_m0"] = relative_error(row["replay_goal"], baseline["replay_goal"])
        row.update(compute_mapping_drift(mapping_cache[baseline_mapping_path], mapping_cache[current_mapping_path]))
        preset = str(row["preset"])
        if preset in INVERSE_THRESHOLDS:
            row["inverse_pass"] = row["goal_gap_on_m0"] <= INVERSE_THRESHOLDS[preset]
        else:
            row["inverse_pass"] = None
    return sort_rows(inverse_rows)


def aggregate_report(
    mesh_rows: list[dict[str, Any]],
    forward_rows: list[dict[str, Any]],
    inverse_rows: list[dict[str, Any]],
    work_root: Path,
    executed_phases: list[str],
    subjects: list[SubjectConfig],
    inverse_cases: list[dict[str, Any]],
    preset_ini_paths: dict[str, Path],
    debug_mesh: bool,
) -> dict[str, Any]:
    """
    聚合并落盘所有报告。

    Parameters
    ----------
    mesh_rows : list[dict[str, Any]]
        mesh 结果行。
    forward_rows : list[dict[str, Any]]
        forward 结果行。
    inverse_rows : list[dict[str, Any]]
        inverse 结果行。
    work_root : Path
        工作根目录。
    executed_phases : list[str]
        本次执行的阶段列表。
    subjects : list[SubjectConfig]
        subject 配置列表。
    inverse_cases : list[dict[str, Any]]
        inverse case 配置列表。
    preset_ini_paths : dict[str, Path]
        preset ini 路径映射。
    debug_mesh : bool
        是否保留 mesh 调试输出。

    Returns
    -------
    dict[str, Any]
        汇总结果。
    """
    reports_dir = work_root / "reports"
    phase_to_rows = {"mesh": mesh_rows, "forward": forward_rows, "inverse": inverse_rows}
    phase_to_json = {
        "mesh": reports_dir / "mesh_stats.json",
        "forward": reports_dir / "forward_metrics.json",
        "inverse": reports_dir / "inverse_metrics.json",
    }
    merged_rows: dict[str, list[dict[str, Any]]] = {}
    for phase_name, current_rows in phase_to_rows.items():
        if phase_name in executed_phases:
            merged_rows[phase_name] = current_rows
        else:
            merged_rows[phase_name] = load_json(phase_to_json[phase_name], [])
    mesh_rows = annotate_mesh_rows(merged_rows["mesh"])
    forward_rows = annotate_forward_rows(merged_rows["forward"])
    inverse_rows = annotate_inverse_rows(merged_rows["inverse"], work_root, subjects, inverse_cases, preset_ini_paths, debug_mesh)
    summary: dict[str, Any] = {"mesh": {}, "forward": {}, "inverse": {}}
    for preset in PRESET_ORDER:
        preset_mesh = [row for row in mesh_rows if row.get("preset") == preset]
        if preset_mesh:
            ok_rows = [row for row in preset_mesh if row.get("status") == "ok"]
            summary["mesh"][preset] = {"total": len(preset_mesh), "ok": len(ok_rows)}
        preset_forward = [row for row in forward_rows if row.get("preset") == preset]
        if preset_forward:
            ok_rows = [row for row in preset_forward if row.get("status") == "ok"]
            passed_rows = [row for row in ok_rows if row.get("forward_pass") is True]
            comparable_rows = [row for row in ok_rows if row.get("forward_pass") is not None]
            summary["forward"][preset] = {
                "total": len(preset_forward),
                "ok": len(ok_rows),
                "comparable": len(comparable_rows),
                "pass_rate": len(passed_rows) / len(comparable_rows) if comparable_rows else math.nan,
            }
        preset_inverse = [row for row in inverse_rows if row.get("preset") == preset]
        if preset_inverse:
            ok_rows = [row for row in preset_inverse if row.get("status") == "ok"]
            passed_rows = [row for row in ok_rows if row.get("inverse_pass") is True]
            comparable_rows = [row for row in ok_rows if row.get("inverse_pass") is not None]
            summary["inverse"][preset] = {
                "total": len(preset_inverse),
                "ok": len(ok_rows),
                "comparable": len(comparable_rows),
                "pass_rate": len(passed_rows) / len(comparable_rows) if comparable_rows else math.nan,
            }
    write_rows_csv(reports_dir / "mesh_stats.csv", mesh_rows)
    write_rows_csv(reports_dir / "forward_metrics.csv", forward_rows)
    write_rows_csv(reports_dir / "inverse_metrics.csv", inverse_rows)
    write_json(reports_dir / "mesh_stats.json", mesh_rows)
    write_json(reports_dir / "forward_metrics.json", forward_rows)
    write_json(reports_dir / "inverse_metrics.json", inverse_rows)
    write_json(reports_dir / "summary.json", summary)
    return summary
