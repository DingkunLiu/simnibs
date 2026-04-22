"""
Mesh validation V2 报告聚合逻辑。
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

from neuracle.mesh_validation.mesh_validation_schema import load_stage_results, write_json
from neuracle.mesh_validation.mesh_validation_stages import (
    FORWARD_THRESHOLDS,
    bidirectional_surface_distance,
    load_volume_data,
    relative_error,
    scalp_surface_points,
)
from neuracle.mesh_validation.mesh_validation_workspace import PRESET_ORDER
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
        排序后的结果行。
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


def flatten_forward_results(forward_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    将 case 级 forward 结果展开为 ROI 行。

    Parameters
    ----------
    forward_results : list[dict[str, Any]]
        case 级结果。

    Returns
    -------
    list[dict[str, Any]]
        ROI 级结果行。
    """
    rows: list[dict[str, Any]] = []
    for result in forward_results:
        base = {
            "subject_id": result.get("subject_id"),
            "case_name": result.get("case_name"),
            "preset": result.get("preset"),
            "status": result.get("status"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "ti_mesh_path": result.get("ti_mesh_path"),
            "ti_volume_path": result.get("ti_volume_path"),
            "final_labels_path": result.get("final_labels_path"),
            "failure_stage": result.get("failure_stage"),
            "error": result.get("error"),
            "traceback": result.get("traceback"),
            "hotspot_value": result.get("hotspot_value"),
        }
        if result.get("status") != "ok":
            rows.append(base)
            continue
        for roi in result.get("roi_metrics", []):
            rows.append({**base, **roi})
    return sort_rows(rows)


def annotate_mesh_rows(mesh_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    为 mesh 结果补充 M0 的头皮表面对比指标。

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
        baseline_points_by_subject[str(row["subject_id"])] = scalp_surface_points(Path(str(row["mesh_file"])))
    for row in mesh_rows:
        if row.get("status") != "ok":
            continue
        baseline_points = baseline_points_by_subject.get(str(row["subject_id"]))
        if baseline_points is None:
            row["scalp_mean_distance_mm"] = None
            row["scalp_hausdorff95_mm"] = None
            continue
        current_points = scalp_surface_points(Path(str(row["mesh_file"])))
        row.update(bidirectional_surface_distance(baseline_points, current_points))
    return sort_rows(mesh_rows)


def mark_forward_baseline_unavailable(row: dict[str, Any]) -> None:
    """
    标记 forward comparison 不可用。

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
    评估 forward 是否通过。

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
    为 forward 结果补充 M0 comparison。

    Parameters
    ----------
    forward_rows : list[dict[str, Any]]
        ROI 级 forward 结果。

    Returns
    -------
    list[dict[str, Any]]
        补充后的结果行。
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
            reference_img = nib.load(str(Path(str(row["ti_volume_path"]))))
            current_ti = load_volume_data(Path(str(row["ti_volume_path"])), reference_img)
            baseline_ti = load_volume_data(Path(str(baseline["ti_volume_path"])), reference_img)
            current_labels = load_volume_data(Path(str(row["final_labels_path"])), reference_img)
            baseline_labels = load_volume_data(Path(str(baseline["final_labels_path"])), reference_img)
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
                "roi_mean_rel_error": relative_error(float(row["roi_mean"]), float(baseline["roi_mean"])),
                "roi_median_rel_error": relative_error(float(row["roi_median"]), float(baseline["roi_median"])),
                "roi_p95_rel_error": relative_error(float(row["roi_p95"]), float(baseline["roi_p95"])),
                "roi_p99_rel_error": relative_error(float(row["roi_p99"]), float(baseline["roi_p99"])),
                "roi_max_rel_error": relative_error(float(row["roi_max"]), float(baseline["roi_max"])),
                "hotspot_rel_error": relative_error(float(row["hotspot_value"]), float(baseline["hotspot_value"])),
            }
        )
        row["forward_pass"] = evaluate_forward_pass(row, str(row["preset"]))
    return sort_rows(forward_rows)


def _report_presets(selected_presets: list[str]) -> list[str]:
    """
    生成报告加载所需的 preset 列表。

    Parameters
    ----------
    selected_presets : list[str]
        用户请求的 preset 列表。

    Returns
    -------
    list[str]
        加载所需的 preset 列表。
    """
    ordered = ["M0", *selected_presets]
    return list(dict.fromkeys(ordered))


def _filter_requested_presets(rows: list[dict[str, Any]], selected_presets: list[str]) -> list[dict[str, Any]]:
    """
    过滤出用户请求的 preset。

    Parameters
    ----------
    rows : list[dict[str, Any]]
        结果行列表。
    selected_presets : list[str]
        用户请求的 preset。

    Returns
    -------
    list[dict[str, Any]]
        过滤后的结果行。
    """
    allowed = set(selected_presets)
    return [row for row in rows if str(row.get("preset")) in allowed]


def _build_summary(
    mesh_rows: list[dict[str, Any]],
    forward_rows: list[dict[str, Any]],
    inverse_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    构建 summary.json。

    Parameters
    ----------
    mesh_rows : list[dict[str, Any]]
        mesh 报告行。
    forward_rows : list[dict[str, Any]]
        forward 报告行。
    inverse_rows : list[dict[str, Any]]
        inverse 正式结果。
    replay_rows : list[dict[str, Any]]
        replay 正式结果。

    Returns
    -------
    dict[str, Any]
        summary 结果。
    """
    summary: dict[str, Any] = {"mesh": {}, "forward": {}, "inverse": {}, "replay": {}}
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
            summary["inverse"][preset] = {"total": len(preset_inverse), "ok": len(ok_rows)}
        preset_replay = [row for row in replay_rows if row.get("preset") == preset]
        if preset_replay:
            ok_rows = [row for row in preset_replay if row.get("status") == "ok"]
            passed_rows = [row for row in ok_rows if row.get("inverse_pass") is True]
            comparable_rows = [row for row in ok_rows if row.get("inverse_pass") is not None]
            summary["replay"][preset] = {
                "total": len(preset_replay),
                "ok": len(ok_rows),
                "comparable": len(comparable_rows),
                "pass_rate": len(passed_rows) / len(comparable_rows) if comparable_rows else math.nan,
            }
    return summary


def aggregate_report(work_root: Path, selected_presets: list[str]) -> dict[str, Any]:
    """
    聚合 V2 正式结果并写出报告。

    Parameters
    ----------
    work_root : Path
        工作根目录。
    selected_presets : list[str]
        用户请求输出的 preset 列表。

    Returns
    -------
    dict[str, Any]
        summary 结果。
    """
    reports_dir = work_root / "reports"
    presets_for_loading = _report_presets(selected_presets)
    mesh_results = load_stage_results(work_root, "mesh", presets_for_loading)
    forward_results = load_stage_results(work_root, "forward", presets_for_loading)
    inverse_results = load_stage_results(work_root, "inverse", presets_for_loading)
    replay_results = load_stage_results(work_root, "replay", presets_for_loading)
    mesh_report_rows = annotate_mesh_rows(mesh_results)
    forward_report_rows = annotate_forward_rows(flatten_forward_results(forward_results))
    inverse_report_rows = sort_rows(inverse_results)
    replay_report_rows = sort_rows(replay_results)
    output_mesh_rows = _filter_requested_presets(mesh_report_rows, selected_presets)
    output_forward_rows = _filter_requested_presets(forward_report_rows, selected_presets)
    output_inverse_rows = _filter_requested_presets(inverse_report_rows, selected_presets)
    output_replay_rows = _filter_requested_presets(replay_report_rows, selected_presets)
    summary = _build_summary(output_mesh_rows, output_forward_rows, output_inverse_rows, output_replay_rows)
    write_rows_csv(reports_dir / "mesh_report.csv", output_mesh_rows)
    write_rows_csv(reports_dir / "forward_report.csv", output_forward_rows)
    write_rows_csv(reports_dir / "inverse_report.csv", output_inverse_rows)
    write_rows_csv(reports_dir / "replay_report.csv", output_replay_rows)
    write_json(reports_dir / "mesh_report.json", output_mesh_rows)
    write_json(reports_dir / "forward_report.json", output_forward_rows)
    write_json(reports_dir / "inverse_report.json", output_inverse_rows)
    write_json(reports_dir / "replay_report.json", output_replay_rows)
    write_json(reports_dir / "summary.json", summary)
    return summary
