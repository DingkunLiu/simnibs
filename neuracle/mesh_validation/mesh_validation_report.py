"""
Mesh validation V2 报告聚合逻辑。
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuracle.mesh_validation.mesh_validation_schema import load_stage_results, write_json
from neuracle.mesh_validation.mesh_validation_stages import (
    FORWARD_HOTSPOT_THRESHOLDS_ABS,
    FORWARD_THRESHOLDS,
    bidirectional_surface_distance,
    load_volume_data,
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
            "gm_peak_value": result.get("gm_peak_value"),
            "gm_threshold_metrics": result.get("gm_threshold_metrics", []),
        }
        rows.append(base)
    return sort_rows(rows)


def _threshold_token(threshold: float) -> str:
    return f"{int(round(float(threshold) * 100)):03d}"


def _gm_threshold_columns(threshold: float) -> dict[str, str]:
    token = _threshold_token(threshold)
    return {
        "tp": f"gm_threshold_tp_abs_{token}",
        "fp": f"gm_threshold_fp_abs_{token}",
        "fn": f"gm_threshold_fn_abs_{token}",
        "tn": f"gm_threshold_tn_abs_{token}",
        "tpr": f"gm_threshold_tpr_abs_{token}",
        "fpr": f"gm_threshold_fpr_abs_{token}",
        "precision": f"gm_threshold_precision_abs_{token}",
        "dice": f"gm_threshold_dice_abs_{token}",
        "jaccard": f"gm_threshold_jaccard_abs_{token}",
        "baseline": f"baseline_gm_threshold_voxels_abs_{token}",
        "current": f"current_gm_threshold_voxels_abs_{token}",
    }


def _gm_thresholds_for_row(row: dict[str, Any], baseline: dict[str, Any] | None = None) -> list[float]:
    raw_current = row.get("gm_threshold_metrics", [])
    raw_baseline = baseline.get("gm_threshold_metrics", []) if baseline else []
    raw_metrics = raw_current or raw_baseline
    if raw_metrics:
        return [float(metric["threshold_abs"]) for metric in raw_metrics]
    return list(FORWARD_HOTSPOT_THRESHOLDS_ABS)


def _set_gm_threshold_columns(
    row: dict[str, Any],
    threshold_metrics: list[dict[str, Any]] | None,
    thresholds: list[float],
) -> None:
    metrics_by_threshold = {}
    for metric in threshold_metrics or []:
        metrics_by_threshold[round(float(metric["threshold_abs"]), 6)] = metric
    for threshold in thresholds:
        metric = metrics_by_threshold.get(round(float(threshold), 6), {})
        columns = _gm_threshold_columns(float(threshold))
        row[columns["tp"]] = metric.get("tp")
        row[columns["fp"]] = metric.get("fp")
        row[columns["fn"]] = metric.get("fn")
        row[columns["tn"]] = metric.get("tn")
        row[columns["tpr"]] = metric.get("tpr")
        row[columns["fpr"]] = metric.get("fpr")
        row[columns["precision"]] = metric.get("precision")
        row[columns["dice"]] = metric.get("dice")
        row[columns["jaccard"]] = metric.get("jaccard")
        row[columns["baseline"]] = metric.get("baseline_gm_threshold_voxels")
        row[columns["current"]] = metric.get("current_gm_threshold_voxels")


def _safe_rate(numerator: int, denominator: int, empty_value: float) -> float:
    if denominator == 0:
        return empty_value
    return float(numerator / denominator)


def _trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))


def compute_gm_threshold_consistency_metrics(
    current_ti: np.ndarray,
    baseline_ti: np.ndarray,
    gm_mask: np.ndarray,
    thresholds: list[float],
) -> dict[str, Any]:
    threshold_metrics: list[dict[str, Any]] = []
    gm_voxels = int(np.count_nonzero(gm_mask))
    for threshold in thresholds:
        baseline_active = gm_mask & (baseline_ti >= float(threshold))
        current_active = gm_mask & (current_ti >= float(threshold))
        tp = int(np.count_nonzero(current_active & baseline_active))
        fp = int(np.count_nonzero(current_active & ~baseline_active & gm_mask))
        fn = int(np.count_nonzero(~current_active & baseline_active & gm_mask))
        tn = gm_voxels - tp - fp - fn
        threshold_metrics.append(
            {
                "threshold_abs": float(threshold),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "tpr": _safe_rate(tp, tp + fn, 1.0),
                "fpr": _safe_rate(fp, fp + tn, 0.0),
                "precision": _safe_rate(tp, tp + fp, 1.0),
                "dice": _safe_rate(2 * tp, 2 * tp + fp + fn, 1.0),
                "jaccard": _safe_rate(tp, tp + fp + fn, 1.0),
                "baseline_gm_threshold_voxels": int(np.count_nonzero(baseline_active)),
                "current_gm_threshold_voxels": int(np.count_nonzero(current_active)),
            }
        )
    roc_points = sorted(
        ((float(metric["fpr"]), float(metric["tpr"])) for metric in threshold_metrics),
        key=lambda item: item[0],
    )
    roc_x = np.asarray([0.0, *[point[0] for point in roc_points], 1.0], dtype=float)
    roc_y = np.asarray([0.0, *[point[1] for point in roc_points], 1.0], dtype=float)
    return {
        "gm_threshold_metrics_abs": threshold_metrics,
        "gm_threshold_consistency_auc_abs": _trapezoid_area(roc_x, roc_y),
        "gm_threshold_mean_dice_abs": float(np.mean([float(metric["dice"]) for metric in threshold_metrics])) if threshold_metrics else math.nan,
        "gm_threshold_mean_jaccard_abs": float(np.mean([float(metric["jaccard"]) for metric in threshold_metrics])) if threshold_metrics else math.nan,
    }


def _percentile_or_nan(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return math.nan
    return float(np.percentile(values, percentile))


def _relative_error_or_nan(value: float, baseline: float) -> float:
    if baseline in (0, 0.0) or math.isnan(baseline):
        return math.nan
    return float(abs(value - baseline) / abs(baseline))


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
    thresholds = _gm_thresholds_for_row(row)
    row.update(
        {
            "comparison_status": "baseline_unavailable",
            "comparison_reason": "M0 baseline is unavailable for this subject/case",
            "gm_pearson_r": None,
            "gm_nrmse": None,
            "gm_99_percentile_value": None,
            "baseline_gm_99_percentile_value": None,
            "gm_99_percentile_rel_error": None,
            "gm_threshold_consistency_auc_abs": None,
            "gm_threshold_mean_dice_abs": None,
            "gm_threshold_mean_jaccard_abs": None,
            "forward_pass": None,
        }
    )
    _set_gm_threshold_columns(row, None, thresholds)


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
        row.get("gm_99_percentile_rel_error", math.nan) <= threshold["hotspot"],
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
        (str(row["subject_id"]), str(row["case_name"])): row for row in forward_rows if row.get("status") == "ok" and row.get("preset") == "M0"
    }
    gm_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for row in forward_rows:
        if row.get("status") != "ok":
            continue
        if row.get("preset") == "M0":
            thresholds = _gm_thresholds_for_row(row)
            gm_voxels = next(
                (
                    int(metric.get("gm_voxels"))
                    for metric in row.get("gm_threshold_metrics", [])
                    if metric.get("gm_voxels") is not None
                ),
                0,
            )
            baseline_threshold_metrics = []
            for metric in row.get("gm_threshold_metrics", []):
                active_voxels = int(metric.get("current_gm_threshold_voxels", 0))
                baseline_threshold_metrics.append(
                    {
                        "threshold_abs": float(metric["threshold_abs"]),
                        "tp": active_voxels,
                        "fp": 0,
                        "fn": 0,
                        "tn": gm_voxels - active_voxels,
                        "tpr": 1.0,
                        "fpr": 0.0,
                        "precision": 1.0,
                        "dice": 1.0,
                        "jaccard": 1.0,
                        "baseline_gm_threshold_voxels": active_voxels,
                        "current_gm_threshold_voxels": active_voxels,
                    }
                )
            row["comparison_status"] = "baseline_reference"
            row["comparison_reason"] = None
            row["gm_pearson_r"] = 1.0
            row["gm_nrmse"] = 0.0
            reference_img = nib.load(str(Path(str(row["ti_volume_path"]))))
            baseline_ti = load_volume_data(Path(str(row["ti_volume_path"])), reference_img)
            baseline_labels = load_volume_data(Path(str(row["final_labels_path"])), reference_img)
            baseline_gm = baseline_ti[baseline_labels == ElementTags.GM]
            gm_99_percentile_value = _percentile_or_nan(baseline_gm, 99)
            row["gm_99_percentile_value"] = gm_99_percentile_value
            row["baseline_gm_99_percentile_value"] = gm_99_percentile_value
            row["gm_99_percentile_rel_error"] = 0.0
            row["gm_threshold_consistency_auc_abs"] = 1.0
            row["gm_threshold_mean_dice_abs"] = 1.0
            row["gm_threshold_mean_jaccard_abs"] = 1.0
            _set_gm_threshold_columns(row, baseline_threshold_metrics, thresholds)
            row["forward_pass"] = True
            continue
        baseline = baseline_by_key.get((str(row["subject_id"]), str(row["case_name"])))
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
            current_gm_mask = current_labels == ElementTags.GM
            baseline_gm_mask = baseline_labels == ElementTags.GM
            gm_mask = current_gm_mask & baseline_gm_mask
            gm_cache[gm_key] = {
                "current_ti": current_ti,
                "baseline_ti": baseline_ti,
                "gm_mask": gm_mask,
                "current_gm": current_ti[gm_mask],
                "baseline_gm": baseline_ti[gm_mask],
                "current_own_gm": current_ti[current_gm_mask],
                "baseline_own_gm": baseline_ti[baseline_gm_mask],
            }
        current_gm = gm_cache[gm_key]["current_gm"]
        baseline_gm = gm_cache[gm_key]["baseline_gm"]
        current_own_gm = gm_cache[gm_key]["current_own_gm"]
        baseline_own_gm = gm_cache[gm_key]["baseline_own_gm"]
        if current_gm.size > 1 and np.std(current_gm) > 0 and np.std(baseline_gm) > 0:
            pearson_r = float(np.corrcoef(current_gm, baseline_gm)[0, 1])
        else:
            pearson_r = math.nan
        if baseline_gm.size and np.linalg.norm(baseline_gm) > 0:
            nrmse = float(np.linalg.norm(current_gm - baseline_gm) / np.linalg.norm(baseline_gm))
        else:
            nrmse = math.nan
        current_p99 = _percentile_or_nan(current_own_gm, 99)
        baseline_p99 = _percentile_or_nan(baseline_own_gm, 99)
        gm_threshold_consistency = compute_gm_threshold_consistency_metrics(
            current_ti=gm_cache[gm_key]["current_ti"],
            baseline_ti=gm_cache[gm_key]["baseline_ti"],
            gm_mask=gm_cache[gm_key]["gm_mask"],
            thresholds=_gm_thresholds_for_row(row, baseline),
        )
        row.update(
            {
                "comparison_status": "ok",
                "comparison_reason": None,
                "gm_pearson_r": pearson_r,
                "gm_nrmse": nrmse,
                "gm_99_percentile_value": current_p99,
                "baseline_gm_99_percentile_value": baseline_p99,
                "gm_99_percentile_rel_error": _relative_error_or_nan(current_p99, baseline_p99),
                "gm_threshold_consistency_auc_abs": gm_threshold_consistency["gm_threshold_consistency_auc_abs"],
                "gm_threshold_mean_dice_abs": gm_threshold_consistency["gm_threshold_mean_dice_abs"],
                "gm_threshold_mean_jaccard_abs": gm_threshold_consistency["gm_threshold_mean_jaccard_abs"],
            }
        )
        _set_gm_threshold_columns(
            row,
            gm_threshold_consistency["gm_threshold_metrics_abs"],
            _gm_thresholds_for_row(row, baseline),
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
