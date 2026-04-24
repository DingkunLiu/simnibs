from pathlib import Path

import nibabel as nib
import numpy as np

from neuracle.mesh_validation.mesh_validation_report import annotate_forward_rows, evaluate_forward_pass
from simnibs.utils.mesh_element_properties import ElementTags


def _write_nifti(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.Nifti1Image(data.astype(float), np.eye(4)).to_filename(path)
    return path


def _forward_row(
    tmp_path: Path,
    preset: str,
    values: np.ndarray,
    labels: np.ndarray,
    peak_value: float | None = None,
) -> dict:
    ti_path = _write_nifti(tmp_path / preset / "ti.nii.gz", values)
    labels_path = _write_nifti(tmp_path / preset / "labels.nii.gz", labels)
    return {
        "subject_id": "ernie",
        "case_name": "ti_demo",
        "preset": preset,
        "status": "ok",
        "ti_volume_path": str(ti_path),
        "final_labels_path": str(labels_path),
        "gm_peak_value": float(np.max(values)) if peak_value is None else peak_value,
        "gm_threshold_metrics": [],
    }


def test_annotate_forward_rows_sets_baseline_percentile_metrics(tmp_path: Path) -> None:
    values = np.arange(100, dtype=float).reshape(10, 10, 1)
    labels = np.full(values.shape, ElementTags.GM, dtype=float)
    rows = annotate_forward_rows([_forward_row(tmp_path, "M0", values, labels)])

    row = rows[0]

    assert row["comparison_status"] == "baseline_reference"
    assert row["gm_99_percentile_value"] == np.percentile(values.ravel(), 99)
    assert row["baseline_gm_99_percentile_value"] == np.percentile(values.ravel(), 99)
    assert row["gm_99_percentile_rel_error"] == 0.0
    assert row["forward_pass"] is True
    assert "gm_peak_rel_error" not in row
    assert "baseline_gm_peak_value" not in row


def test_annotate_forward_rows_computes_percentile_rel_error(tmp_path: Path) -> None:
    baseline_values = np.arange(100, dtype=float).reshape(10, 10, 1)
    current_values = np.roll(np.arange(100, dtype=float), 1).reshape(10, 10, 1) * 1.05
    baseline_labels = np.full(baseline_values.shape, ElementTags.GM, dtype=float)
    current_labels = np.full(current_values.shape, ElementTags.GM, dtype=float)
    rows = annotate_forward_rows(
        [
            _forward_row(tmp_path, "M0", baseline_values, baseline_labels),
            _forward_row(tmp_path, "M1", current_values, current_labels),
        ]
    )
    row = next(item for item in rows if item["preset"] == "M1")
    baseline_gm = baseline_values[baseline_labels == ElementTags.GM]
    current_gm = current_values[current_labels == ElementTags.GM]
    expected_baseline_p99 = float(np.percentile(baseline_gm, 99))
    expected_current_p99 = float(np.percentile(current_gm, 99))
    expected_rel_error = abs(expected_current_p99 - expected_baseline_p99) / abs(expected_baseline_p99)

    assert row["gm_99_percentile_value"] == expected_current_p99
    assert row["baseline_gm_99_percentile_value"] == expected_baseline_p99
    assert row["gm_99_percentile_rel_error"] == expected_rel_error
    assert "gm_90_percentile_dice" not in row


def test_evaluate_forward_pass_uses_p99_relative_error_not_peak_error() -> None:
    row = {
        "comparison_status": "ok",
        "gm_peak_rel_error": 0.0,
        "gm_99_percentile_rel_error": 0.2,
        "gm_pearson_r": 1.0,
        "gm_nrmse": 0.0,
    }

    assert evaluate_forward_pass(row, "M1") is False
    row["gm_99_percentile_rel_error"] = 0.01
    assert evaluate_forward_pass(row, "M1") is True
