from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from neuracle.atlas import iter_atlas_specs, load_atlas_registry
from neuracle.atlas.registry import ATLAS_VALIDATION_DIR, SIMNIBS_MNI_TEMPLATE


def _read_color_table(path: Path) -> dict[int, tuple[int, int, int, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["index"]): (
                int(row["r"]),
                int(row["g"]),
                int(row["b"]),
                int(row["a"]),
            )
            for row in rows
        }


def _save_overlay_png(
    template_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    color_table: dict[int, tuple[int, int, int, int]],
    output_path: Path,
) -> None:
    # 当前只输出中间 z 切面的叠加图，用于快速检查是否存在明显偏移或翻转。
    template = np.asanyarray(template_img.dataobj)
    atlas = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int32)
    z = atlas.shape[2] // 2
    bg = template[:, :, z].astype(np.float32)
    bg = bg - bg.min()
    if bg.max() > 0:
        bg = bg / bg.max()
    bg = (bg * 255).astype(np.uint8)
    rgba = np.stack([bg, bg, bg, np.full_like(bg, 255)], axis=-1)

    slice_labels = atlas[:, :, z]
    for label in np.unique(slice_labels):
        if label == 0 or label not in color_table:
            continue
        r, g, b, a = color_table[label]
        alpha = a / 255.0
        mask = slice_labels == label
        rgba[mask, 0] = np.clip((1 - alpha) * rgba[mask, 0] + alpha * r, 0, 255)
        rgba[mask, 1] = np.clip((1 - alpha) * rgba[mask, 1] + alpha * g, 0, 255)
        rgba[mask, 2] = np.clip((1 - alpha) * rgba[mask, 2] + alpha * b, 0, 255)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.transpose(rgba, (1, 0, 2))).save(output_path)


def validate_one(spec: dict, template_img: nib.Nifti1Image) -> dict:
    atlas_path = Path(spec["standardized_atlas"])
    if not atlas_path.exists():
        return {"atlas": spec["name"], "status": "missing_standardized_atlas"}

    atlas_img = nib.load(atlas_path)
    # 这里做的是轻量空间一致性检查，不是逐脑区的完整验收。
    report = {
        "atlas": spec["name"],
        "status": "ok",
        "shape_matches": list(atlas_img.shape[:3]) == list(template_img.shape[:3]),
        "affine_matches": np.allclose(atlas_img.affine, template_img.affine),
        "roi_dir_exists": Path(spec["roi_dir"]).exists(),
    }
    color_table = _read_color_table(Path(spec["color_table"]))
    _save_overlay_png(
        template_img,
        atlas_img,
        color_table,
        ATLAS_VALIDATION_DIR / f"{spec['name']}_overlay.png",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="验证标准化 atlas 产物")
    parser.add_argument("--atlas", help="只处理单个 atlas 名称", default=None)
    args = parser.parse_args()

    registry = load_atlas_registry()
    specs = iter_atlas_specs(registry)
    if args.atlas:
        specs = [spec for spec in specs if spec["name"] == args.atlas]
        if not specs:
            raise ValueError(f"未找到 atlas: {args.atlas}")

    template_img = nib.load(str(SIMNIBS_MNI_TEMPLATE))
    reports = [validate_one(spec, template_img) for spec in specs]
    ATLAS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ATLAS_VALIDATION_DIR / "validation_report.json"
    report_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已写出验证报告: {report_path}")


if __name__ == "__main__":
    main()
