from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from neuracle.atlas import iter_atlas_specs, load_atlas_registry


def _read_labels(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate_rois_for_atlas(spec: dict, force: bool) -> int:
    atlas_path = Path(spec["standardized_atlas"])
    if not atlas_path.exists():
        raise FileNotFoundError(f"标准化 atlas 不存在: {atlas_path}")

    label_rows = _read_labels(Path(spec["standardized_label_table"]))
    atlas_img = nib.load(atlas_path)
    # 先读取标准化后的 3D 离散 atlas，再按标签值切出单脑区二值 mask。
    atlas_data = np.rint(np.asanyarray(atlas_img.dataobj)).astype(np.int16)
    roi_header = atlas_img.header.copy()
    roi_header.set_data_dtype(np.uint8)
    roi_dir = Path(spec["roi_dir"])
    roi_dir.mkdir(parents=True, exist_ok=True)
    filename_by_index = {
        int(area["index"]): area["roi_filename"] for area in spec["areas"]
    }
    expected_filenames = set(filename_by_index.values())

    if force:
        # 强制重建时删除旧文件，避免历史命名规则残留造成目录污染。
        for stale_roi in roi_dir.glob("*.nii.gz"):
            if stale_roi.name not in expected_filenames:
                stale_roi.unlink()

    generated = 0
    for row in label_rows:
        index = int(row["index"])
        if index == 0:
            continue
        # 统一按 index 取目标文件名，避免重复 label_en 相互覆盖。
        filename = filename_by_index[index]
        roi_path = roi_dir / filename
        if roi_path.exists() and not force:
            continue
        mask = (atlas_data == index).astype(np.uint8)
        nib.save(nib.Nifti1Image(mask, atlas_img.affine, roi_header), roi_path)
        generated += 1
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="从标准化 atlas 生成单脑区 ROI")
    parser.add_argument("--atlas", help="只处理单个 atlas 名称", default=None)
    parser.add_argument("--force", action="store_true", help="覆盖已有输出")
    args = parser.parse_args()

    registry = load_atlas_registry()
    specs = iter_atlas_specs(registry)
    if args.atlas:
        specs = [spec for spec in specs if spec["name"] == args.atlas]
        if not specs:
            raise ValueError(f"未找到 atlas: {args.atlas}")

    for spec in specs:
        generated = generate_rois_for_atlas(spec, force=args.force)
        print(f"{spec['name']}: 新生成 {generated} 个 ROI 文件")


if __name__ == "__main__":
    main()
