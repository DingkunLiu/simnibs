from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import psutil
from nibabel.processing import resample_from_to

from neuracle.atlas import iter_atlas_specs, load_atlas_registry
from neuracle.atlas.registry import SIMNIBS_MNI_TEMPLATE

logger = logging.getLogger(__name__)


def _save_standardized_labels(
    source_label_table: Path, target_label_table: Path
) -> None:
    # 标签表直接复制，避免标准化阶段引入额外的数据变换。
    target_label_table.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_label_table, target_label_table)


def _resample_discrete(
    src_img: nib.Nifti1Image, target_img: nib.Nifti1Image
) -> nib.Nifti1Image:
    # 离散 atlas 必须使用最近邻插值，否则标签值会被插值污染。
    resampled = resample_from_to(
        src_img, (target_img.shape[:3], target_img.affine), order=0
    )
    data = np.rint(np.asanyarray(resampled.dataobj)).astype(np.int16)
    header = target_img.header.copy()
    header.set_data_dtype(np.int16)
    return nib.Nifti1Image(data, target_img.affine, header)


def _resample_difumo_4d_to_discrete(
    src_img: nib.Nifti1Image, target_img: nib.Nifti1Image
) -> nib.Nifti1Image:
    src_data = np.asanyarray(src_img.dataobj)
    if src_data.ndim != 4:
        raise ValueError("DiFuMo 输入必须是 4D maps.nii.gz")

    target_spec = (target_img.shape[:3], target_img.affine)
    # 按 component 流式重采样并更新 argmax，避免 4D 全量堆叠导致内存爆炸。
    best_values = np.full(target_img.shape[:3], -np.inf, dtype=np.float32)
    labels = np.zeros(target_img.shape[:3], dtype=np.int16)
    for component_idx in range(src_data.shape[3]):
        component_img = nib.Nifti1Image(
            np.asarray(src_data[:, :, :, component_idx], dtype=np.float32),
            src_img.affine,
            src_img.header,
        )
        resampled_component = resample_from_to(component_img, target_spec, order=1)
        component_values = np.asarray(resampled_component.dataobj, dtype=np.float32)
        update_mask = component_values > best_values
        best_values[update_mask] = component_values[update_mask]
        labels[update_mask] = component_idx + 1
        if component_idx % 64 == 0 or component_idx == src_data.shape[3] - 1:
            _log_memory(
                f"DiFuMo 组件 {component_idx + 1}/{src_data.shape[3]} 重采样完成"
            )

    labels[best_values <= 0] = 0
    header = target_img.header.copy()
    header.set_data_dtype(np.int16)
    return nib.Nifti1Image(labels, target_img.affine, header)


def _log_memory(stage: str) -> None:
    # 记录 RSS 峰值，便于定位大 atlas 标准化时的内存占用。
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    logger.info("%s，当前 RSS: %.2f GB", stage, rss_gb)


def standardize_one(spec: dict, target_img: nib.Nifti1Image, force: bool) -> Path:
    target_path = Path(spec["standardized_atlas"])
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not force:
        # atlas 文件已存在时仍然同步标签表，确保标准化目录完整可用。
        _save_standardized_labels(
            Path(spec["raw_label_table"]),
            Path(spec["standardized_label_table"]),
        )
        return target_path

    src_img = nib.load(spec["raw_atlas"])
    _log_memory(f"开始标准化 {spec['name']}")
    if spec["name"].startswith("DiFuMo"):
        # DiFuMo 的权威输入是 4D maps，必须先重采样再在目标空间做 argmax。
        standardized = _resample_difumo_4d_to_discrete(src_img, target_img)
    else:
        standardized = _resample_discrete(src_img, target_img)

    nib.save(standardized, target_path)
    _save_standardized_labels(
        Path(spec["raw_label_table"]),
        Path(spec["standardized_label_table"]),
    )
    _log_memory(f"完成标准化 {spec['name']}")
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(description="标准化 atlas 到 SimNIBS MNI 空间")
    parser.add_argument("--atlas", help="只处理单个 atlas 名称", default=None)
    parser.add_argument("--force", action="store_true", help="覆盖已有输出")
    args = parser.parse_args()

    registry = load_atlas_registry()
    target_img = nib.load(str(SIMNIBS_MNI_TEMPLATE))
    specs = iter_atlas_specs(registry)
    if args.atlas:
        specs = [spec for spec in specs if spec["name"] == args.atlas]
        if not specs:
            raise ValueError(f"未找到 atlas: {args.atlas}")

    for spec in specs:
        out = standardize_one(spec, target_img, force=args.force)
        print(f"已标准化 atlas: {spec['name']} -> {out}")


if __name__ == "__main__":
    main()
