"""
Atlas 标准化 ROI 工具模块

负责根据 atlas 名称和脑区名称定位标准化 ROI 文件，包括：
1. 脑区名称规范化
2. 脑区查找匹配
3. 合并 ROI 生成
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from neuracle.atlas.loader import get_atlas_spec
from neuracle.atlas.registry import _slugify

logger = logging.getLogger(__name__)


def _normalize_area_key(value: str) -> str:
    """
    规范化脑区名称键值。

    Parameters
    ----------
    value : str
        原始脑区名称

    Returns
    -------
    str
        规范化后的名称（小写，去除多余空白）

    Notes
    -----
    将所有空白字符替换为单个空格并去除首尾空格，然后转为小写。
    """
    return re.sub(r"\s+", " ", value).strip().lower()


def _find_matching_areas(
    atlas_name: str,
    area_name: str,
    registry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    在指定 atlas 中查找匹配的脑区。

    Parameters
    ----------
    atlas_name : str
        atlas 名称
    area_name : str
        脑区名称
    registry : dict[str, Any] | None, optional
        registry 字典，默认为 None（从文件加载）

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        (atlas spec, 匹配的脑区列表)

    Notes
    -----
    同时支持英文名、中文名、文件名 stem 以及直接传 index。
    """
    spec = get_atlas_spec(atlas_name, registry)
    area_key = _normalize_area_key(area_name)
    matches: list[dict[str, Any]] = []
    for area in spec["areas"]:
        if area_key in {
            _normalize_area_key(area["label_en"]),
            _normalize_area_key(area["label_zh"]),
            _normalize_area_key(Path(area["roi_filename"]).stem),
            str(area["index"]),
        }:
            matches.append(area)
    return spec, matches


def _build_merged_roi_path(
    spec: dict[str, Any], area_name: str, matches: list[dict[str, Any]]
) -> Path:
    """
    构建合并 ROI 文件的路径。

    Parameters
    ----------
    spec : dict[str, Any]
        atlas spec
    area_name : str
        脑区名称
    matches : list[dict[str, Any]]
        匹配的脑区列表

    Returns
    -------
    Path
        合并 ROI 文件的路径
    """
    merged_dir = Path(spec["standardized_dir"]) / "merged_rois"
    merged_dir.mkdir(parents=True, exist_ok=True)
    match_token = "_".join(
        str(area["index"])
        for area in sorted(matches, key=lambda item: int(item["index"]))
    )
    return merged_dir / f"{_slugify(area_name)}__merged__{match_token}.nii.gz"


def _ensure_merged_roi(
    spec: dict[str, Any], area_name: str, matches: list[dict[str, Any]]
) -> Path:
    """
    确保合并 ROI 文件存在，如不存在则创建。

    Parameters
    ----------
    spec : dict[str, Any]
        atlas spec
    area_name : str
        脑区名称
    matches : list[dict[str, Any]]
        匹配的脑区列表

    Returns
    -------
    Path
        合并 ROI 文件的路径

    Notes
    -----
    同名脑区只在首次命中时生成一次合并 mask，后续直接复用缓存结果。
    多个 component 时做体素级并集，输出仍保持二值 mask。
    """
    merged_path = _build_merged_roi_path(spec, area_name, matches)
    source_paths = [Path(area["roi_path"]) for area in matches]
    if merged_path.exists():
        merged_mtime = merged_path.stat().st_mtime
        if all(
            source_path.exists() and source_path.stat().st_mtime <= merged_mtime
            for source_path in source_paths
        ):
            return merged_path

    if len(source_paths) == 1:
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_paths[0], merged_path)
        return merged_path

    first_img = nib.load(str(source_paths[0]))
    merged_mask = np.asarray(first_img.dataobj, dtype=np.uint8) > 0
    for source_path in source_paths[1:]:
        source_img = nib.load(str(source_path))
        merged_mask |= np.asarray(source_img.dataobj, dtype=np.uint8) > 0

    header = first_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(merged_mask.astype(np.uint8), first_img.affine, header),
        merged_path,
    )
    return merged_path


def _ensure_merged_roi_from_paths(
    standardized_dir: Path, area_name: str, source_paths: list[Path]
) -> Path:
    """
    Ensure a merged ROI exists for a list of ROI paths.
    """
    merged_dir = standardized_dir / "merged_rois"
    merged_dir.mkdir(parents=True, exist_ok=True)
    source_paths = sorted(source_paths, key=lambda path: path.name)
    match_token = "_".join(path.stem.replace(".nii", "") for path in source_paths)
    merged_path = merged_dir / f"{_slugify(area_name)}__merged__{match_token}.nii.gz"
    if merged_path.exists():
        merged_mtime = merged_path.stat().st_mtime
        if all(
            source_path.exists() and source_path.stat().st_mtime <= merged_mtime
            for source_path in source_paths
        ):
            return merged_path

    first_img = nib.load(str(source_paths[0]))
    merged_mask = np.asarray(first_img.dataobj, dtype=np.uint8) > 0
    for source_path in source_paths[1:]:
        source_img = nib.load(str(source_path))
        merged_mask |= np.asarray(source_img.dataobj, dtype=np.uint8) > 0

    header = first_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(merged_mask.astype(np.uint8), first_img.affine, header),
        merged_path,
    )
    return merged_path


def get_standardized_roi_path(
    atlas_name: str,
    area_name: str | list[str],
    registry: dict[str, Any] | None = None,
) -> Path:
    """
    根据 atlas 名称和脑区名称定位标准化 ROI 文件。

    Parameters
    ----------
    atlas_name : str
        atlas 名称
    area_name : str
        脑区名称（支持英文名、中文名、文件名 stem、index）
    registry : dict[str, Any] | None, optional
        registry 字典，默认为 None（从文件加载）

    Returns
    -------
    Path
        ROI 文件的绝对路径

    Raises
    ------
    KeyError
        当在 atlas 中找不到指定的脑区时

    Notes
    -----
    重名脑区会自动合并。
    """
    if isinstance(area_name, list):
        spec = get_atlas_spec(atlas_name, registry)
        source_paths = [get_standardized_roi_path(atlas_name, area, registry) for area in area_name]
        return _ensure_merged_roi_from_paths(
            Path(spec["standardized_dir"]),
            "__".join(area_name),
            source_paths,
        )

    spec, matches = _find_matching_areas(atlas_name, area_name, registry)
    if len(matches) == 1:
        return Path(matches[0]["roi_path"])
    if len(matches) > 1:
        return _ensure_merged_roi(spec, area_name, matches)
    logger.error("未在 atlas %s 中找到脑区: %s", atlas_name, area_name)
    raise KeyError(f"未在 atlas {atlas_name} 中找到脑区: {area_name}")
