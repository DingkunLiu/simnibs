"""
Atlas 加载器模块

负责加载和解析 atlas registry，包括：
1. 路径解析（相对路径转绝对路径）
2. 获取指定 atlas 的规范信息
3. 遍历所有 atlas 规范信息
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from neuracle.atlas.registry import load_atlas_registry
from neuracle.utils.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def _resolve_repo_path(path_str: str | Path) -> Path:
    """
    将路径字符串解析为绝对路径。

    Parameters
    ----------
    path_str : str | Path
        输入路径字符串或 Path 对象

    Returns
    -------
    Path
        绝对路径

    Notes
    -----
    运行时再把相对路径恢复成绝对路径，调用方不需要关心路径拼接。
    如果输入已是绝对路径，则直接返回。
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_area_entry(area: dict[str, Any]) -> dict[str, Any]:
    """
    解析脑区条目中的相对路径为绝对路径。

    Parameters
    ----------
    area : dict[str, Any]
        包含 roi_path 的脑区条目

    Returns
    -------
    dict[str, Any]
        roi_path 已转换为绝对路径的脑区条目
    """
    resolved = dict(area)
    resolved["roi_path"] = str(_resolve_repo_path(area["roi_path"]))
    return resolved


def _resolve_atlas_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """
    解析 atlas spec 中的相对路径为绝对路径。

    Parameters
    ----------
    spec : dict[str, Any]
        atlas spec 字典

    Returns
    -------
    dict[str, Any]
        所有路径已转换为绝对路径的 atlas spec

    Notes
    -----
    对外返回的 spec 一律展开成绝对路径，避免各处重复 resolve。
    """
    resolved = dict(spec)
    for key in (
        "raw_atlas",
        "raw_label_table",
        "color_table",
        "space_template",
        "standardized_dir",
        "standardized_atlas",
        "standardized_label_table",
        "roi_dir",
    ):
        resolved[key] = str(_resolve_repo_path(spec[key]))
    resolved["areas"] = [_resolve_area_entry(area) for area in spec["areas"]]
    return resolved


def get_atlas_spec(
    atlas_name: str, registry: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    获取指定 atlas 的完整规范信息。

    Parameters
    ----------
    atlas_name : str
        atlas 名称
    registry : dict[str, Any] | None, optional
        registry 字典，默认为 None（从文件加载）

    Returns
    -------
    dict[str, Any]
        atlas 完整规范信息，包含所有路径的绝对路径形式

    Raises
    ------
    KeyError
        当指定的 atlas 不存在时
    """
    if registry is None:
        registry = load_atlas_registry()
    try:
        return _resolve_atlas_spec(registry["atlases"][atlas_name])
    except KeyError as exc:
        logger.error("未找到 atlas: %s", atlas_name)
        raise KeyError(f"未找到 atlas: {atlas_name}") from exc


def iter_atlas_specs(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """
    遍历所有 atlas 的规范信息。

    Parameters
    ----------
    registry : dict[str, Any] | None, optional
        registry 字典，默认为 None（从文件加载）

    Returns
    -------
    list[dict[str, Any]]
        所有 atlas 的完整规范信息列表
    """
    if registry is None:
        registry = load_atlas_registry()
    return [get_atlas_spec(name, registry) for name in registry["atlases"]]
