"""
Atlas 脑图谱模块

提供脑图谱数据管理和 ROI 定位功能。
"""

from neuracle.atlas.loader import get_atlas_spec, iter_atlas_specs
from neuracle.atlas.registry import load_atlas_registry
from neuracle.atlas.standardized import get_standardized_roi_path

__all__ = [
    "load_atlas_registry",
    "get_atlas_spec",
    "iter_atlas_specs",
    "get_standardized_roi_path",
]
