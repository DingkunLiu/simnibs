"""
电导率相关工具函数

提供组织电导率字典与列表之间的转换功能，用于 SimNIBS 仿真参数配置。
"""

import logging

from neuracle.utils.constants import CONDUCTIVITY_TISSUE_NAMES

logger = logging.getLogger(__name__)


def cond_dict_to_list(cond_dict: dict[str, float]) -> list[float]:
    """
    将组织名称到电导率的字典转换为 SimNIBS 内部使用的列表。

    Parameters
    ----------
    cond_dict : dict[str, float]
        组织名称到电导率的映射

    Returns
    -------
    list[float]
        按 tissue tag 顺序的电导率列表

    Raises
    ------
    ValueError
        当 cond_dict 包含不在定义组织列表中的 key 时
    """
    for tissue in cond_dict:
        if tissue not in CONDUCTIVITY_TISSUE_NAMES:
            logger.error("未知的组织类型: %s", tissue)
            raise ValueError(f"未知的组织类型: {tissue}")
    return [cond_dict[tissue] for tissue in CONDUCTIVITY_TISSUE_NAMES]
