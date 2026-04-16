"""
TI Optimization 结果处理模块

原理：
    1. 读取优化后的电极映射结果
    2. 从 electrode_mapping.json 文件解析 EEG 电极标签

用法：
    from neuracle.ti_optimization import get_electrode_mapping

    electrode_A, electrode_B = get_electrode_mapping(output_dir)
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def get_electrode_mapping(output_dir: str) -> tuple[list[str], list[str]]:
    """
    获取优化电极映射到的 EEG 电极标签

    Parameters
    ----------
    output_dir : str
        优化输出目录

    Returns
    -------
    tuple[list[str], list[str]]
        (electrode_A, electrode_B) - 两组电极的映射标签列表

    Raises
    ------
    FileNotFoundError
        当 electrode_mapping.json 不存在时
    """
    mapping_path = os.path.join(output_dir, "electrode_mapping.json")
    if not os.path.exists(mapping_path):
        logger.error("电极映射文件不存在: %s", mapping_path)
        raise FileNotFoundError(f"电极映射文件不存在: {mapping_path}")

    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    mapped_labels = mapping.get("mapped_labels", [])
    n = len(mapped_labels) // 2
    electrode_A = mapped_labels[:n]
    electrode_B = mapped_labels[n:]

    logger.info(
        "电极映射结果: electrode_A=%s, electrode_B=%s", electrode_A, electrode_B
    )
    return electrode_A, electrode_B
