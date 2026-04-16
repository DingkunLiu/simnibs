"""
neuracle.utils 模块

提供通用工具函数和数据结构，包括：
- 电导率相关工具函数
- 参数转换工具
- TI 导出工具
- nifti 工具
- 常量定义
"""

import logging
import os

from neuracle.utils.conductivity import cond_dict_to_list
from neuracle.utils.constants import (
    BUILT_IN_DIR_PATH,
    BUILT_IN_DTI_FILE_PATH,
    CONDUCTIVITY_TISSUE_NAMES,
    DATA_ROOT,
    DEBUG,
    DEFAULT_STS_ROLE_SESSION_NAME,
    DEFAULT_STS_TOKEN_DURATION_SECONDES,
    EEG10_10_CUTINI_2011,
    EEG10_10_NEUROELECTRICS,
    EEG10_10_UI_JURAK_2007,
    EEG10_20_EXTENDED_SPM12,
    EEG10_20_OKAMOTO_2004,
    N_WORKERS,
    NEURACLE_DIR,
    NON_ROI_THRESHOLD,
    PROJECT_ROOT,
    STANDARD_COND,
)
from neuracle.utils.ti_export import export_ti_to_nifti

logger = logging.getLogger(__name__)


def find_montage_file(dir_path: str, montage: str) -> str:
    """
    在头模目录中查找指定名称的 montage CSV 文件。

    Parameters
    ----------
    dir_path : str
        头模目录路径
    montage : str
        montage 文件名或路径

    Returns
    -------
    str
        完整的 CSV 文件路径

    Raises
    ------
    FileNotFoundError
        当 montage 文件不存在时
    """
    candidate_paths = []
    if os.path.isabs(montage):
        candidate_paths.append(montage)
    else:
        candidate_paths.append(os.path.join(dir_path, "eeg_positions", montage))
        candidate_paths.append(
            os.path.join(dir_path, "eeg_positions", f"{montage}.csv")
        )
    for montage_path in candidate_paths:
        if os.path.exists(montage_path):
            return montage_path

    logger.error("montage 文件不存在: %s", montage)
    raise FileNotFoundError(f"montage 文件不存在: {montage}")


__all__ = [
    "N_WORKERS",
    "NON_ROI_THRESHOLD",
    "PROJECT_ROOT",
    "NEURACLE_DIR",
    "DATA_ROOT",
    "STANDARD_COND",
    "CONDUCTIVITY_TISSUE_NAMES",
    "BUILT_IN_DIR_PATH",
    "BUILT_IN_DTI_FILE_PATH",
    "DEBUG",
    "DEFAULT_STS_TOKEN_DURATION_SECONDES",
    "DEFAULT_STS_ROLE_SESSION_NAME",
    "EEG10_10_CUTINI_2011",
    "EEG10_10_NEUROELECTRICS",
    "EEG10_10_UI_JURAK_2007",
    "EEG10_20_OKAMOTO_2004",
    "EEG10_20_EXTENDED_SPM12",
    "cond_dict_to_list",
    "export_ti_to_nifti",
    "find_montage_file",
]
