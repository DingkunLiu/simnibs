"""
CHARM 步骤3: 输入图像降噪

使用 SANLM (Statistically Adaptive Non-Local Means) 滤波对输入图像进行降噪。

原理：
    SANLM 滤波是一种自适应非局部均值降噪方法，通过利用图像中的自相似性
    来去除噪声，同时保留结构信息。降噪后的图像有利于后续的分割步骤。

输入：
    - T1fs.nii.gz (必须)
    - T2_reg.nii.gz (可选)

输出：
    - T1fs_denoised.nii.gz
    - T2_reg_denoised.nii.gz (如果存在 T2)

用法：
    python -m neuracle.charm.denoise <subid>
"""

import logging
import os

from neuracle.utils.charm_utils import read_settings
from simnibs.segmentation import charm_utils
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


def denoise_inputs(subject_dir: str) -> None:
    """
    对输入图像进行降噪

    使用 SANLM (Statistically Adaptive Non-Local Means) 滤波对输入图像进行降噪。
    降噪后的图像有利于后续的分割步骤。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})

    Returns
    -------
    None

    See Also
    --------
    _denoise_if_needed : 条件降噪辅助函数
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    settings = read_settings()
    denoise_settings = settings["preprocess"]
    do_denoise = denoise_settings["denoise"]
    if not do_denoise:
        logger.info("设置中已禁用降噪，跳过。")
        return
    logger.info("开始对输入图像进行降噪")
    _denoise_if_needed(
        sub_files.reference_volume,
        sub_files.T1_denoised,
        "T1",
    )
    _denoise_if_needed(
        sub_files.T2_reg,
        sub_files.T2_reg_denoised,
        "registered T2",
    )
    logger.info("降噪完成")


def _denoise_if_needed(
    input_path: str,
    output_path: str,
    name: str,
) -> None:
    """
    如果输入文件存在且降噪输出缺失或过期，则执行降噪

    Parameters
    ----------
    input_path : str
        输入文件路径
    output_path : str
        输出文件路径
    name : str
        文件类型名称（用于日志）
    """
    if not os.path.exists(input_path):
        return
    if not _should_regenerate_denoised(input_path, output_path):
        logger.info("%s 降噪文件已存在: %s", name, output_path)
    else:
        logger.info("正在对 %s 进行降噪并保存。", name)
        charm_utils._denoise_input_and_save(input_path, output_path)


def _should_regenerate_denoised(
    input_path: str,
    output_path: str,
) -> bool:
    """
    判断是否需要重建降噪输出。

    Parameters
    ----------
    input_path : str
        输入文件路径
    output_path : str
        输出文件路径

    Returns
    -------
    bool
        当输出缺失或早于输入时返回 True
    """
    if not os.path.exists(output_path):
        return True
    return os.path.getmtime(output_path) < os.path.getmtime(input_path)
