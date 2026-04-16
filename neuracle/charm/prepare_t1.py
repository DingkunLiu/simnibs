"""
CHARM 步骤1: T1 图像准备

将原始 T1 加权 MRI 图像转换为 SimNIBS 所需格式，执行 qform/sform 检查。

原理：
    1. 加载原始 T1 NIfTI 文件
    2. 检查并修正 qform/sform 编码
    3. 去除单例维度（singleton dimensions）
    4. 转换为 float32 并保存到 subject_dir

用法：
    python -m neuracle.charm.prepare_t1 <subid> <T1_file> [--forcesform]
"""

import logging

import nibabel as nib
import numpy as np

from neuracle.utils.charm_utils import check_q_and_s_form
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


def prepare_t1(
    subject_dir: str,
    t1: str,
    force_qform: bool = False,
    force_sform: bool = False,
) -> None:
    """
    准备 T1 图像

    将原始 T1 加权 MRI 图像转换为 SimNIBS 所需格式，执行 qform/sform 检查、
    去除单例维度并转换为 float32。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})
    t1 : str
        输入 T1 NIfTI 文件路径
    force_qform : bool, optional
        强制使用 qform 编码 (default: False)
    force_sform : bool, optional
        强制使用 sform 编码 (default: False)

    Returns
    -------
    None

    Raises
    ------
    ValueError
        当 force_qform 和 force_sform 同时为 True 时抛出

    See Also
    --------
    check_q_and_s_form : qform/sform 检查函数
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    logger.info("正在准备 T1 图像: %s", t1)
    if force_qform and force_sform:
        logger.error("不能同时强制使用 qform 和 sform")
        raise ValueError("Cannot force both qform and sform, please use only one")
    t1_tmp = nib.load(t1)
    t1_tmp = check_q_and_s_form(t1_tmp, force_qform, force_sform)
    if (np.array(t1_tmp.shape) == 1).any():
        logger.info("检测到单例维度，正在去除")
        data_tmp = np.squeeze(t1_tmp.get_fdata())
        t1_tmp = nib.Nifti1Image(data_tmp, t1_tmp.affine)
    t1_tmp.set_data_dtype(np.float32)
    nib.save(t1_tmp, sub_files.reference_volume)
    logger.info("T1 图像已保存至: %s", sub_files.reference_volume)
