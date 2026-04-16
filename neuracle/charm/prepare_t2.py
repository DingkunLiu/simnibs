"""
CHARM 步骤2: T2 图像配准与准备

将 T2 加权图像配准到 T1 空间，或进行格式转换。

原理：
    1. 如果指定 registerT2=True，执行 T2-to-T1 刚性配准
    2. 否则仅进行格式转换和 qform/sform 检查
    3. 保存到 subject_dir/T2_reg.nii.gz

用法：
    python -m neuracle.charm.prepare_t2 <subid> <T2_file> [--register-t2]
"""

import logging
import os

import nibabel as nib
import numpy as np

from neuracle.utils.charm_utils import check_q_and_s_form
from simnibs.segmentation import charm_utils
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


def prepare_t2(
    subject_dir: str,
    t2: str,
    register_t2: bool = False,
    force_qform: bool = False,
    force_sform: bool = False,
) -> None:
    """
    准备 T2 图像

    将 T2 加权图像配准到 T1 空间，或仅进行格式转换和 qform/sform 检查。
    如果指定 register_t2=True，执行 T2-to-T1 刚性配准。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})
    t2 : str
        输入 T2 NIfTI 文件路径
    register_t2 : bool, optional
        是否执行 T2-to-T1 配准 (default: False)
    force_qform : bool, optional
        强制使用 qform 编码 (default: False)
    force_sform : bool, optional
        强制使用 sform 编码 (default: False)

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        当输入 T2 文件不存在时抛出

    See Also
    --------
    check_q_and_s_form : qform/sform 检查函数
    _registerT1T2 : T1-T2 配准函数
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    logger.info("正在准备 T2 图像: %s", t2)
    logger.info("配准 T2 到 T1: %s", register_t2)
    if not os.path.exists(t2):
        logger.error("找不到输入的 T2 文件: %s", t2)
        raise FileNotFoundError(f"Could not find input T2 file: {t2}")
    t2_tmp = nib.load(t2)
    t2_tmp = check_q_and_s_form(t2_tmp, force_qform, force_sform)
    if (np.array(t2_tmp.shape) == 1).any():
        logger.info("检测到单例维度，正在去除")
        data_tmp = np.squeeze(t2_tmp.get_fdata())
        t2_tmp = nib.Nifti1Image(data_tmp, t2_tmp.affine)
    if register_t2:
        nib.save(t2_tmp, sub_files.T2_reg)
        charm_utils._registerT1T2(
            sub_files.reference_volume,
            sub_files.T2_reg,
            sub_files.T2_reg,
        )
        logger.info("T2 图像已配准到 T1 并保存至: %s", sub_files.T2_reg)
    else:
        t2_tmp.set_data_dtype(np.float32)
        nib.save(t2_tmp, sub_files.T2_reg)
        logger.info("T2 图像已保存至: %s", sub_files.T2_reg)
