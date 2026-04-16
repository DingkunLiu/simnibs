"""
CHARM 流程辅助工具函数

本模块提供 CHARM 分割流程中所需的辅助函数，包括：
    - qform/sform 编码检查与修正
    - CHARM 配置文件的读取
    - Atlas 路径和参数的设置

用法：
    from neuracle.utils.charm_utils import check_q_and_s_form, read_settings
"""

import logging
import os

import nibabel as nib
import numpy as np

from simnibs import SIMNIBSDIR
from simnibs.utils import file_finder, settings_reader

logger = logging.getLogger(__name__)


def check_q_and_s_form(
    scan: nib.Nifti1Image,
    force_qform: bool = False,
    force_sform: bool = False,
) -> nib.Nifti1Image:
    """
    检查并修正 qform/sform 编码。

    Parameters
    ----------
    scan : nib.Nifti1Image
        输入 nifti 图像
    force_qform : bool
        强制使用 qform
    force_sform : bool
        强制使用 sform

    Returns
    -------
    nib.Nifti1Image
        修正后的图像
    """
    if not scan.get_qform(coded=True)[1] > 0 and force_sform is False:
        logger.error("qform_code 为 0，请检查输入图像的头部信息: %s", scan)
        raise ValueError(
            "qform_code 为 0，请检查输入图像的头部信息。"
            "可以使用 --forcesform 选项强制使用 sform。"
        )
    if not np.allclose(scan.get_qform(), scan.get_sform(), rtol=1e-5, atol=1e-6):
        if not (force_qform or force_sform):
            logger.error("qform 和 sform 矩阵不匹配")
            raise ValueError(
                "qform 和 sform 矩阵不匹配。"
                "请使用 --forceqform（推荐）或 --forcesform 选项。"
            )
        if force_qform:
            (qmat, qcode) = scan.get_qform(coded=True)
            scan.set_sform(qmat, code=qcode)
        elif force_sform:
            if not scan.get_sform(coded=True)[1] > 0:
                logger.error("sform_code 为 0 但被强制使用")
                raise ValueError(
                    "sform_code 为 0 但被强制使用。请修复 sform_code 或使用 qform。"
                )
            (mat_tmp, code_tmp) = scan.get_sform(coded=True)
            scan.set_qform(mat_tmp, code=code_tmp)
            (mat_tmp, code_tmp) = scan.get_qform(coded=True)
            scan.set_sform(mat_tmp, code=code_tmp)
    return scan


def read_settings() -> dict:
    """
    读取 CHARM 设置文件。

    Returns
    -------
    dict
        设置字典
    """
    src_settings = os.path.join(SIMNIBSDIR, "charm.ini")
    return settings_reader.read_ini(src_settings)


def setup_atlas(
    samseg_settings: dict,
    t2_reg: str,
    use_settings: str | None,
) -> tuple:
    """
    设置 Atlas 路径和参数。

    Parameters
    ----------
    samseg_settings : dict
        Samseg 设置
    t2_reg : str
        T2 配准后文件路径
    use_settings : str or None
        自定义设置文件路径

    Returns
    -------
    tuple
        (template_name, atlas_settings, atlas_path, atlas_level1, atlas_level2,
         atlas_affine_name, gmm_parameters)
    """
    atlas_name = samseg_settings["atlas_name"]
    logger.info("使用 %s 作为 charm atlas。", atlas_name)
    atlas_path = os.path.join(file_finder.templates.charm_atlas_path, atlas_name)
    atlas_settings = settings_reader.read_ini(
        os.path.join(atlas_path, atlas_name + ".ini")
    )
    atlas_settings_names = atlas_settings["names"]
    template_name = os.path.join(atlas_path, atlas_settings_names["template_name"])
    atlas_affine_name = os.path.join(atlas_path, atlas_settings_names["affine_atlas"])
    atlas_level1 = os.path.join(atlas_path, atlas_settings_names["atlas_level1"])
    atlas_level2 = os.path.join(atlas_path, atlas_settings_names["atlas_level2"])
    custom_gmm_parameters = samseg_settings["gmm_parameter_file"]
    if not use_settings or not custom_gmm_parameters:
        if os.path.exists(t2_reg):
            gmm_parameters = os.path.join(
                atlas_path, atlas_settings_names["gaussian_parameters_t2"]
            )
        else:
            gmm_parameters = os.path.join(
                atlas_path, atlas_settings_names["gaussian_parameters_t1"]
            )
    else:
        settings_dir = os.path.dirname(use_settings)
        gmm_parameters = os.path.join(settings_dir, custom_gmm_parameters)
        if not os.path.exists(gmm_parameters):
            logger.error("找不到 GMM 参数文件: %s", gmm_parameters)
            raise FileNotFoundError(f"找不到 GMM 参数文件: {gmm_parameters}")
    return (
        template_name,
        atlas_settings,
        atlas_path,
        atlas_level1,
        atlas_level2,
        atlas_affine_name,
        gmm_parameters,
    )
