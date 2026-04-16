"""
TI 导出工具函数

提供将 TI (Temporal Interference) 仿真结果从 mesh 导出到 nifti 格式的功能。
"""

import logging
import os

from simnibs import ElementTags, mesh_io
from simnibs.utils.transformations import interpolate_to_volume

logger = logging.getLogger(__name__)


def export_ti_to_nifti(
    msh_path: str,
    output_dir: str,
    reference: str,
    field_name: str,
    prefix: str = "",
) -> str:
    """
    将 TI 场从 mesh 导出到 nifti 格式。

    Parameters
    ----------
    msh_path : str
        源 mesh 文件路径
    output_dir : str
        输出目录
    reference : str
        参考 nifti 文件路径（用于确定输出空间）
    field_name : str
        mesh 中使用的字段名，固定为 "max_TI"
    prefix : str
        文件名前缀，如 "{subid}_simulation" 或 "{subid}_optimization"，
        最终输出为 {prefix}_{field_name}.nii.gz

    Returns
    -------
    str
        nifti 文件路径

    Notes
    -----
    interpolate_to_volume 会在输出文件名后加 _field_name，
    所以传 "{prefix}.nii.gz" 会生成 "{prefix}_{field_name}.nii.gz"
    """
    logger.info("将 TI 结果导出到 nifti 格式...")
    mesh = mesh_io.read_msh(msh_path)

    # interpolate_to_volume 会在输出文件名后加 _field_name
    # 所以传 "{prefix}.nii.gz" 会生成 "{prefix}_{field_name}.nii.gz"
    nifti_path = os.path.join(output_dir, f"{prefix}.nii.gz")
    interpolate_to_volume(
        mesh,
        reference,
        nifti_path,
        method="linear",
        keep_tissues=[ElementTags.WM, ElementTags.GM],
    )

    # 返回实际生成的文件路径
    actual_path = os.path.join(output_dir, f"{prefix}_{field_name}.nii.gz")
    logger.info("nifti 导出完成: %s", actual_path)
    return actual_path
