"""
TI Optimization 初始化模块

原理：
    1. 基于 SimNIBS 的 TesFlexOptimization 框架
    2. 初始化优化结构，设置电导率、各向异性类型等基本参数
    3. 支持自定义电导率和 DTI 张量

用法：
    from neuracle.ti_optimization import init_optimization

    opt = init_optimization(subject_dir, output_dir)
"""

import logging
import os

from simnibs import opt_struct

logger = logging.getLogger(__name__)


def init_optimization(
    subject_dir: str,
    output_dir: str,
    msh_file_path: str | None = None,
    anisotropy_type: str = "scalar",
    cond: list | None = None,
    fname_tensor: str | None = None,
) -> opt_struct.TesFlexOptimization:
    """
    初始化优化结构

    Parameters
    ----------
    subject_dir : str
        Subject 目录（m2m_{subid}）
    msh_file_path : str, optional
        头模网格文件路径；如果提供，则优先使用该 mesh 文件
    output_dir : str
        优化输出目录
    anisotropy_type : str
        各向异性类型 (default: "scalar")
    cond : list, optional
        电导率列表
    fname_tensor : str, optional
        DTI 张量文件路径

    Returns
    -------
    opt_struct.TesFlexOptimization
        初始化好的优化对象
    """
    opt = opt_struct.TesFlexOptimization()
    opt.subpath = subject_dir
    opt.fn_mesh = msh_file_path
    opt.output_folder = output_dir
    opt.open_in_gmsh = False
    opt.anisotropy_type = anisotropy_type
    logger.info("电导率各向异性类型: %s", anisotropy_type)

    if cond is not None:
        opt.cond = cond
        logger.info("使用自定义电导率值")
    else:
        logger.info("使用标准电导率值 (standard_cond)")

    if fname_tensor is not None:
        opt.fname_tensor = fname_tensor
        logger.info("DTI 张量文件: %s", fname_tensor)
    if msh_file_path is not None:
        logger.info("头模网格文件: %s", msh_file_path)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("优化结构初始化完成")

    return opt
