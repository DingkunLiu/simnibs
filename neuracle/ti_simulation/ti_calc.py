"""
TI Calc 模块 - Temporal Interference 计算

提供 TI 场计算功能，根据两组电极的仿真结果计算干涉场。

原理：
    Temporal Interference (TI) 是一种无创脑刺激技术，利用两个高频电场 (f1, f2)
    的干涉效应来精确调控深部脑区。两个高频电场的频率差 (Δf = |f1 - f2|)
    决定了干涉产生的低频调制包络的频率。

    TI 最大调制振幅的计算方法（来自 SimNIBS TI_utils）：
    对于每个位置，计算两个电场向量 (E1, E2) 的所有可能线性组合中，
    使得合成电场幅度最大的那个组合的幅度：
    max_TI = max_{α∈[0,2π]} |E1 + α * E2|

    网格裁剪说明：
    仿真结果包含完整头部模型（头皮、颅骨、脑脊液、灰质、白质等）。
    TI 计算只需要保留脑组织区域，去除头皮、颅骨等外部结构以及
    电极和盐水（saline）层。

用法：
    from neuracle.ti_simulation import calculate_ti

    ti_mesh_path = calculate_ti(
        mesh1_path="path/to/TDCS_1_scalar.msh",
        mesh2_path="path/to/TDCS_2_scalar.msh",
        output_dir="path/to/output"
    )
"""

import logging
import os
from copy import deepcopy

import numpy as np

from simnibs import ElementTags, mesh_io

logger = logging.getLogger(__name__)


def calculate_ti(
    mesh1_path: str,
    mesh2_path: str,
    output_dir: str,
) -> str:
    """
    计算 TI 场

    从两个仿真结果网格文件计算 Temporal Interference 最大调制振幅。

    Parameters
    ----------
    mesh1_path : str
        第一组电极仿真结果网格路径（对应频率 f1）
    mesh2_path : str
        第二组电极仿真结果网格路径（对应频率 f2）
    output_dir : str
        输出目录，存放 TI 计算结果

    Returns
    -------
    str
        TI 结果网格路径 ({base_name}_TI.msh)

    Raises
    ------
    FileNotFoundError
        如果网格文件不存在
    ValueError
        如果网格文件不包含 E 场数据

    Notes
    -----
    输出文件：
    - {output_dir}/{base_name}_TI.msh: TI 场网格文件，包含 max_TI 字段
    - {output_dir}/{base_name}_TI.msh.png: Gmsh 可视化脚本

    TI 最大调制振幅的范围通常是 [0, ~25] V/m（取决于刺激强度）。
    """
    logger.info("开始计算 TI 场...")
    logger.info("读取网格文件: %s, %s", mesh1_path, mesh2_path)

    if not os.path.exists(mesh1_path):
        raise FileNotFoundError(f"网格文件不存在: {mesh1_path}")
    if not os.path.exists(mesh2_path):
        raise FileNotFoundError(f"网格文件不存在: {mesh2_path}")

    m1 = mesh_io.read_msh(mesh1_path)
    m2 = mesh_io.read_msh(mesh2_path)

    logger.info("裁剪网格，保留脑组织区域...")
    tags_keep = np.hstack(
        (
            np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
            np.arange(
                ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1
            ),
        )
    )
    m1 = m1.crop_mesh(tags=tags_keep)
    m2 = m2.crop_mesh(tags=tags_keep)

    logger.info("提取电场数据并计算 TI 最大调制振幅...")
    ef1 = m1.field["E"]
    ef2 = m2.field["E"]

    if ef1 is None or ef2 is None:
        raise ValueError("网格文件不包含 E 场数据")

    from simnibs.utils import TI_utils as TI

    ti_max = TI.get_maxTI(ef1.value, ef2.value)

    logger.info("生成 TI 可视化输出...")
    mout = deepcopy(m1)
    mout.elmdata = []
    mout.add_element_field(ti_max, "max_TI")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(mesh1_path).replace("_TDCS_1_scalar.msh", "")
    ti_mesh_path = os.path.join(output_dir, f"{base_name}_TI.msh")
    mesh_io.write_msh(mout, ti_mesh_path)

    v = mout.view(
        visible_tags=[1002, 1006],
        visible_fields="max_TI",
    )
    v.write_opt(ti_mesh_path)

    logger.info("TI 计算完成，输出文件: %s", ti_mesh_path)
    return ti_mesh_path
