"""
Run 模块 - TDCS 仿真执行

提供 TDCS 仿真执行功能，运行 SimNIBS 进行电场计算。

原理：
    SimNIBS 通过有限元方法 (FEM) 计算头部模型中的电场分布。
    对于 TI (Temporal Interference) 仿真，需要分别运行两次 TDCS 仿真：
    1. 第一组电极对产生的电场分布 (对应频率 f1)
    2. 第二组电极对产生的电场分布 (对应频率 f2)

    仿真结果为网格文件，包含以下信息：
    - 网格节点坐标
    - 网格元素（四面体）
    - 各组织的电导率
    - 电场强度 (E field)
    - 电流密度 (J field)

用法：
    from neuracle.ti_simulation import run_tdcs_simulation

    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir="path/to/m2m_subid",
        output_dir="path/to/output",
        n_workers=24
    )
"""

import logging
import os

from simnibs import run_simnibs, sim_struct

logger = logging.getLogger(__name__)


def run_tdcs_simulation(
    session: sim_struct.SESSION,
    subject_dir: str,
    output_dir: str,
    n_workers: int = 1,
) -> tuple[str, str]:
    """
    运行 TDCS 仿真

    执行两组电极的 TDCS 正向仿真，计算电场分布。

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象，已配置好电极对
    subject_dir : str
        Subject 目录（用于提取 subid）
    output_dir : str
        仿真输出目录
    n_workers : int, optional
        并行工作进程数，用于多线程加速仿真计算
        (default: 1)

    Returns
    -------
    tuple[str, str]
        (mesh1_path, mesh2_path) - 两组电极仿真结果网格路径
        - mesh1_path: 第一对电极仿真结果，如 {output_dir}/{subid}_TDCS_1_scalar.msh
        - mesh2_path: 第二对电极仿真结果，如 {output_dir}/{subid}_TDCS_2_scalar.msh

    Raises
    ------
    RuntimeError
        如果 SimNIBS 仿真执行失败

    Notes
    -----
    仿真结果文件格式为 .msh (Gmsh 格式)，包含：
    - 网格几何信息（节点、元素）
    - 物理标记（不同组织的标签）
    - 场数据（电场 E、电流密度 J 等）

    仿真过程会自动创建必要的输出目录。
    """
    logger.info("开始运行 TDCS 仿真，n_workers=%s", n_workers)
    logger.info("仿真配置: subject_dir=%s, output_dir=%s", subject_dir, output_dir)

    try:
        run_simnibs(session, cpus=n_workers)
        logger.info("TDCS 仿真完成")
    except Exception as e:
        logger.error("TDCS 仿真执行失败: %s", str(e))
        raise RuntimeError(f"TDCS 仿真执行失败: {str(e)}") from e

    base_name = os.path.splitext(os.path.basename(session.fnamehead))[0]
    mesh1_path = os.path.join(output_dir, f"{base_name}_TDCS_1_scalar.msh")
    mesh2_path = os.path.join(output_dir, f"{base_name}_TDCS_2_scalar.msh")

    logger.info("TDCS 仿真结果路径: %s, %s", mesh1_path, mesh2_path)
    return mesh1_path, mesh2_path
