"""
TI Optimization 运行模块

原理：
    1. 调用 SimNIBS 的 TesFlexOptimization.run() 执行优化
    2. 支持多核并行计算

用法：
    from neuracle.ti_optimization import run_optimization

    output_dir = run_optimization(opt, n_workers=24)
"""

import logging

from simnibs import opt_struct

logger = logging.getLogger(__name__)


def run_optimization(
    opt: opt_struct.TesFlexOptimization,
    n_workers: int | None = None,
) -> str:
    """
    运行优化算法

    Parameters
    ----------
    opt : opt_struct.TesFlexOptimization
        优化对象
    n_workers : int, optional
        并行计算的 CPU 核心数

    Returns
    -------
    str
        优化结果输出目录
    """
    logger.info("开始优化...")
    opt.run(cpus=n_workers)
    logger.info("优化完成，结果保存在: %s", opt.output_folder)
    return opt.output_folder
