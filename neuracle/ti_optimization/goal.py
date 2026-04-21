"""
TI Optimization 目标函数配置模块

原理：
    1. 配置优化目标函数（mean, max, focality, focality_inv）
    2. 设置 E-field 后处理方式
    3. 配置优化算法和相关参数

用法：
    from neuracle.ti_optimization import setup_goal

    setup_goal(opt, goal="focality", focality_threshold=[0.1, 0.2])
"""

import logging

from simnibs import opt_struct

logger = logging.getLogger(__name__)


def setup_goal(
    opt: opt_struct.TesFlexOptimization,
    goal: str,
    e_postproc: str = "max_TI",
    focality_threshold: list[float] | None = None,
    min_electrode_distance: float = 5.0,
    constrain_electrode_locations: bool = False,
    overlap_factor: float = 1.0,
    map_to_net_electrodes: bool = True,
    run_mapped_electrodes_simulation: bool = True,
    net_electrode_file: str | None = None,
    optimizer: str = "differential_evolution",
    polish: bool = False,
    run_final_simulation: bool = True,
    disable_SPR_for_volume_roi: bool = True,
    weights: list[float] | None = None,
    track_focality: bool = False,
    detailed_results: bool = False,
) -> None:
    """
    配置目标函数和优化选项

    Parameters
    ----------
    opt : opt_struct.TesFlexOptimization
        优化对象
    goal : str
        目标函数类型："mean", "max", "focality", "focality_inv"
    e_postproc : str
        E-field 后处理方式 (default: "max_TI")
    focality_threshold : list[float], optional
        focality 阈值
    min_electrode_distance : float
        电极最小距离（mm）
    constrain_electrode_locations : bool
        是否限制电极阵列位置
    overlap_factor : float
        允许 lambda 区域的重叠因子
    map_to_net_electrodes : bool
        是否将优化电极映射到 EEG 网格
    run_mapped_electrodes_simulation : bool
        是否用映射位置运行仿真
    net_electrode_file : str, optional
        EEG 电极位置 CSV 文件路径
    optimizer : str
        优化算法
    polish : bool
        是否使用 L-BFGS-B 细化
    run_final_simulation : bool
        是否运行最终仿真
    disable_SPR_for_volume_roi : bool
        是否对体素 ROI 禁用 SPR 插值
    weights : list[float], optional
        ROI 特定权重
    track_focality : bool
        是否追踪 focality 变化
    detailed_results : bool
        是否写入详细结果
    """
    opt.goal = [goal]
    opt.e_postproc = e_postproc

    if goal in ["focality", "focality_inv"]:
        if focality_threshold is None:
            focality_threshold = [0.1, 0.2]
        opt.threshold = focality_threshold

    opt.min_electrode_distance = min_electrode_distance
    opt.constrain_electrode_locations = constrain_electrode_locations
    opt.overlap_factor = overlap_factor
    opt.map_to_net_electrodes = map_to_net_electrodes
    opt.run_mapped_electrodes_simulation = run_mapped_electrodes_simulation
    opt.net_electrode_file = net_electrode_file
    opt.optimizer = optimizer
    opt.polish = polish
    opt.run_final_electrode_simulation = run_final_simulation
    opt.disable_SPR_for_volume_roi = disable_SPR_for_volume_roi
    if weights is not None:
        opt.weights = weights
    opt.track_focality = track_focality
    opt.detailed_results = detailed_results

    logger.info("目标函数配置完成: %s", goal)
