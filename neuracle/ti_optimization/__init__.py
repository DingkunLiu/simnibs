"""
TI Optimization 模块 - Temporal Interference 逆向仿真优化

基于 SimNIBS 的 TesFlexOptimization 框架，提供四种优化目标函数。

目标函数说明：
    - mean: 最大化 ROI 内平均电场
    - max: 最小化 ROI 内最大电场
    - focality: 最大化 ROI 内 focality（聚焦度）
    - focality_inv: 反向 focality 优化

导出 API：
    - init_optimization: 初始化优化结构
    - setup_goal: 配置目标函数
    - setup_electrodes_and_roi: 配置电极和 ROI
    - run_optimization: 运行优化
    - get_electrode_mapping: 获取电极映射结果
"""

from neuracle.ti_optimization.electrodes_roi import setup_electrodes_and_roi
from neuracle.ti_optimization.goal import setup_goal
from neuracle.ti_optimization.init import init_optimization
from neuracle.ti_optimization.result import get_electrode_mapping
from neuracle.ti_optimization.run import run_optimization

__all__ = [
    "init_optimization",
    "setup_goal",
    "setup_electrodes_and_roi",
    "run_optimization",
    "get_electrode_mapping",
]
