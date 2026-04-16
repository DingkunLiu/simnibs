# TI 逆向优化方案

## 概述

TI (Temporal Interference) 优化模块基于 SimNIBS 的 TesFlexOptimization 框架，提供四种优化目标函数，用于优化经颅电刺激中的电极配置。

## 优化目标

| 目标函数 | 说明 |
|---------|------|
| `mean` | 最大化 ROI 内平均电场 |
| `max` | 最小化 ROI 内最大电场 |
| `focality` | 最大化 ROI 内 focality（聚焦度） |
| `focality_inv` | 反向 focality 优化 |

## 原理

1. **ElectrodeArrayPair 配置**：使用双电极阵列对，每组电极对阵列中心、半径和电流可独立配置
2. **球形 ROI**：使用球形区域定义优化目标区域
3. **Focality 优化**：额外定义 Non-ROI 区域，计算 ROI 与 Non-ROI 的电场比率
4. **EEG 电极映射**：支持将优化后的电极位置映射到标准 EEG 网格

## 使用方法

```python
from neuracle.ti_optimization import (
    init_optimization,
    setup_goal,
    setup_electrodes_and_roi,
    run_optimization,
)

# 1. 初始化优化结构
opt = init_optimization(subject_dir, output_dir)

# 2. 配置目标函数
setup_goal(opt, goal="focality", focality_threshold=[0.1, 0.2])

# 3. 配置电极和 ROI
setup_electrodes_and_roi(
    opt,
    goal="focality",
    roi_center=[-41.0, -13.0, 66.0],
    roi_radius=20.0,
    electrode_pair1_center=[[0, 0]],
    electrode_pair2_center=[[0, 0]],
    electrode_radius=[10],
)

# 4. 运行优化
output_dir = run_optimization(opt, n_workers=24)
```

## 模块结构

```
neuracle/ti_optimization/
├── __init__.py              # 模块初始化，导出主要 API
├── init.py                   # init_optimization 函数
├── goal.py                   # setup_goal 函数
├── electrodes_roi.py         # setup_electrodes_and_roi 函数
├── run.py                    # run_optimization 函数
├── result.py                 # get_electrode_mapping 函数
└── demo/                     # 示例代码
    ├── ti_mean_optimize_demo.py          # Mean 优化示例
    ├── ti_max_optimize_demo.py           # Max 优化示例
    ├── ti_focality_optimize_demo.py     # Focality 优化示例
    └── ti_focality_inv_optimize_demo.py # Focality Inverse 优化示例
```

## 导出 API

```python
from neuracle.ti_optimization import (
    init_optimization,           # 初始化优化结构
    setup_goal,                  # 配置目标函数
    setup_electrodes_and_roi,    # 配置电极和 ROI
    run_optimization,            # 运行优化
    get_electrode_mapping,       # 获取电极映射结果
)
```

## 参数说明

### init_optimization

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `subject_dir` | Subject 目录（m2m_{subid}） | 必需 |
| `output_dir` | 优化输出目录 | 必需 |
| `msh_file_path` | 头模网格文件路径 | None |
| `anisotropy_type` | 各向异性类型 | "scalar" |
| `cond` | 电导率列表 | None |
| `fname_tensor` | DTI 张量文件路径 | None |

### setup_goal

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `goal` | 目标函数类型："mean", "max", "focality", "focality_inv" | 必需 |
| `e_postproc` | E-field 后处理方式 | "max_TI" |
| `focality_threshold` | focality 阈值 | [0.1, 0.2] |
| `min_electrode_distance` | 电极最小距离（mm） | 5.0 |
| `map_to_net_electrodes` | 是否映射到 EEG 网格 | True |
| `net_electrode_file` | EEG 电极位置 CSV 文件 | None |
| `optimizer` | 优化算法 | "differential_evolution" |
| `polish` | 是否使用 L-BFGS-B 细化 | False |
| `run_final_simulation` | 是否运行最终仿真 | True |

### setup_electrodes_and_roi

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `goal` | 目标函数类型 | 必需 |
| `mesh_file_path` | 头模网格文件路径 | None |
| `electrode_pair1_center` | 第一组电极阵列中心位置 | [[0, 0]] |
| `electrode_pair2_center` | 第二组电极阵列中心位置 | [[0, 0]] |
| `electrode_radius` | 电极半径 | [10] |
| `electrode_current1` | 第一组电极电流 | [0.002, -0.002] |
| `electrode_current2` | 第二组电极电流 | [0.002, -0.002] |
| `roi_center` | ROI 球形区域中心 | [-41.0, -13.0, 66.0] |
| `roi_radius` | ROI 球形区域半径（mm） | 20.0 |
| `roi_center_space` | ROI 坐标空间 ("subject" 或 "mni") | "subject" |
| `roi_mask_path` | ROI mask 文件路径 | None |
| `non_roi_center` | Non-ROI 球形区域中心 | None |
| `non_roi_radius` | Non-ROI 球形区域半径 | None |

### run_optimization

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `n_workers` | 并行计算的 CPU 核心数 | None |

### get_electrode_mapping

| 参数 | 说明 |
|------|------|
| `output_dir` | 优化输出目录 |

返回：`tuple[list[str], list[str]]` - (electrode_A, electrode_B) 两组电极的映射标签列表
