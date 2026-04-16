# TI 正向仿真方案

## 1. 概述

### 1.1 目标
在 `neuracle/ti_simulation/` 下创建 TI（Temporal Interference，时间干涉）正向仿真模块，提供从电极对设置到 TI 场计算的全流程功能。

### 1.2 参考文档
- SimNIBS 官方文档：https://simnibs.github.io/simnibs/build/html/documentation/documentation.html
- TI.py 示例：`simnibs/examples/simulations/TI.py`
- TI_utils.py 工具函数：`simnibs/utils/TI_utils.py`

---

## 2. 文件结构

```
neuracle/
├── ti_simulation/
│   ├── __init__.py              # 包初始化，导出公共 API
│   ├── session.py               # 会话配置：setup_session
│   ├── electrode.py             # 电极配置：setup_electrode_pair1, setup_electrode_pair2, _setup_electrode_pair (内部函数)
│   ├── run.py                   # 仿真执行：run_tdcs_simulation
│   ├── ti_calc.py              # TI 计算：calculate_ti
│   └── demo/
│       └── ti_simulation_demo.py # 示例代码
└── docs/
    └── TI 正向仿真方案.md        # 本文档
```

---

## 3. 函数设计

### 3.1 `setup_session()`

**功能**：配置 SimNIBS 会话参数。

**参数**：
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `subject_dir` | str | Subject 目录（m2m_{subid}） |
| `output_dir` | str | 仿真输出目录 |
| `msh_file_path` | str, optional | 头模网格文件路径 |
| `anisotropy_type` | str | 电导率各向异性类型 (default: "scalar") |
| `cond` | list, optional | 电导率列表 |
| `fname_tensor` | str, optional | DTI 张量文件路径 |
| `eeg_cap` | str, optional | EEG 电极帽 CSV 文件路径 |

**返回值**：`sim_struct.SESSION` - 配置好的会话对象

### 3.2 `setup_electrode_pair1()`

**功能**：配置第一个电极对。

**参数**：
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `session` | sim_struct.SESSION | SimNIBS 会话对象 |
| `electrode_pair1` | list[str] | 第一个电极对 [elec1_name, elec2_name]，如 ['F5', 'P5'] |
| `current1` | list[float] | 第一组电极对电流列表 [anode_current, cathode_current]，单位 A |
| `electrode_shape` | str | 电极形状 (default: "ellipse") |
| `electrode_dimensions` | list[float] | 电极尺寸 [width, height] (default: [40, 40]) |
| `electrode_thickness` | float | 电极厚度 (default: 2.0) |

**返回值**：`sim_struct.TDCSLIST` - 配置好的 TDCS 列表对象

### 3.3 `setup_electrode_pair2()`

**功能**：配置第二个电极对。

**参数**：与 `setup_electrode_pair1()` 相同，用于配置第二组电极对。

**返回值**：`sim_struct.TDCSLIST` - 配置好的 TDCS 列表对象

### 3.4 `run_tdcs_simulation()`

**功能**：执行两组电极的 TDCS 正向仿真。

**参数**：
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `session` | sim_struct.SESSION | SimNIBS 会话对象 |
| `subject_dir` | str | Subject 目录（用于提取 subid） |
| `output_dir` | str | 仿真输出目录 |
| `n_workers` | int | 并行工作进程数 (default: 1) |

**返回值**：`tuple[str, str]` - (mesh1_path, mesh2_path) 两组电极仿真结果网格路径

**输出文件**：
- `{output_dir}/{subid}_TDCS_1_scalar.msh` - 第一对电极仿真结果
- `{output_dir}/{subid}_TDCS_2_scalar.msh` - 第二对电极仿真结果

### 3.5 `calculate_ti()`

**功能**：从两个仿真结果计算 TI 场。

**参数**：
| 参数名 | 类型 | 说明 |
|--------|------|------|
| `mesh1_path` | str | 第一组电极仿真结果网格路径 |
| `mesh2_path` | str | 第二组电极仿真结果网格路径 |
| `output_dir` | str | 输出目录 |

**返回值**：`str` - TI 结果网格路径

**输出文件**：
- `{output_dir}/TI.msh` - TI 场可视化结果

---

## 4. 实现步骤

### 步骤 1：创建文件夹和基础文件
```
neuracle/ti_simulation/
├── __init__.py
├── session.py
├── electrode.py
├── run.py
├── ti_calc.py
└── demo/
    └── ti_simulation_demo.py
```

### 步骤 2：实现 session.py
- 实现 `setup_session()` 函数
- 配置 SESSION 参数（`open_in_gmsh = False` 禁止自动打开 Gmsh）

### 步骤 3：实现 electrode.py
- 实现 `setup_electrode_pair1()` 函数
- 实现 `setup_electrode_pair2()` 函数
- 实现 `_setup_electrode_pair()` 内部辅助函数

### 步骤 4：实现 run.py
- 实现 `run_tdcs_simulation()` 函数
- 调用 `run_simnibs(S, cpus=n_workers)` 执行仿真
- 返回两组仿真结果路径

### 步骤 5：实现 ti_calc.py
- 实现 `calculate_ti()` 函数
- 读取网格文件
- 裁剪网格（去除电极元素）
- 使用 TI_utils 计算 TI 最大调制振幅
- 生成可视化输出

### 步骤 6：更新 __init__.py
- 导出公共 API：setup_session, setup_electrode_pair1, setup_electrode_pair2, run_tdcs_simulation, calculate_ti

---

## 5. 关键代码片段

### 5.1 会话配置
```python
S = sim_struct.SESSION()
S.subpath = subject_dir
S.pathfem = output_dir
S.open_in_gmsh = False  # 禁止自动打开 Gmsh
```

### 5.2 电极对配置
```python
tdcs = session.add_tdcslist()
tdcs.currents = currents  # [current1, -current1]，电流方向相反

electrode = tdcs.add_electrode()
electrode.channelnr = 1
electrode.centre = electrode_pair[0]  # 如 'F5'
electrode.shape = electrode_shape
electrode.dimensions = electrode_dimensions
electrode.thickness = electrode_thickness
```

### 5.3 TI 计算
```python
from simnibs.utils import TI_utils as TI
ef1 = m1.field['E']
ef2 = m2.field['E']
ti_max = TI.get_maxTI(ef1.value, ef2.value)
```

### 5.4 网格裁剪（去除电极元素）
```python
tags_keep = np.hstack((
    np.arange(ElementTags.TH_START, ElementTags.SALINE_START - 1),
    np.arange(ElementTags.TH_SURFACE_START, ElementTags.SALINE_TH_SURFACE_START - 1)
))
m1 = m1.crop_mesh(tags=tags_keep)
m2 = m2.crop_mesh(tags=tags_keep)
```

---

## 6. Demo 示例设计

### 6.1 `ti_simulation_demo.py`

**功能**：演示如何使用 `run_ti_simulation()` 函数执行 TI 正向仿真。

**数据来源**：`data/m2m_ernie/`

**配置**：
- 电极对1：F5-P5，电流 1mA
- 电极对2：F6-P6，电流 1mA
- 电极形状：椭圆 40x40 mm
- 电极厚度：2 mm
- n_workers: 24

**输出**：
- `{output_dir}/model_TDCS_1_scalar.msh` - 第一对电极仿真结果
- `{output_dir}/model_TDCS_2_scalar.msh` - 第二对电极仿真结果
- `{output_dir}/TI.msh` - TI 场可视化结果
- `{output_dir}/TI.nii.gz` - TI 场 NIfTI 格式（通过 `export_ti_to_nifti` 导出）

