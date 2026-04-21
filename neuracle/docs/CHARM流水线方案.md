# CHARM Pipeline 步骤拆分方案

## 一、背景

`simnibs/segmentation/charm_main.py` 中的 `run()` 函数包含完整的 CHARM 分割流程。本方案将其拆分为独立的步骤脚本，支持单独运行某个步骤。

## 二、步骤划分

| 步骤 | 脚本文件             | 功能                         | 主函数               |
| ---- | -------------------- | ---------------------------- | -------------------- |
| 1    | `prepare_t1.py`      | T1 图像准备与格式转换        | `prepare_t1()`       |
| 2    | `prepare_t2.py`      | T2 图像配准与准备            | `prepare_t2()`       |
| 3    | `denoise.py`         | 输入图像降噪                 | `denoise_inputs()`   |
| 4    | `init_atlas.py`      | atlas 初始仿射配准与颈部校正 | `init_atlas()`       |
| 5    | `segment.py`         | 体积与表面分割               | `run_segmentation()` |
| 6    | `create_surfaces.py` | 皮层表面重建                 | `create_surfaces()`  |
| 7    | `mesh.py`            | 四面体网格生成               | `create_mesh_step()` |

## 三、目录结构

```
neuracle/charm/
├── __init__.py
├── prepare_t1.py      # 步骤1: T1准备
├── prepare_t2.py      # 步骤2: T2准备
├── denoise.py         # 步骤3: 降噪
├── init_atlas.py      # 步骤4: atlas初始化
├── segment.py         # 步骤5: 分割
├── create_surfaces.py # 步骤6: 表面重建
├── mesh.py            # 步骤7: 网格生成
└── demo/              # 各步骤示例代码
    ├── charm_prepare_t1_demo.py
    ├── charm_prepare_t2_demo.py
    ├── charm_denoise_demo.py
    ├── charm_init_atlas_demo.py
    ├── charm_segment_demo.py
    ├── charm_create_surfaces_demo.py
    └── charm_mesh_demo.py
```

## 四、每个步骤的输入输出

### 步骤1: prepare_t1.py

- **函数**: `prepare_t1(subject_dir: str, t1: str, force_qform: bool = False, force_sform: bool = False) -> None`
- **输入**: 原始 T1 NIfTI 文件
- **输出**: `m2m_{subid}/T1fs.nii.gz`
- **功能**: 格式转换、维度检查（去除单例维度）、qform/sform 校验、转换为 float32

### 步骤2: prepare_t2.py

- **函数**: `prepare_t2(subject_dir: str, t2: str, register_t2: bool = False, force_qform: bool = False, force_sform: bool = False) -> None`
- **输入**: 原始 T2 NIfTI 文件 + T1fs.nii.gz
- **输出**: `m2m_{subid}/T2_reg.nii.gz`
- **功能**: T2-to-T1 刚性配准（可选）、格式转换、qform/sform 校验

### 步骤3: denoise.py

- **函数**: `denoise_inputs(subject_dir: str) -> None`
- **输入**: T1fs.nii.gz, T2_reg.nii.gz（可选）
- **输出**: `m2m_{subid}/T1fs_denoised.nii.gz`, `T2_reg_denoised.nii.gz`（可选）
- **功能**: SANLM (Statistically Adaptive Non-Local Means) 滤波降噪

### 步骤4: init_atlas.py

- **函数**: `init_atlas(subject_dir: str, use_transform: str | None = None, init_transform: str | None = None, noneck: bool = False) -> None`
- **输入**: T1fs.nii.gz（+ T1fs_denoised.nii.gz）、T2_reg.nii.gz（可选）
- **输出**: `m2m_{subid}/segmentation/template_coregistered.nii.gz`
- **功能**: MNI atlas 仿射配准 + 颈部校正
- **支持初始化方式**:
  - `atlas`: 使用 atlas 方法（默认）
  - `mni`: 使用 MNI 模板进行仿射初始化
  - `trega`: 使用 BrainNet/TREGA 方法进行 MNI152 到 RAS 的仿射估计
- **支持**: 外部变换矩阵输入（`use_transform`、`init_transform`）

#### 初始化方式详解

| 方式   | 原理                                                                 | 适用场景                                                    | 依赖                     |
| ------ | -------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------ |
| `atlas` | 使用预定义的 atlas mesh 与图像进行配准，通过缩放、旋转、平移搜索最优变换 | 适用于大多数 T1/T2 加权图像                                  | atlas mesh 文件          |
| `mni`  | 使用 MNI T1w 模板，通过互相关（cross-correlation）进行仿射配准        | 仅适用于 T1w 图像，需要额外的 initmni 参数调优              | MNI T1w 模板             |
| `trega` | 使用 BrainNet 深度学习网络估计 MNI152 到 RAS 空间的仿射变换          | 支持 CPU/GPU 运行，可获得更准确的初始变换（默认、推荐）  | brainnet 包 + PyTorch    |

**三种方式的差异**:

1. **trega**（默认、推荐）
   - 使用 BrainNet 预训练模型直接预测 MNI152 到 RAS 的仿射变换
   - 基于深度学习方法，精度更高
   - 无需额外配置参数

2. **atlas**
   - 使用预训练的 atlas mesh 进行配准
   - 通过网格变形搜索最优仿射变换参数（缩放、旋转、平移）
   - 参数由 `initatlas` 配置节控制
   - 不依赖额外的深度学习框架

3. **mni**
   - 仅使用 MNI T1w 模板（不是 mesh）
   - 通过互相关最大化进行配准
   - 参数由 `initmni` 配置节控制
   - 适合没有 GPU 的环境，但精度可能略低

### 步骤5: segment.py

- **函数**: `run_segmentation(subject_dir: str, debug: bool = False) -> None`
- **输入**: template_coregistered.nii.gz + 组织概率图
- **输出**:
  - `m2m_{subid}/segmentation/tissue_labeling_upsampled.nii.gz`
  - `m2m_{subid}/segmentation/tissue_labeling_upsampled_LUT.txt`
  - `m2m_{subid}/segmentation/norm_image.nii.gz`
  - `m2m_{subid}/segmentation/segmentation/BiasCorrectedT1.nii.gz`
- **功能**: 分割、偏置场校正、形态学操作、MNI 变形场生成

### 步骤6: create_surfaces.py

- **函数**: `create_surfaces(subject_dir: str, fs_dir: str | None = None) -> None`
- **输入**: tissue_labeling_upsampled.nii.gz + norm_image.nii.gz
- **输出**: `m2m_{subid}/surfaces/{lh,rh}.*` 表面文件
  - lh.white, rh.white
  - lh.pial, rh.pial
  - lh.central, rh.central
  - lh.sphere, rh.sphere
  - lh.sphere.reg, rh.sphere.reg
- **功能**: 皮层表面重建
- **重建方法**:
  - **TopoFit 方法**（默认）: 使用 `brainnet` + `cortech` 库进行皮层表面估计，基于深度学习模型。相关参数（`topofit_contrast`、`topofit_resolution`、`topofit_device`）在 `[surfaces]` 配置节中设置
  - **FreeSurfer 方法**: 从已有 FreeSurfer 目录加载表面
- **可选**: 根据表面更新分割结果（`update_segmentation_from_surfaces`）
  - 开启后，使用重建的皮层表面（white、pial）来修正步骤5的体积分割结果
  - 修正逻辑：根据表面边界重新标记体素标签（GM→CSF、GM→WM 等）
  - 保护机制：通过 `update_segmentation_protect` 配置保护特定结构不被重新标记
  - **`update_segmentation_protect`** 配置项（按以下五类组织）：
    1. `gm_to_csf`: 灰质外 pial 表面转为 CSF（防止灰质被错误标记为脑脊液）
    2. `wm_to_gm`: 白质外 WM 表面转为灰质（防止白质被错误标记为灰质）
    3. `gm_to_wm`: 灰质内 WM 表面转为白质（防止灰质被错误标记为白质）
    4. `csf_to_gm`: CSF 内 pial 表面转为灰质（防止脑脊液被错误标记为灰质）
    5. `csf_to_wm`: CSF 内 WM 表面转为白质（防止脑脊液被错误标记为白质）

### 步骤7: mesh.py

- **函数**: `create_mesh_step(subject_dir: str, debug: bool = False, settings_path: str | None = None, output_dir: str | None = None) -> None`
- **输入**: tissue_labeling_upsampled.nii.gz
- **输出**:
  - `m2m_{subid}/{subid}.msh` (四面体网格)
  - `m2m_{subid}/eeg_positions/*.csv`, `*.geo` (EEG 电极位置)
  - `m2m_{subid}/final_tissues.nii.gz`
  - `m2m_{subid}/final_tissues_LUT.txt`
  - `m2m_{subid}/toMNI/final_tissues_MNI.nii.gz`
- **功能**: 四面体网格生成 + EEG 电极位置变换到受试者空间
- `relabel_internal_air()` - 重新标记内部空气边界
- **扩展参数**:
  - `settings_path`: 允许读取指定的 CHARM ini，用于对比不同 mesh 参数 preset
  - `output_dir`: 允许将 mesh、`eeg_positions`、`final_tissues*` 写入独立目录，避免覆盖原始 `m2m` 数据

## 五、配置文件

各步骤依赖 `charm.ini` 中的设置项：

| 步骤            | 依赖的配置节                                       |
| --------------- | -------------------------------------------------- |
| prepare_t1      | 无                                                 |
| prepare_t2      | 无                                                 |
| denoise         | `preprocess.denoise`                               |
| init_atlas      | `samseg.init_type`, `initmni.*`, `initatlas.*`     |
| segment         | `segment.*`, `samseg.*`                            |
| create_surfaces | `surfaces.*`, `atlas.ini` 中的 `simnibs_tissues` |

> **注**：
> 1. `atlas.ini` 位于 SimNIBS 安装目录下的 atlas 文件夹中：
>    ```
>    {SIMNIBSDIR}/segmentation/atlases/{atlas_name}/{atlas_name}.ini
>    ```
>    具体路径由 `charm.ini` 中的 `samseg.atlas_name` 配置决定，默认值为 `charm_atlas_mni_v1-1`。
>
> 2. `simnibs_tissues` 只负责**组织名称到标签号**的映射（如 `"WM": 1`, `"GM": 2`），实际的**电导率数值**在 `simnibs/utils/mesh_element_properties.py` 中定义。
>
> 3. 仿真时的电导率：
>    - 默认使用 `mesh_element_properties.py` 中的值
>    - 用户可通过修改仿真结构体（如 `SimStruct.cond`）自定义电导率值
| mesh            | `mesh.*`                                           |

### 5.1 配置节详解

#### [general] 通用设置

| 参数     | 类型   | 默认值 | 说明                                      |
| -------- | ------ | ------ | ---------------------------------------- |
| `threads` | int   | 8      | GEMS 代码使用的最大线程数。设为 0 则使用全部可用线程 |

#### [preprocess] 预处理设置

| 参数     | 类型   | 默认值 | 说明                                      |
| -------- | ------ | ------ | ---------------------------------------- |
| `denoise` | bool  | false  | 是否对输入图像进行 SANLM 降噪               |

#### [samseg] samseg 分割设置

| 参数                | 类型   | 默认值           | 说明                                      |
| ------------------ | ------ | ---------------- | ---------------------------------------- |
| `atlas_name`       | str    | "charm_atlas_mni_v1-1" | 使用的 atlas 名称                        |
| `gmm_parameter_file` | str  | ""               | 自定义 GMM 强度参数文件（默认使用 atlas 内置） |
| `init_type`        | str    | "trega"          | 初始化方式：`atlas`、`mni` 或 `trega`      |

#### [initmni] MNI 初始化设置（当 init_type=mni 时使用）

| 参数                  | 类型     | 默认值        | 说明                                      |
| -------------------- | -------- | ------------- | ---------------------------------------- |
| `translation_scale`   | float   | -100          | 优化器的平移尺度（一般不需修改）              |
| `max_iter`           | int     | 300           | 最大迭代次数                              |
| `shrink_factors`     | list    | [2, 1, 0]    | 优化器的缩放因子，定义 为 2^f（0 表示不降采样） |
| `bg_value`           | float   | 0             | 背景填充值                                |
| `smoothing_factors`  | list    | [4.0, 2.0, 0.0] | 优化时的强度平滑因子（mm）                |
| `center_of_mass`     | bool    | true          | 是否使用质心初始化，否则使用几何中心          |
| `samp_factor`        | float   | 1.0           | 度量的随机采样因子                        |
| `num_histogram_bins` | int     | 64            | 直方图 bin 数量                           |

#### [initatlas] Atlas 初始化设置（当 init_type=atlas 时使用）

| 参数                      | 类型   | 默认值                              | 说明                          |
| ------------------------ | ------ | ---------------------------------- | ---------------------------- |
| `affine_scales`           | list   | [[0.85, 0.85, 0.85], ...]        | 仿射变换的缩放因子搜索范围      |
| `affine_rotations`        | list   | [-7, -3.5, 0, 3.5, 7]            | 绕 LR 轴的旋转角度搜索范围（度） |
| `affine_horizontal_shifts` | list   | [-20.0, -10.0, 0, 10.0, 20.0]    | 水平平移搜索范围（mm）         |
| `affine_vertical_shifts`   | list   | [-10.0, 0.0, 10.0]                | 垂直平移搜索范围（mm）         |
| `neck_search_bounds`       | list   | [-0.3, 0.1]                       | 颈部变形搜索边界               |
| `downsampling_factor_affine` | float | 2                                  | 仿射配准的降采样因子（mm）     |

#### [segment] 分割设置

| 参数                       | 类型   | 默认值      | 说明                                      |
| ------------------------- | ------ | ----------- | ---------------------------------------- |
| `downsampling_targets`     | list   | [2.0, 1.0] | 降采样目标分辨率（mm），多分辨率处理        |
| `bias_kernel_width`        | float  | 70          | 偏置场基函数的核大小（mm），越小越灵活      |
| `background_mask_sigma`    | float  | 4.0         | 背景掩码的高斯平滑因子（mm）                |
| `background_mask_threshold` | float  | 0.001       | 背景掩码的阈值                            |
| `mesh_stiffness`           | float  | 0.1         | 分割网格的变形惩罚                         |
| `diagonal_covariances`     | bool   | false       | 多模态数据是否使用对角协方差建模           |
| `csf_factor`               | float  | 0.3         | 使用 T1+T2 时 CSF 的降权因子               |

#### [surfaces] 表面重建设置

| 参数                              | 类型   | 默认值         | 说明                                      |
| -------------------------------- | ------ | ------------- | ---------------------------------------- |
| `topofit_contrast`                | str    | "T1w"         | TopoFit 对比度类型：`T1w`、`synth`       |
| `topofit_resolution`             | str    | "1mm"         | TopoFit 分辨率：`1mm`、`random`           |
| `topofit_device`                  | str    | "cpu"         | 计算设备：`cpu` 或 `cuda`                  |
| `central_surface_fraction`        | float  | 0.5           | 中央灰质表面估计的分数（0-1）               |
| `central_surface_method`         | str    | "equivolume"  | 中央表面估计方法：`equidistance`、`equivolume`、`linear_fit_model` |
| `update_segmentation_from_surfaces` | bool | true          | 是否用皮层表面更新分割结果                  |
| `spherical_registration_process_pool` | int | 2             | 球面配准的工作进程数（1 为顺序，2 为并行）  |
| `update_segmentation_protect` | list | 见下方说明     | 更新分割时保护的结构列表                     |

**`update_segmentation_protect`** 说明：指定更新分割时需要保护（不重新标记）的结构，由五类操作组成：`gm_to_csf`、`wm_to_gm`、`gm_to_wm`、`csf_to_gm`、`csf_to_wm`。详见步骤6说明。

#### [mesh] 网格生成设置

| 参数              | 类型    | 默认值        | 说明                                      |
| ---------------- | ------- | ------------ | ---------------------------------------- |
| `elem_sizes`      | dict    | 见下文       | 基于厚度的单元大小定义                     |
| `smooth_size_field` | float | 2            | 用于平滑尺寸场的三角核大小（可为 0）         |
| `skin_facet_size` | float   | 2.0          | 最外层表面三角形的最大尺寸（mm），false 则禁用 |
| `facet_distances` | dict    | 见下文       | 厚度与边界面到真实标签边界最大距离的关系     |
| `optimize`        | bool    | false         | 是否使用 sliver 扰动、exudation 和 Lloyd 优化（默认不使用，依赖 MMG） |
| `apply_cream`      | bool    | true          | 是否在头部周围应用 cream 层以改善去刺效果    |
| `remove_spikes`    | bool    | true          | 是否移除网格刺以获得更平滑的表面             |
| `skin_tag`         | int     | 1005          | 皮肤表面标签，false 则不输出皮肤表面        |
| `hierarchy`        | list    | false         | 表面标签层级，决定保留哪一侧三角形          |
| `smooth_steps`      | int     | 10            | 最终表面的平滑步数（可为 0）               |
| `skin_care`        | int     | 20            | 皮肤额外平滑步数（可为 0）                 |
| `mmg_noinsert`     | bool    | false         | 是否在网格质量优化步骤中不插入额外点（不推荐） |

**`elem_sizes` 默认值**：
```ini
elem_sizes = {"standard": {"range": [1, 5], "slope": 1.0},
              "1": {"range": [1, 7], "slope": 1.0},
              "2": {"range": [1, 2], "slope": 1.0},
              "5": {"range": [1, 10], "slope": 0.6}}
```
- `range`: 单元大小的上下限（mm）
- `slope`: 越小网格越细
- `standard`: 适用于未定义的其他组织
- 组织标签 1 (GM): range [1, 7]
- 组织标签 2 (WM): range [1, 2]
- 组织标签 5 (CSF): range [1, 10], slope 0.6

**`facet_distances` 默认值**：
```ini
facet_distances = {"standard": {"range": [0.1, 3], "slope": 0.5}}
```
- `range`: 边界面到真实标签边界的距离上下限（mm）
- `slope`: 越小网格越贴合分割边界

## 六、调用方式

```python
from neuracle.charm import (
    prepare_t1,
    prepare_t2,
    denoise_inputs,
    init_atlas,
    run_segmentation,
    create_surfaces,
    create_mesh,
)

# 步骤1: T1准备
prepare_t1(subject_dir, t1_file)

# 步骤2: T2准备
prepare_t2(subject_dir, t2_file, register_t2=True)

# 步骤3: 降噪
denoise_inputs(subject_dir)

# 步骤4: Atlas初始化（支持 atlas/mni/trega 三种初始化方式）
init_atlas(subject_dir)

# 步骤5: 分割
run_segmentation(subject_dir)

# 步骤6: 表面重建（支持 TopoFit 深度学习方法和 FreeSurfer 方法）
create_surfaces(subject_dir)

# 步骤7: 网格生成
create_mesh(subject_dir)
```
