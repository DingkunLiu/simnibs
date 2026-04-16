# Atlas ROI 离线预处理方案

## 1. 概述

本文档描述 atlas ROI / MNI ROI 的离线预处理流程，以及如何通过 `neuracle.atlas` 模块在线定位标准化 ROI。

核心模块位于 `neuracle/atlas/`，包含：

- `registry.py` — atlas registry 的构建与持久化
- `loader.py` — registry 解析与路径解析
- `standardized.py` — 标准化 ROI 定位与合并逻辑
- `build_standardized_registry.py` — 生成 `atlas_registry.json`
- `standardize_atlases.py` — 将 atlas 标准化到 SimNIBS MNI 空间
- `generate_standardized_rois.py` — 从标准化 atlas 生成单脑区 ROI
- `validate_standardized_atlases.py` — 校验标准化产物

## 2. 非仓库产物说明

`neuracle/atlas/standardized/`、`neuracle/atlas/validation/` 和 `neuracle/atlas/manifests/atlas_registry.json` 属于**离线生成产物**。

这些文件默认不依赖 GitHub 仓库中的现成内容，使用前需要**先在本地离线运行一遍预处理脚本**。也就是说：

- 新机器首次运行 atlas ROI 相关功能前，必须先生成一次标准化 atlas、ROI mask 和 registry
- 如果只拉代码、不做离线预处理，atlas ROI 会因为找不到标准化 ROI 文件而失败

## 3. 首次使用前必须执行的离线步骤

项目要求先激活 `conda` 环境：

```powershell
& conda shell.powershell hook | Out-String | Invoke-Expression
conda activate simnibs
```

然后在仓库根目录按顺序执行：

```powershell
python -m neuracle.atlas.build_standardized_registry
python -m neuracle.atlas.standardize_atlases
python -m neuracle.atlas.generate_standardized_rois
python -m neuracle.atlas.validate_standardized_atlases
```

这四步分别负责：

1. 生成 `atlas_registry.json`
2. 将 atlas 统一到 SimNIBS MNI 模板空间
3. 生成每个脑区的单独 ROI mask
4. 生成基础 validation 报告和 overlay 图

如果 atlas 原始资源有更新，或者脚本逻辑有更新，需要重新执行上述流程。需要覆盖旧产物时可使用：

```powershell
python -m neuracle.atlas.standardize_atlases --force
python -m neuracle.atlas.generate_standardized_rois --force
```

## 4. 目录约定

当前目录结构按下面的分层工作：

```
neuracle/atlas/
├── __init__.py                     # 公共 API 导出
├── registry.py                     # registry 构建与读写
├── loader.py                       # registry 解析与路径解析
├── standardized.py                 # ROI 定位与合并逻辑
├── build_standardized_registry.py  # registry 生成脚本
├── standardize_atlases.py         # atlas 标准化脚本
├── generate_standardized_rois.py  # ROI 生成脚本
├── validate_standardized_atlases.py # validation 脚本
├── atlas/                          # 原始 atlas 数据
├── manifests/                      # atlas_registry.json
├── standardized/                   # 离线生成的标准化 atlas 与 ROI
├── validation/                     # 离线生成的校验报告与 overlay 图
└── demo/                           # ROI demo 代码
```

其中：

- `atlas/` 是原始输入
- `standardized/` 是在线阶段真正读取的 atlas 产物
- `manifests/atlas_registry.json` 是在线查找入口
- `validation/` 是离线校验输出

## 5. registry 设计

registry 由 `build_standardized_registry.py` 生成，格式与实现保持一致：

- registry 内部保存的是**仓库相对路径**
- 运行时由 `loader.py` 自动解析为绝对路径
- 每个脑区对应一个 `index`、`label_en`、`label_zh` 和 `roi_filename`

这样做的原因：

1. 避免把本机绝对路径写死到 JSON
2. 方便跨机器、跨目录迁移
3. 在线代码仍然可以直接拿到绝对路径，不需要自己拼接

## 6. 各 atlas 的当前实现

### 6.1 BN

输入资源：

- `neuracle/atlas/atlas/BN_Atlas_246_1mm/BN_Atlas_246_1mm.nii.gz`
- `neuracle/atlas/atlas/BN_Atlas_246_1mm/BN_Atlas_246_labels_zh.csv`

当前实现：

- 按离散 atlas 处理
- 使用最近邻重采样
- 校验结果表明，标准化后的 BN atlas 与 SimNIBS 模板 `shape` 与 `affine` 一致

### 6.2 JulichBrainAtlas

输入资源：

- `neuracle/atlas/atlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152/JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz`
- `neuracle/atlas/atlas/JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152/JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv`

当前实现：

- 按离散 atlas 处理
- 使用最近邻重采样
- 生成标准化 atlas 和逐脑区 ROI

### 6.3 DiFuMo

输入资源：

- `neuracle/atlas/atlas/DiFuMo/DiFuMo{64,128,256,512,1024}/2mm/maps.nii.gz`
- `neuracle/atlas/atlas/DiFuMo/DiFuMo{64,128,256,512,1024}/labels_*_dictionary_zh.csv`

当前实现：

- 以原始 4D `maps.nii.gz` 作为算法权威输入
- 先重采样到 SimNIBS MNI 空间
- 再在目标空间执行 `argmax`
- 生成算法使用的 3D 离散 atlas
- 再按标签表切分成单脑区 ROI

额外说明：

- `standardize_atlases.py` 中已将 DiFuMo 处理改成**流式 argmax**
- 不再把全部 4D 重采样结果 `np.stack` 到内存中
- 这是为了解决 `DiFuMo1024` 会把内存打满的问题

## 7. 重名脑区的处理

当前实现有一个重要特性：**如果同名脑区匹配到多个 component，则自动合并成一个缓存 mask**。

实现位置在 `standardized.py` 的 `get_standardized_roi_path()`。

这主要是为了适配 DiFuMo 标签表中真实存在的重复命名情况，例如：

- 不同 `Component` 可能共享相同的 `Difumo_names`

当前行为：

1. 按 `label_en` / `label_zh` / `roi_filename stem` / `index` 查找
2. 如果只匹配一个 ROI，直接返回该 ROI
3. 如果匹配多个 ROI，生成 `merged_rois/*.nii.gz` 缓存文件
4. 在线阶段直接使用这个合并后的 mask

## 8. 在线阶段的 ROI 定位

### 8.1 atlas ROI

使用 `get_standardized_roi_path()` 函数定位标准化 ROI：

```python
from neuracle.atlas import get_standardized_roi_path

# atlas_name 示例: "BN_Atlas_246_1mm", "DiFuMo1024", "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
# area_name 支持: 英文名、中文名、文件名 stem、index
roi_path = get_standardized_roi_path("DiFuMo1024", "Hippocampus anterior LH")
```

### 8.2 MNI 坐标 ROI

MNI 球形 ROI 不在业务层做坐标转换，直接通过 SimNIBS 的 `roi_sphere_center_space = "mni"` 参数交给 SimNIBS 处理。

## 9. 当前 validation 的实际含义

`validate_standardized_atlases.py` 已实现的是**轻量空间一致性校验**，不是完整 QA。

它当前会输出：

- `validation_report.json`
- 每个 atlas 一张中间 `z` 切面的 overlay PNG

`validation_report.json` 当前检查：

1. 标准化 atlas 文件是否存在
2. `shape` 是否与 SimNIBS 模板一致
3. `affine` 是否与 SimNIBS 模板一致
4. `roi_dir` 是否存在

所以当前 validation 能说明：

- atlas 标准化产物已经生成
- atlas 已经落到 SimNIBS 模板空间对应网格
- ROI 输出目录存在

但当前 validation 还**没有**覆盖：

- 逐脑区 ROI 数量和标签表的严格一致性
- 每个标签值是否都被正确保留
- 多切面的 overlay 检查
- 每个 ROI 映射到 subject 后的完整人工验收

## 10. demo 说明

当前存在的 demo：

- `neuracle/atlas/demo/atlas_bn_roi_demo.py` — BN ROI demo
- `neuracle/atlas/demo/atlas_julich_roi_demo.py` — JulichBrainAtlas ROI demo
- `neuracle/atlas/demo/atlas_difumo_roi_demo.py` — DiFuMo ROI demo（覆盖 64/128/256/512/1024）
- `neuracle/atlas/demo/mni_roi_demo.py` — MNI 球形 ROI demo

说明：

1. BN demo 选用 `A8m_L`
2. Julich demo 选用 `Area 3b (PostCG)_lh`
3. DiFuMo demo 已覆盖 `64/128/256/512/1024`
4. 只有 `DiFuMo1024` 明确提供了左侧海马 `Hippocampus anterior LH`
5. `DiFuMo128/256/512` 只能选择最接近海马的 component，标签本身不分左右
6. `DiFuMo64` 没有海马相关标签，demo 中使用替代脑区

demo 使用的 subject 数据为：

- `data/m2m_ernie/T1.nii.gz`

demo 输出目录为：

- `data/roi_demo_outputs/`

## 11. 公共 API

`neuracle.atlas` 模块对外暴露以下 API：

```python
from neuracle.atlas import (
    load_atlas_registry,      # 加载 atlas_registry.json
    get_atlas_spec,           # 获取指定 atlas 的完整规范信息
    iter_atlas_specs,         # 遍历所有 atlas 的规范信息
    get_standardized_roi_path, # 根据 atlas 名称和脑区名称定位标准化 ROI
)
```
