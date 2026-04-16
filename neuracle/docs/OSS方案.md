# 三类任务接入 OSS 的改造方案

## 0. 当前本地任务实际依赖的文件清单

### 0.1 TI 正向仿真

| 类别 | 文件/目录 | 来源 | 是否必需 | 备注 |
| --- | --- | --- | --- | --- |
| 网格文件 | `dir_path/model.msh` | 根据 `dir_path` 固定拼接 | 是 | 正向仿真核心输入 |
| 电极帽文件 | `dir_path/eeg_positions/<montage>.csv` | `find_montage_file(dir_path, montage)` 本地查找 | 是 | `montage` 仅传名称，例如 `EEG10-10_Cutini_2011` |
| DTI 文件 | `DTI_file_path` 指向的文件 | 参数显式传入 | 是 | 当前文档按最保守口径统一视为必需 |

### 0.2 TI 逆向仿真

| 类别 | 文件/目录 | 来源 | 是否必需 | 备注 |
| --- | --- | --- | --- | --- |
| 网格文件 | `dir_path/model.msh` | 根据 `dir_path` 固定拼接 | 是 | 逆向仿真核心输入 |
| 电极帽文件 | `dir_path/eeg_positions/<montage>.csv` | `find_montage_file(dir_path, montage)` 本地查找 | 是 | `montage` 仅传名称，例如 `EEG10-10_Cutini_2011` |
| DTI 文件 | `DTI_file_path` 指向的文件 | 参数显式传入 | 是 | 当前文档按最保守口径统一视为必需 |
| subject 目录网格 | `dir_path/<subid>.msh` | `SubjectFiles(subpath=dir_path)` | 是 | `dir_path` 需保持 `m2m_*` 目录语义 |
| central surface | `dir_path/surfaces/lh.central.gii` | `SubjectFiles(subpath=dir_path)` | 是 | 当前 ROI 使用 `surface + central` |
| central surface | `dir_path/surfaces/rh.central.gii` | `SubjectFiles(subpath=dir_path)` | 是 | 当前 ROI 使用 `surface + central` |
| MNI 变换文件 | `dir_path/toMNI/MNI2Conform_nonl.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| MNI 变换文件 | `dir_path/toMNI/Conform2MNI_nonl.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| 参考体积 | `dir_path/label_prep/T1_upsampled.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| 参考体积回退文件 | `dir_path/T1.nii.gz` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |
| atlas ROI 文件 | 仓库内标准化 ROI 文件 | `get_standardized_roi_path(...)` | 是 | 当前文档按最保守口径统一视为必需 |
| 约束定位文件 | `dir_path/eeg_positions/Fiducials.csv` | `SubjectFiles(subpath=dir_path)` | 是 | 当前文档按最保守口径统一视为必需 |

### 0.3 当前可排除文件/目录

| 文件/目录 | 是否可排除 | 备注 |
| --- | --- | --- |
| `dir_path/final_tissues.nii.gz` | 是 | 当前 TI 正向/逆向链路未直接读取 |
| `final_tissues_LUT.txt` | 是 | 当前 TI 正向/逆向不作为任务输入 |
| `*.lg.png` | 是 | 当前 TI 正向/逆向不作为计算输入 |

## 1. 总体原则

| 项目 | 处理方式 |
| --- | --- |
| `dir_path` | 作为 `data/` 下的相对目录使用；为空时默认使用 `m2m_ernie` |
| `montage` | 只传名称，从 `dir_path/eeg_positions/` 中查找对应 `.csv` |
| 正向网格文件 | 固定使用 `dir_path/model.msh` |
| 逆向网格文件 | 固定使用 `dir_path/model.msh` |
| 本地缓存命中 | `data/` 下已有 `dir_path` 时直接复用 |
| 本地缓存未命中 | 从 OSS 下载所有以 `dir_path` 开头的文件到 `data/dir_path/` |
| DTI 文件 | 使用 `dir_path/DTI_file_path` |
| 任务完成后输出 | 上传指定结果到 OSS |
| 任务完成后清理 | 删除正向/逆向任务输出目录 |

## 2. 本地目录规则

| 任务 | 本地输入目录 | 本地输出目录 |
| --- | --- | --- |
| 头模生成 | `data/dir_path/` | `data/dir_path/` |
| TI 正向仿真 | `data/dir_path/` | `data/{dir_path}_TI_simulation_{task_id}/` |
| TI 逆向仿真 | `data/dir_path/` | `data/{dir_path}_TI_optimization_{task_id}/` |

## 3. 头模生成方案

### 3.1 输入

| 参数 | 来源 |
| --- | --- |
| `T1_file_path` | OSS |
| `T2_file_path` | OSS |
| `DTI_file_path` | OSS |
| `dir_path` | 请求参数 |

### 3.2 执行流程

| 步骤 | 处理 |
| --- | --- |
| 1 | 根据 `dir_path` 在 `data/` 下创建本地目录 |
| 2 | 从 OSS 下载 `T1/T2/DTI` 到 `data/dir_path/` |
| 3 | 在 `data/dir_path/` 下执行头模生成任务 |

### 3.3 上传范围

| 类型 | OSS 路径 |
| --- | --- |
| 目录 | `{dir_path}/eeg_positions/` |
| 目录 | `{dir_path}/label_prep/` |
| 目录 | `{dir_path}/surfaces/` |
| 目录 | `{dir_path}/toMNI/` |
| 目录 | `{dir_path}/segmentation/` |
| 文件 | `{dir_path}/model.msh` |
| 文件 | `{dir_path}/model.msh.opt` |
| 文件 | `{dir_path}/T2_reg.nii.gz` |

## 4. TI 正向仿真方案

### 4.1 输入

| 参数 | 处理 |
| --- | --- |
| `dir_path` | 先检查 `data/dir_path/` 是否存在 |
| `montage` | 只传名称，从 `data/dir_path/eeg_positions/` 中查找对应 `.csv` |
| `DTI_file_path` | 使用 `data/dir_path/DTI_file_path`；未传则不找 |

### 4.2 数据准备

| 场景 | 处理 |
| --- | --- |
| `data/dir_path/` 已存在 | 直接使用 |
| `data/dir_path/` 不存在 | 创建目录后，从 OSS 下载所有以 `dir_path` 开头的文件到 `data/dir_path/` |

### 4.3 执行参数

| 项目 | 固定取值 |
| --- | --- |
| 网格文件 | `data/dir_path/model.msh` |
| `montage` 文件 | `data/dir_path/eeg_positions/` 下匹配文件 |
| `DTI` 文件 | `data/dir_path/DTI_file_path` |
| `output_dir` | `data/{dir_path}_TI_simulation_{task_id}/` |

### 4.4 上传与返回

| 项目 | 处理 |
| --- | --- |
| 上传文件 | `TI_max_TI.nii.gz` |
| 上传 OSS 路径 | `{dir_path}_TI_simulation_{task_id}/TI_max_TI.nii.gz` |
| 返回字段 | `TI_file` 返回 `{dir_path}_TI_simulation_{task_id}/TI_max_TI.nii.gz` |
| 任务结束清理 | 删除 `data/{dir_path}_TI_simulation_{task_id}/` |

## 5. TI 逆向仿真方案

### 5.1 输入

| 参数 | 处理 |
| --- | --- |
| `dir_path` | 先检查 `data/dir_path/` 是否存在 |
| `montage` | 只传名称，从 `data/dir_path/eeg_positions/` 中查找对应 `.csv` |
| `DTI_file_path` | 使用 `data/dir_path/DTI_file_path`；未传则不找 |

### 5.2 数据准备

| 场景 | 处理 |
| --- | --- |
| `data/dir_path/` 已存在 | 直接使用 |
| `data/dir_path/` 不存在 | 创建目录后，从 OSS 下载所有以 `dir_path` 开头的文件到 `data/dir_path/` |

### 5.3 执行参数

| 项目 | 固定取值 |
| --- | --- |
| 网格文件 | `data/dir_path/model.msh` |
| `montage` 文件 | `data/dir_path/eeg_positions/` 下匹配文件 |
| `DTI` 文件 | `data/dir_path/DTI_file_path` |
| `output_dir` | `data/{dir_path}_TI_optimization_{task_id}/` |

### 5.4 导出规则

| 项目 | 处理 |
| --- | --- |
| 导出来源 | `mapped_electrodes_simulation/model_tes_mapped_opt_head_mesh.msh` |
| 本地输出目录 | `data/{dir_path}_TI_optimization_{task_id}/mapped_electrodes_simulation/` |
| 上传 OSS 路径 | `{dir_path}_TI_optimization_{task_id}/TI_max_TI.nii.gz` |

### 5.5 上传与返回

| 项目 | 处理 |
| --- | --- |
| 上传文件 | `TI_max_TI.nii.gz` |
| 上传 OSS 路径 | `{dir_path}_TI_optimization_{task_id}/TI_max_TI.nii.gz` |
| 返回字段 | `TI_file` 返回 `{dir_path}_TI_optimization_{task_id}/TI_max_TI.nii.gz`，同时返回 `electrode_A` 和 `electrode_B` |
| 任务结束清理 | 删除 `data/{dir_path}_TI_optimization_{task_id}/` |

## 6. 删除规则

| 任务 | 删除内容 |
| --- | --- |
| 头模生成 | 不删 `data/dir_path/` |
| TI 正向仿真 | 删除 `data/{dir_path}_TI_simulation_{task_id}/`（DEBUG=True 时跳过删除） |
| TI 逆向仿真 | 删除 `data/{dir_path}_TI_optimization_{task_id}/`（DEBUG=True 时跳过删除） |

### 6.1 DEBUG 模式

通过 `neuracle/utils/constants.py` 中的 `DEBUG` 常量控制：
- `DEBUG=True` 时，跳过正向/逆向仿真 output_dir 的删除
- 默认为 `False`，任务完成后正常删除 output_dir
