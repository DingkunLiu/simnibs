# SimNIBS 网格尺度参数验证方案

## 1. 目标

本文档描述当前 `mesh_validation` 实现的实际执行流程，用于比较不同 mesh preset 对 SimNIBS 正向 TI 仿真和逆向优化结果的影响。

当前实现的核心目标是：

- 为每个 `subject/preset` 构建独立工作区，避免直接污染原始 `m2m_<subid>`。
- 支持按阶段执行 `mesh`、`forward`、`inverse`、`report`。
- 允许直接执行 `forward` 或 `inverse`，脚本会在阶段内自动准备所需 workspace。
- 将 mesh、forward、inverse 的结果统一落盘到 `reports/`，便于重复运行和横向比较。

需要特别说明的是，当前实现虽然保留了稳定的 workspace 路径，但 workspace 不是跨阶段持久复用的缓存。每次调用 `prepare_workspace()` 都会重建对应 `workspace/`，阶段内只通过进程内缓存避免同一 `(subject, preset)` 的重复准备。

## 2. 入口与输入

### 2.1 入口脚本

当前入口脚本为：

- `neuracle/mesh_validation/run_mesh_validation.py`

相关实现模块为：

- `neuracle/mesh_validation/mesh_validation_workspace.py`
- `neuracle/mesh_validation/mesh_validation_stages.py`
- `neuracle/mesh_validation/mesh_validation_report.py`

### 2.2 manifest

脚本读取单个 manifest JSON，当前代码实际使用的顶层字段为：

- `work_root`
- `subjects`
- `forward_cases`
- `inverse_cases`

其中：

- `work_root` 可选。若 CLI 未传 `--work-root`，则优先使用 manifest 中的 `work_root`，再退回到 `manifest_path.parent / "work"`。
- `subjects` 中每个条目必须包含 `id`、`m2m_dir`、`reference_t1`。
- `forward_cases` 和 `inverse_cases` 允许为空列表。

示例：

```json
{
  "work_root": "D:/simnibs/mesh_validation/work",
  "subjects": [
    {
      "id": "ernie",
      "m2m_dir": "D:/simnibs/data/m2m_ernie",
      "reference_t1": "D:/simnibs/data/m2m_ernie/T1.nii.gz"
    }
  ],
  "forward_cases": [],
  "inverse_cases": []
}
```

### 2.3 CLI

当前命令行参数为：

- `--manifest`
- `--work-root`
- `--phases`
- `--presets`
- `--subject-ids`
- `--forward-cases`
- `--inverse-cases`
- `--preset-workers`
- `--debug-mesh`
- `--check-only`

参数语义：

- `--phases` 默认执行 `mesh forward inverse report`。
- `--presets` 默认执行全部 `M0-M8`。
- `--preset-workers` 控制 preset 级并行度，默认值为 `1`，底层使用 `spawn` 进程池。
- `--check-only` 当前仅执行 manifest 读取和 preset ini 生成，然后立即退出。

需要注意：

- `--check-only` 不会调用 `ensure_manifest_paths()`，因此不会检查 `m2m_dir` 和 `reference_t1` 是否存在。
- `forward`、`inverse` 阶段不会依赖 `mesh` 阶段的落盘结果，而是各自在阶段内部通过 `ensure_workspace()` 自动重建所需 workspace。

## 3. preset 定义

当前支持的 preset 顺序固定为：

- `M0`
- `M1`
- `M2`
- `M3`
- `M4`
- `M5`
- `M6`
- `M7`
- `M8`

这些 preset 的具体 `elem_sizes`、`skin_facet_size`、`facet_distances` 定义在 `mesh_validation_workspace.py` 的 `PRESET_CONFIGS` 中，并由入口脚本在 `work_root/_settings/` 下生成对应的 `*.ini` 文件。

当前实现中的通过阈值并没有覆盖全部 preset：

- forward 通过阈值仅定义了 `M1-M3`
- inverse 通过阈值仅定义了 `M1-M4`
- `M0` 作为基线，comparison 可用时直接视为通过
- 对于未配置阈值的 preset，相关比较指标仍会计算，但 `forward_pass` 或 `inverse_pass` 会保持为 `null`

## 4. 工作区模型

### 4.1 目录结构

当前目录结构如下：

```text
work_root/
├── _settings/
│   ├── M0.ini
│   ├── M1.ini
│   └── ...
├── logs/
│   └── mesh_validation.log
├── <subject>/
│   └── <preset>/
│       ├── workspace/
│       │   └── m2m_<subid>/
│       │       ├── <subid>.msh
│       │       ├── eeg_positions/
│       │       ├── final_tissues.nii.gz
│       │       ├── final_tissues_LUT.txt
│       │       └── toMNI/
│       │           ├── Conform2MNI_nonl.nii.gz
│       │           ├── MNI2Conform_nonl.nii.gz
│       │           └── final_tissues_MNI.nii.gz
│       ├── forward/
│       │   └── <case>/
│       └── inverse/
│           └── <case>/
│               └── seed_<n>/
└── reports/
```

### 4.2 workspace 准备流程

`prepare_workspace()` 的实际行为如下：

1. 删除并重建 `<subject>/<preset>/workspace/`
2. 在 `workspace/m2m_<subid>/` 下创建当前 subject 的运行目录
3. 从原始 `m2m_dir` 物化只读输入资产，但排除以下内容：
   - `<subid>.msh`
   - `final_tissues.nii.gz`
   - `final_tissues_LUT.txt`
   - `eeg_positions/`
   - `toMNI/`
4. 调用 `create_mesh_step(..., output_dir=workspace_dir)` 生成当前 preset 的 mesh 及相关产物
5. 从原始 `m2m_dir/toMNI/` 物化两个形变场：
   - `Conform2MNI_nonl.nii.gz`
   - `MNI2Conform_nonl.nii.gz`
6. 保留 mesh 阶段生成的 `toMNI/final_tissues_MNI.nii.gz`
7. 调用 `validate_workspace()` 校验完整性

### 4.3 物化规则

当前物化逻辑由 `_materialize_path()` 实现，规则是：

- 优先尝试创建符号链接
- 如果符号链接失败，则降级为复制
- 明确禁止把路径物化到其自身

`toMNI/` 目录的约束如下：

- `toMNI/` 本身不能是符号链接
- `toMNI/` 内若存在自引用符号链接，视为 workspace 准备失败
- 当前只从原始 `m2m_dir/toMNI/` 物化两个形变场，不再整目录合并旧 `toMNI`

## 5. 阶段职责

### 5.1 mesh

`mesh` 阶段由 `build_mesh_variants()` 驱动，职责是：

- 为每个 `(subject, preset)` 准备 workspace
- 读取生成后的 `.msh`
- 统计基础 mesh 指标：
  - `node_count`
  - `tetra_count`
  - `tag_volume_<tag>`

需要注意：

- 头皮表面距离指标不是在 mesh 阶段即时计算的，而是在 `report` 阶段由 `annotate_mesh_rows()` 基于 M0 结果补充
- 若 workspace 准备失败或 mesh 生成失败，会输出 `status=failed` 的结果行

### 5.2 forward

`forward` 阶段由 `run_forward_validation()` 驱动，职责是：

- 为每个 `(subject, preset)` 自动准备 workspace
- 对每个 case 重建 `forward/<case>/`
- 在当前 workspace 上执行两组 TDCS 仿真
- 计算 TI mesh
- 导出 TI NIfTI
- 按 case 中定义的 `roi_specs` 和 `hotspot_roi` 计算 ROI 指标

当前 forward case 支持的 ROI 类型：

- `sphere`
- `mask`

forward 原始输出行会包含：

- `roi_mean`
- `roi_median`
- `roi_p95`
- `roi_p99`
- `roi_max`
- `hotspot_value`
- `ti_mesh_path`
- `ti_volume_path`
- `final_labels_path`

forward 的基线比较并不在本阶段完成，而是在 `report` 阶段由 `annotate_forward_rows()` 统一补充。比较逻辑是：

- 以同一 `(subject_id, case_name, roi_name)` 的 `M0` 为基线
- 计算 ROI 相对误差
- 计算 `hotspot_rel_error`
- 计算 GM 内的 `gm_pearson_r` 和 `gm_nrmse`
- 根据 preset 阈值给出 `forward_pass`

当基线不存在时：

- 保留当前 preset 的原始仿真结果
- `comparison_status = "baseline_unavailable"`
- `forward_pass = null`

### 5.3 inverse

`inverse` 阶段由 `run_inverse_validation()` 驱动，职责是：

- 为每个 `(subject, preset)` 自动准备 workspace
- 对每个 case 和每个 seed 重建 `inverse/<case>/seed_<n>/`
- 初始化 `TesFlexOptimization`
- 根据 case 配置目标函数、ROI、电极布局和优化器
- 执行优化
- 解析 `summary.hdf5`
- 读取 `electrode_mapping.json`
- 导出映射后电极对应的 TI NIfTI

inverse 原始输出行会包含：

- `optimizer`
- `optimizer_fopt`
- `optimizer_n_test`
- `optimizer_n_sim`
- `mapped_mesh_path`
- `mapped_ti_volume_path`
- `electrode_mapping_path`
- `mapped_labels`
- `mapping_distance_mean_mm`

当前 inverse 阶段本身不直接给出和 `M0` 的比较结果。真正的 replay 与比较发生在 `report` 阶段：

1. 在 `report` 阶段串行执行 `run_replay_on_m0()`
2. 将当前 preset 的映射后电极回放到 `M0` workspace
3. 重算 replay TI NIfTI
4. 根据 replay 结果计算 `replay_goal`
5. 与同一 `(subject_id, case_name, seed)` 的 M0 基线比较
6. 计算：
   - `goal_gap_on_m0`
   - `label_consistent`
   - `optimized_center_drift_mean_mm`
   - `optimized_center_drift_max_mm`
   - `mapped_center_drift_mean_mm`
   - `mapped_center_drift_max_mm`
7. 按阈值补充 `inverse_pass`

当基线不存在或 replay 失败时：

- 保留当前 preset 的原始优化结果
- comparison 字段标记为 `baseline_unavailable` 或 `comparison_failed`
- `inverse_pass = null`

### 5.4 report

`report` 阶段由 `aggregate_report()` 驱动，职责是：

- 对 mesh 结果补充与 M0 的头皮表面对比指标
- 对 forward 结果补充与 M0 的 ROI/GM 比较指标
- 对 inverse 结果执行 replay on M0，并补充 comparison 指标
- 输出 `reports/*.json`
- 从 JSON 行结果导出 `reports/*.csv`
- 生成 `reports/summary.json`

当前 `report` 还有一个重要行为：

- 如果某个阶段本次没有执行，`report` 会尝试复用已有的 `reports/<phase>.json`

这意味着分阶段运行时，只要历史 JSON 仍在，`report` 可以在不重跑该阶段的情况下继续汇总。

## 6. 并行与缓存策略

当前并行粒度是 preset 级别：

- `mesh`、`forward`、`inverse` 三个阶段都通过 `collect_parallel_rows()` 调度
- 底层使用 `ProcessPoolExecutor`
- 进程启动方式固定为 `spawn`
- 并行度由 `--preset-workers` 控制

缓存策略是：

- 同一 worker 进程内，对同一 `(subject, preset)` 使用 `workspace_cache` 复用 workspace
- 不同阶段之间不共享这个缓存
- `report` 阶段做 inverse replay 时，也会独立维护自己的 `workspace_cache`

因此当前实现更接近“阶段内避免重复准备 workspace”，而不是“全流程复用同一份 workspace”。

## 7. 校验与失败分类

### 7.1 workspace 校验

`validate_workspace()` 当前会显式检查以下产物：

- `workspace/m2m_<subid>/`
- `<subid>.msh`
- `eeg_positions/`
- `final_tissues.nii.gz`
- `final_tissues_LUT.txt`
- `toMNI/`
- `toMNI/Conform2MNI_nonl.nii.gz`
- `toMNI/MNI2Conform_nonl.nii.gz`
- `toMNI/final_tissues_MNI.nii.gz`

另外还会校验：

- `toMNI/` 目录本身不能是符号链接
- `toMNI/` 下的符号链接不能形成自引用

### 7.2 失败分类

当前代码使用以下 `failure_stage`：

- `workspace_prepare_failed`
- `mesh_generation_failed`
- `forward_execution_failed`
- `inverse_execution_failed`

触发方式：

- `WorkspacePreparationError` 映射为 `workspace_prepare_failed`
- `MeshGenerationError` 映射为 `mesh_generation_failed`
- forward 和 inverse 阶段中的其他异常分别映射为对应执行失败

失败结果行会保留：

- `status = "failed"`
- `failure_stage`
- `error`
- `traceback`

## 8. 报告输出

当前 `reports/` 固定输出如下文件：

- `mesh_stats.csv`
- `mesh_stats.json`
- `forward_metrics.csv`
- `forward_metrics.json`
- `inverse_metrics.csv`
- `inverse_metrics.json`
- `summary.json`

`summary.json` 的汇总规则是：

- `mesh`：统计每个 preset 的 `total` 和 `ok`
- `forward`：统计每个 preset 的 `total`、`ok`、`comparable`、`pass_rate`
- `inverse`：统计每个 preset 的 `total`、`ok`、`comparable`、`pass_rate`

其中：

- `comparable` 表示 comparison 字段可用、能够参与通过率计算的结果数
- 对没有阈值的 preset，虽然可能存在 comparison 指标，但 `pass_rate` 只会基于 `forward_pass` 或 `inverse_pass` 非空的结果计算

## 9. 推荐运行方式

建议在激活 `simnibs_env` 后分阶段执行：

1. 只生成 preset ini，确认配置写出正常
2. 运行 `mesh report`
3. 运行 `forward report`
4. 运行 `inverse report`
5. 必要时再执行全流程

示例：

```powershell
python neuracle/mesh_validation/run_mesh_validation.py `
  --manifest neuracle/mesh_validation/full_manifest_remote.json `
  --check-only
```

```powershell
python neuracle/mesh_validation/run_mesh_validation.py `
  --manifest neuracle/mesh_validation/full_manifest_remote.json `
  --phases mesh report `
  --presets M0 M1 M2 `
  --preset-workers 2
```

```powershell
python neuracle/mesh_validation/run_mesh_validation.py `
  --manifest neuracle/mesh_validation/full_manifest_remote.json `
  --phases forward report `
  --presets M0 M1 M2 M3
```

```powershell
python neuracle/mesh_validation/run_mesh_validation.py `
  --manifest neuracle/mesh_validation/full_manifest_remote.json `
  --phases inverse report `
  --presets M0 M1 M2 M3 M4
```

## 10. 当前边界

当前实现边界如下：

- 这是研究验证脚本，不是稳定公共接口
- 只覆盖现有 `PRESET_CONFIGS`、`forward_cases`、`inverse_cases` 的执行与报告
- `forward_pass` 的阈值只覆盖 `M1-M3`
- `inverse_pass` 的阈值只覆盖 `M1-M4`
- `inverse` 的 replay on M0 在 `report` 阶段串行执行，当前不是并行流程
- `--check-only` 只生成 ini，不做路径存在性校验

如果后续实现改成“跨阶段复用 workspace”或“扩展 preset 阈值覆盖范围”，本文档需要同步更新。
