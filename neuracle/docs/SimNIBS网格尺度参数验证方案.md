# SimNIBS 网格尺度参数验证方案（V2）

## 1. 目标

Mesh validation V2 将旧实现中混杂的三类职责拆开：

- 阶段执行：`mesh -> forward -> inverse -> replay`
- 正式产物落盘：每个阶段只写自己的 `result.json` 与 `run.json`
- 报告聚合：`report` 只读正式结果，不再补跑 replay，也不再兼容旧 `reports/*.json`

这次重构是 **breaking change**：

- 新运行器只识别 V2 schema
- 旧 `work_root` 必须先迁移
- `report` 不再承担任何执行职责

## 2. 运行入口

官方运行方式固定为：

```powershell
conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest <manifest.json> `
  --work-root <work_root>
```

阶段集合为：

- `mesh`
- `forward`
- `inverse`
- `replay`
- `report`

`--check-only` 的行为：

- 生成 `_settings/*.ini`
- 初始化 `schema.json`
- 初始化 `manifest.snapshot.json`
- 不执行任何仿真阶段

## 3. V2 目录结构

```text
work_root/
├── schema.json
├── manifest.snapshot.json
├── _settings/
├── logs/
├── subjects/<subject_id>/presets/<preset>/
│   ├── workspace/m2m_<subject_id>/
│   ├── mesh/
│   │   ├── result.json
│   │   └── run.json
│   ├── forward/<case_name>/
│   │   ├── result.json
│   │   ├── run.json
│   │   └── artifacts/
│   ├── inverse/<case_name>/seeds/<seed>/
│   │   ├── result.json
│   │   ├── run.json
│   │   └── artifacts/
│   └── replay/<case_name>/seeds/<seed>/
│       ├── result.json
│       ├── run.json
│       └── artifacts/
└── reports/
```

说明：

- `workspace/` 只承载 mesh 相关主体产物
- `artifacts/` 目录承载 forward / inverse / replay 的阶段运行输出
- `result.json` 是正式结果
- `run.json` 是运行状态

## 4. run.json 状态机

统一字段如下：

- `state`: `running | completed | failed`
- `stage`
- `subject_id`
- `preset`
- `case_name`
- `seed`
- `started_at`
- `updated_at`
- `finished_at`
- `command`
- `error`

约束：

- 任务启动时写入 `running`
- 运行中的任务必须至少每 30 秒刷新一次 `updated_at`
- 任务结束后写入 `completed` 或 `failed`

## 5. 各阶段职责

### 5.1 mesh

职责：

- 构建或复用 `workspace/m2m_<subject_id>/`
- 校验 workspace 完整性
- 统计 mesh 基础指标
- 写入 `mesh/result.json`

`mesh/result.json` 只保留原始结果，不写 M0 比较字段。

`mesh/result.json` 原始字段：

| 字段 | 说明 |
|---|---|
| `node_count` | 网格节点总数 |
| `tetra_count` | 四面体单元总数 |
| `tag_volume_<tag>` | 各组织标签（如 `1`=WM、`2`=GM）的四面体体积之和（mm³） |

`report` 阶段追加（与 M0 头皮表面比较）：

| 字段 | 说明 |
|---|---|
| `scalp_mean_distance_mm` | 当前 preset 与 M0 头皮表面双向最近邻距离的均值（mm） |
| `scalp_hausdorff95_mm` | 双向最近邻距离的 95 百分位数（mm） |

### 5.2 forward

职责：

- 消费已存在的 workspace
- 执行 TI 正向仿真
- 导出 TI 体数据（NIfTI）
- 计算 GM 内的峰值与阈值激活指标
- 写入 `forward/<case>/result.json`

`forward/result.json` 原始字段：

| 字段 | 说明 |
|---|---|
| `gm_peak_value` | GM 体素内 TI 最大值（V/m） |
| `gm_threshold_metrics` | 各绝对阈值下的 GM 激活统计列表（见下） |

`gm_threshold_metrics` 每项（阈值固定为 0.10 / 0.15 / 0.20 / 0.25 / 0.30 V/m，可在 manifest 的 `gm_thresholds_abs` 字段覆盖）：

| 子字段 | 说明 |
|---|---|
| `threshold_abs` | 绝对阈值（V/m） |
| `gm_voxels` | GM 体素总数 |
| `current_gm_threshold_voxels` | TI ≥ 阈值的 GM 体素数 |
| `current_gm_threshold_fraction` | 激活比例 = 激活数 / 总数 |

`report` 阶段追加（与 M0 基线比较，在 shared GM mask 上计算）：

| 字段 | 说明 |
|---|---|
| `gm_pearson_r` | 当前 preset 与 M0 GM 体素 TI 值的 Pearson 相关系数 |
| `gm_nrmse` | 归一化 RMSE = ‖current − baseline‖ / ‖baseline‖ |
| `gm_99_percentile_value` | 当前 preset 自身 GM 的 99 百分位 TI（热点强度） |
| `baseline_gm_99_percentile_value` | M0 的 99 百分位 TI |
| `gm_99_percentile_rel_error` | 热点强度相对误差 = \|current_p99 − baseline_p99\| / \|baseline_p99\| |
| `gm_threshold_consistency_auc_abs` | 在各绝对阈值上构造 ROC，对应 AUC |
| `gm_threshold_mean_dice_abs` | 各阈值 Dice 的均值 |
| `gm_threshold_mean_jaccard_abs` | 各阈值 Jaccard 的均值 |
| `forward_pass` | 是否通过（见§7）；M4–M8 无阈值定义，值为 `null` |

每个阈值还展开为独立列（如 `gm_threshold_dice_abs_010`），包含 tp/fp/fn/tn/tpr/fpr/precision/dice/jaccard。

forward comparison 不在本阶段落盘，而是在 `report` 中临时计算。

### 5.3 inverse

职责：

- 消费已存在的 workspace
- 执行优化
- 导出映射后的 mesh / TI
- 解析 `summary.hdf5`
- 写入 `inverse/<case>/seeds/<seed>/result.json`

`inverse/result.json` 原始字段：

| 字段 | 说明 |
|---|---|
| `optimizer` | 优化器名称（如 `differential_evolution`） |
| `optimizer_fopt` | 收敛时的最优目标函数值 |
| `optimizer_n_test` | 候选解总评估次数 |
| `optimizer_n_sim` | 实际执行的 FEM 仿真次数 |
| `mapped_labels` | 竖线分隔的最终映射 EEG 通道标签 |
| `mapping_distance_mean_mm` | 优化位置→最近 EEG 帽位置的平均映射距离（mm） |

inverse 原始结果不再包含 replay/comparison/pass 字段。

### 5.4 replay

职责：

- 消费正式 inverse 结果
- 在 M0 mesh 上回放当前 preset 映射的电极布局
- 计算 replay TI 指标
- 与 M0 基线比较
- 写入 `replay/<case>/seeds/<seed>/result.json`

`replay/result.json` 字段：

**ROI 目标指标**（所有 goal 类型均输出）：

| 字段 | 说明 |
|---|---|
| `replay_roi_mean` | ROI 内 TI 均值（V/m） |
| `replay_roi_p999` | ROI 内 TI 99.9 百分位（V/m） |
| `replay_goal` | 优化目标值：mean→`−roi_mean`，max→`−roi_p999`，focality→`−100·(√2 − roc)` |
| `replay_non_roi_mean` | 非 ROI 区域 TI 均值（仅 focality） |
| `replay_roc` | SimNIBS ROC 指标（仅 focality） |

**与 M0 基线比较**：

| 字段 | 说明 |
|---|---|
| `goal_gap_on_m0` | `|replay_goal − M0_replay_goal| / |M0_replay_goal|`（目标函数相对误差） |
| `label_consistent` | 当前 preset 与 M0 电极标签是否完全一致 |
| `optimized_center_drift_mean_mm` / `_max_mm` | 优化器连续中心位置与 M0 的均值/最大漂移（mm） |
| `mapped_center_drift_mean_mm` / `_max_mm` | 映射后 EEG 帽位置与 M0 的均值/最大漂移（mm） |
| `inverse_pass` | 是否通过（见§7）；M5–M8 无阈值定义，值为 `null` |

replay comparison 和 `inverse_pass` 全部在本阶段产出。

### 5.5 report

职责：

- 只读 V2 正式结果
- 临时补充 mesh 的 M0 头皮表面对比
- 临时补充 forward 的 M0 ROI/GM comparison
- 导出：
  - `mesh_report.json|csv`
  - `forward_report.json|csv`
  - `inverse_report.json|csv`
  - `replay_report.json|csv`
  - `summary.json`

report 不会执行 replay，也不会复用旧版 `reports/*.json`。

## 6. Replay 基线策略

`replay --presets M8` 不要求显式把 `M0` 写进 `--presets`，但系统内部必须先满足 M0 前序条件。

串行预检查顺序：

1. 检查 `M0 mesh/run.json.state == completed`
2. 校验 M0 workspace 完整
3. 检查对应 `M0 inverse/run.json.state`
4. 检查对应 `M0 replay/run.json.state`
5. 若 `M0 replay` 尚不存在但 `M0 inverse` 已完成，则串行生成 M0 replay 基线
6. 所有需要的 M0 基线就绪后，才进入非 M0 preset 的并行 replay

等待规则：

- 若 M0 mesh / inverse / replay 处于 `running`，当前 replay 会等待
- 轮询间隔固定为 10 秒
- 若 `updated_at` 超过 5 分钟未刷新，视为僵死任务并失败退出

## 7. Manifest 格式

```json
{
  "schema_version": 2,
  "work_root": "/path/to/work_root",
  "subjects": [
    {
      "id": "ernie",
      "m2m_dir": "/path/to/m2m_ernie",
      "reference_t1": "/path/to/m2m_ernie/T1.nii.gz"
    }
  ],
  "forward_cases": [
    {
      "name": "ti_demo",
      "eeg_cap": "EEG10-20_extended_SPM12",
      "pair1": ["F5", "P5"],
      "pair2": ["F6", "P6"],
      "current1": [0.001, -0.001],
      "current2": [0.001, -0.001],
      "n_workers": 16,
      "gm_thresholds_abs": [0.10, 0.15, 0.20, 0.25, 0.30]
    }
  ],
  "inverse_cases": [
    {
      "name": "m1_focality_lh",
      "goal": "focality",
      "net_electrode_file": "EEG10-20_extended_SPM12",
      "roi": { "type": "sphere", "center": [-41.0, -13.0, 66.0], "radius": 20.0 },
      "non_roi": { "type": "sphere", "center": [-41.0, -13.0, 66.0], "radius": 25.0 },
      "threshold": [0.1, 0.2],
      "optimizer": "differential_evolution",
      "seeds": [7],
      "n_workers": 16
    }
  ]
}
```

**ROI 类型：**

| `type` | 必填字段 | 说明 |
|---|---|---|
| `sphere` | `center`（XYZ，mm）、`radius`（mm）、可选 `space`（默认 `subject`） | 球形 ROI |
| `atlas` | `atlas_name`、`areas`（列表）、可选 `space`（默认 `mni`） | 标准图谱 ROI，自动查找预构建 mask |
| `mask` | `path`（NIfTI 路径）、可选 `space`（默认 `subject`）、可选 `value`（默认 `1`） | 自定义 mask |

atlas ROI 在运行时会被展开为 mask ROI，`mni` space 的 mask 会自动 warp 到 subject 空间。

## 8. Preset 参数表

Mesh 精度梯度，M0 为最高精度基线，数字越大元素越粗、计算越快。M4 / M8 在同等 `elem_sizes` 基础上额外收紧 `facet_distances`（表面离散化距离），代表不同曲面采样策略的组合变体。

| Preset | GM range (mm) | Standard range (mm) | skin_facet_size (mm) | facet_distances |
|---|---|---|---|---|
| M0 | [1.0, 2.0] | [1.0, 5.0] | 2.0 | — |
| M1 | [1.0, 2.5] | [1.25, 6.0] | 2.5 | — |
| M2 | [1.2, 3.0] | [1.5, 7.0] | 3.0 | — |
| M3 | [1.5, 3.5] | [2.0, 8.0] | 4.0 | — |
| M4 | [1.2, 3.0] | [1.5, 7.0] | 3.0 | standard [0.2, 4.0] slope 0.7 |
| M5 | [1.5, 3.75] | [2.25, 9.0] | 4.5 | — |
| M6 | [1.5, 4.0] | [2.5, 10.0] | 5.0 | — |
| M7 | [1.5, 4.5] | [3.0, 11.0] | 6.0 | — |
| M8 | [1.5, 4.5] | [3.0, 11.0] | 6.0 | standard [0.3, 5.0] slope 0.9 |

所有 preset 的 `slope` 均随 range 单调递增（见 `PRESET_CONFIGS`）。`facet_distances` 仅 M4 / M8 设置，影响曲面网格的离散化精度。

## 9. 通过/失败阈值

### Forward（`forward_pass`）

仅 M1–M3 有定义；以下三条件须**全部**满足：

| Preset | `gm_99_percentile_rel_error` ≤ | `gm_pearson_r` ≥ | `gm_nrmse` ≤ |
|---|---|---|---|
| M1 | 0.10 | 0.95 | 0.08 |
| M2 | 0.15 | 0.95 | 0.12 |
| M3 | 0.20 | 0.95 | 0.12 |

M4–M8：`forward_pass = null`（无阈值定义）。

### Inverse/Replay（`inverse_pass`）

仅 M1–M4 有定义；单条件：

| Preset | `goal_gap_on_m0` ≤ |
|---|---|
| M1 | 0.05 |
| M2 | 0.08 |
| M3 | 0.12 |
| M4 | 0.12 |

M5–M8：`inverse_pass = null`（无阈值定义）。

## 10. 迁移脚本

旧版 workspace 迁移命令：

```powershell
conda run -n simnibs_env python -m neuracle.mesh_validation.migrate_workspace_schema `
  --old-work-root <old_work_root> `
  --new-work-root <new_work_root>
```

迁移行为：

- 复制旧 `_settings/` 和 `logs/`
- 将旧 `<subject>/<preset>/` 重组为 `subjects/<subject>/presets/<preset>/`
- 将旧 `inverse/.../replay_on_m0` 移到 V2 `replay/.../artifacts/`
- 从旧 `reports/*.json` 生成新的 `result.json`
- 为每个迁移后的阶段结果写入 `run.json`

迁移完成后，新运行器只使用 V2 目录，不再读取旧 `reports/*.json`

## 11. 推荐运行顺序

如需只验证 `M8` 并使用 M0 作为基线，建议按下面顺序：

```powershell
conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest <manifest.json> `
  --work-root <work_root> `
  --phases mesh `
  --presets M0 M8

conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest <manifest.json> `
  --work-root <work_root> `
  --phases forward `
  --presets M8

conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest <manifest.json> `
  --work-root <work_root> `
  --phases inverse `
  --presets M0 M8

conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest <manifest.json> `
  --work-root <work_root> `
  --phases replay `
  --presets M8 `
  --preset-workers 2

conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest <manifest.json> `
  --work-root <work_root> `
  --phases report `
  --presets M8
```

### 11.1 远端 full manifest 当前范围

`neuracle/mesh_validation/full_manifest_remote.json` 当前用于远端全量 forward/report 复跑：

- subjects：`ernie`、`sub-MSC01`、`sub-MSC02`、`sub-MSC03`、`sub-MSC04`、`sub-MSC05`
- forward cases：`ti_demo`、`motor_rampersad2019`、`hippocampus_rampersad2019`、`thalamus_lee2021`
- inverse cases：`m1_focality_lh`、`thalamus_focality_lh`
- inverse seeds：`7`、`19`、`31`

远端执行命令：

```bash
conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation \
  --manifest /home/dell/simnibs/neuracle/mesh_validation/full_manifest_remote.json \
  --phases forward report \
  --presets M0 M1 M2 M3 M4 M5 M6 M7 M8 \
  --subject-ids ernie sub-MSC01 sub-MSC02 sub-MSC03 sub-MSC04 sub-MSC05 \
  --preset-workers 9
```

## 12. 兼容性结论

- V2 运行器：只认 V2 schema
- V1 历史结果：必须迁移
- `report`：纯读模式
- replay 基线：允许自动等待或串行补齐 M0 replay，但不自动补跑当前 preset 的 inverse
