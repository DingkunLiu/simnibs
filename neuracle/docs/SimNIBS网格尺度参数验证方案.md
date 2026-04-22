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

### 5.2 forward

职责：

- 消费已存在的 workspace
- 执行 TI 正向仿真
- 导出 TI 体数据
- 计算 `hotspot_value`
- 计算 `roi_metrics[]`
- 写入 `forward/<case>/result.json`

forward comparison 不在本阶段落盘，而是在 `report` 中临时计算。

### 5.3 inverse

职责：

- 消费已存在的 workspace
- 执行优化
- 导出映射后的 mesh / TI
- 解析 `summary.hdf5`
- 写入 `inverse/<case>/seeds/<seed>/result.json`

inverse 原始结果不再包含 replay/comparison/pass 字段。

### 5.4 replay

职责：

- 消费正式 inverse 结果
- 在 M0 workspace 上回放映射后的电极
- 计算 replay TI 指标
- 与 M0 基线比较
- 写入 `replay/<case>/seeds/<seed>/result.json`

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

## 7. 迁移脚本

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

## 8. 推荐运行顺序

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

## 9. 兼容性结论

- V2 运行器：只认 V2 schema
- V1 历史结果：必须迁移
- `report`：纯读模式
- replay 基线：允许自动等待或串行补齐 M0 replay，但不自动补跑当前 preset 的 inverse
