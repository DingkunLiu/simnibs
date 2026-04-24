# mesh_validation 补充说明

## run_mesh_validation 运行接口

`run_mesh_validation` 的推荐运行方式是：

```powershell
conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest <manifest.json> `
  --work-root <work_root>
```

命令行接口要点：

- `--manifest`：必填，指定 manifest JSON 路径。
- `--work-root`：可选，用于覆盖 manifest 中的 `work_root`。
- `--phases`：可选，阶段取值为 `mesh`、`forward`、`inverse`、`replay`、`report`，默认全阶段。
- `--presets`：可选，限制执行的 preset 范围。
- `--subject-ids`：可选，限制执行的 subject 范围。
- `--forward-cases` / `--inverse-cases`：可选，限制对应 case 范围。
- `--preset-workers`：可选，控制 preset 并行进程数，最小为 `1`。
- `--check-only`：只初始化 schema、snapshot 与 preset ini，不执行后续阶段。
- `--debug-mesh`：保留 mesh 调试输出。

执行建议：

- 优先显式传入 `--phases`、`--presets`、`--subject-ids`，避免默认全量运行。
- 跑 `forward`/`inverse` 时，先确认并清理该 subject/preset 下会污染本次结果的旧 `forward`、`inverse` 目录
