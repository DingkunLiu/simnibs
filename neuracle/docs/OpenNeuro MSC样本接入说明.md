# OpenNeuro MSC 样本接入说明

## 1. 数据来源

- 数据集：OpenNeuro `ds000224` The Midnight Scan Club (MSC) dataset
- 当前接入 subject：
  - `sub-MSC01`
  - `sub-MSC02`
  - `sub-MSC03`
  - `sub-MSC04`
  - `sub-MSC05`
- 原始下载目录：`D:\simnibs\data\openneuro\ds000224_mesh_validation`

说明：

- `ds000224_mesh_validation` 仍是本地原始 BIDS 下载副本。
- 当前远端接入规范已经切换到 `/data/liudingkun/simnibs/data/openneuro` 下按 subject 展开的 `m2m_sub-*` 目录，不再使用旧的 `*_inputs` 组织。

## 2. 当前远端目录组织

远端 `OpenNeuro` MSC 样本当前按如下目录组织接入：

- `/data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC01`
- `/data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC02`
- `/data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC03`
- `/data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC04`
- `/data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC05`

其中 manifest 当前要求每个 subject 至少满足：

- `m2m_dir = /data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC0x`
- `reference_t1 = /data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC0x/T1.nii.gz`

也就是说，MSC subject 的接入入口现在就是各自的 `m2m_sub-*` 目录本身。

## 3. 原始数据到接入目录的对应关系

原始结构像文件来源仍然是 BIDS 副本中的：

- `sub-MSC0x/ses-struct01/anat/*_run-01_T1w.nii.gz`
- `sub-MSC0x/ses-struct01/anat/*_run-01_T2w.nii.gz`

整理到当前远端接入目录后，目标是：

- `/data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC0x/T1.nii.gz`
- `/data/liudingkun/simnibs/data/openneuro/m2m_sub-MSC0x/T2.nii.gz`

因此，文档中旧的 `ds000224_mesh_validation_inputs\sub-MSC0x` 已经不是当前规范路径。

## 4. Manifest 状态

主 manifest [full_manifest_remote.json](D:\simnibs\neuracle\mesh_validation\full_manifest_remote.json) 已追加这 5 个 MSC subject。

当前 manifest 中存在两类 subject：

- 历史远端样本：`ernie`
- OpenNeuro MSC 样本：`sub-MSC01` 到 `sub-MSC05`

本地运行时，如果只想检查 MSC 样本，建议显式传入 `--subject-ids`，避免命中其他历史远端样本。

示例：

```powershell
conda run -n simnibs_env python -m neuracle.mesh_validation.run_mesh_validation `
  --manifest D:\simnibs\neuracle\mesh_validation\full_manifest_remote.json `
  --work-root D:\simnibs\data\mesh_validation_openneuro_check `
  --subject-ids sub-MSC01 sub-MSC02 sub-MSC03 sub-MSC04 sub-MSC05 `
  --check-only
```

## 5. 当前限制

虽然目录名已经统一为 `m2m_sub-*`，但这只表示接入入口已经和 mesh validation 的远端目录规范对齐，不等于一定已经具备完整 CHARM baseline。

如果某个 subject 目录中当前只有：

- `T1.nii.gz`
- `T2.nii.gz`

那么它仍然属于 **input-only** 状态，尚未满足完整 mesh validation 上游产物要求，例如：

- `segmentation/tissue_labeling_upsampled.nii.gz`
- `segmentation/norm_image.nii.gz`
- `toMNI/Conform2MNI_nonl.nii.gz`
- `toMNI/MNI2Conform_nonl.nii.gz`

因此：

- `--check-only` 可用于 manifest 与路径层面的基础检查
- 真正执行 `mesh` 阶段前，仍需确认每个 MSC subject 对应目录是否已经补齐完整 CHARM baseline

## 6. 后续建议

后续如果继续补齐这批 MSC 样本，建议直接在各自的 `m2m_sub-*` 目录上完成 CHARM 基线产物，而不是再引入一层新的中间输入目录。

按每个 subject 的顺序补齐 CHARM 基线流程：

1. `prepare_t1`
2. `prepare_t2`
3. `init_atlas`
4. `segment`
5. `create_surfaces`
6. `create_mesh_step`

待完整 baseline 生成后，若目录仍保持为当前 `m2m_sub-*` 结构，则 manifest 无需再因目录重命名而调整。
