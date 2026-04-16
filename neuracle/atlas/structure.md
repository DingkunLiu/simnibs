# 项目结构说明

```text
atlas/
├─ atlas/                                             atlas 原始数据主目录
│  ├─ BN_Atlas_246_1mm/                    Brainnetome Atlas 相关文件
│  │  ├─ BN_abbreviation_mapping.csv         BN 缩写、英文全称与中文全称对照表
│  │  ├─ BN_Atlas_246_1mm.nii.gz          BN 246 区 1mm 离散图谱 NIfTI
│  │  ├─ BN_Atlas_246_labels_zh.csv         BN 中文标签表，含统一输出字段
│  │  ├─ BN_Atlas_246_LUT.txt               BN 原始 LUT 文本表，含编号、名称与颜色
│  │  ├─ BNA_subregions.xlsx                BN 原始分区说明表
│  │  └─ MNI152_T1_1mm.nii.gz             BN 配套使用的 1mm MNI 模板
│  ├─ DiFuMo/                                   DiFuMo 图谱、标签表与配套模板
│  │  ├─ DiFuMo_translation_mapping.csv       DiFuMo 统一中英翻译映射表
│  │  ├─ tpl-MNI152NLin6Asym_res-02_T1w.nii.gz  DiFuMo 配套的原始 2mm MNI152NLin6Asym 模板
│  │  ├─ tpl-MNI152NLin6Asym_res-02_T1w_difumo-grid.nii.gz  重采样到 DiFuMo 网格的配套模板
│  │  ├─ DiFuMo64/                          64 成分 DiFuMo 数据
│  │  ├─ DiFuMo128/                         128 成分 DiFuMo 数据
│  │  ├─ DiFuMo256/                         256 成分 DiFuMo 数据
│  │  ├─ DiFuMo512/                         512 成分 DiFuMo 数据
│  │  └─ DiFuMo1024/                        1024 成分 DiFuMo 数据
│  ├─ JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152/  Julich 图谱、标签与模板
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv  Julich 双侧合并后中文标签表
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz  Julich 双侧合并后离散图谱
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.nii.gz  Julich 左半球原始图谱
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_lh_MNI152.xml  Julich 左半球原始标签 XML
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.nii.gz  Julich 右半球原始图谱
│  │  ├─ JulichBrainAtlas_3.1_207areas_MPM_rh_MNI152.xml  Julich 右半球原始标签 XML
│  │  ├─ JulichBrainAtlas_translation_mapping.csv  Julich 中英翻译映射表
│  │  └─ mni_icbm152_t1_tal_nlin_asym_09c.nii  Julich 目录中的附带 MNI 模板
│  └─ 颜色查找表/                              汇总后的颜色查找表目录
├─ demo/                                          示例代码目录
│  ├─ atlas_bn_roi_demo.py                    BN atlas ROI demo
│  ├─ atlas_difumo_roi_demo.py                 DiFuMo atlas ROI demo
│  ├─ atlas_julich_roi_demo.py                JulichBrainAtlas ROI demo
│  ├─ mni_roi_demo.py                         MNI 坐标球形 ROI demo
│  └─ roi_demo_common.py                       demo 公共工具函数
├─ manifests/                                   atlas registry 目录
│  └─ atlas_registry.json                        atlas 注册表（离线脚本生成）
├─ standardized/                               标准化 atlas 产物目录
│  ├─ BN_Atlas_246_1mm/                  BN 标准化产物
│  ├─ JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152/  Julich 标准化产物
│  ├─ DiFuMo64/                              DiFuMo64 标准化产物
│  ├─ DiFuMo128/                             DiFuMo128 标准化产物
│  ├─ DiFuMo256/                             DiFuMo256 标准化产物
│  ├─ DiFuMo512/                             DiFuMo512 标准化产物
│  └─ DiFuMo1024/                            DiFuMo1024 标准化产物
├─ validation/                                 验证报告目录
│  ├─ validation_report.json                   验证报告
│  └─ *.png                                  各 atlas 的叠加验证图
├─ __init__.py                                模块初始化（导出在线使用函数）
├─ loader.py                                  图谱加载器
├─ registry.py                               图谱注册表构建与读写
├─ standardized.py                             标准 ROI 工具
├─ build_standardized_registry.py              离线脚本：生成 registry
├─ standardize_atlases.py                    离线脚本：标准化 atlas
├─ generate_standardized_rois.py              离线脚本：生成 ROI 文件
├─ validate_standardized_atlases.py          离线脚本：验证产物
└─ structure.md                              本文件
```
