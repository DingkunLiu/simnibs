"""
常量定义模块

汇总全局常量，包括：
- 路径常量
- 并行计算相关常量
- ROI 处理相关常量
- 标准电导率值
- 组织名称列表
- EEG Montage 文件名常量
- OSS 相关常量
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

# neuracle 包目录
NEURACLE_DIR: Path = PROJECT_ROOT / "neuracle"

# 数据根目录
DATA_ROOT: Path = PROJECT_ROOT / "data"

# 并行计算相关常量
N_WORKERS = 8

# ROI 处理相关常量
NON_ROI_THRESHOLD = 0.1

# 组织名称列表（与 SimNIBS tissue tag 顺序对应）
CONDUCTIVITY_TISSUE_NAMES = [
    "WM",
    "GM",
    "CSF",
    "Bone",
    "Scalp",
    "Eyes",
    "CompactBone",
    "SpongyBone",
    "Blood",
    "Muscle",
]

# 标准电导率值 (S/m)
STANDARD_COND = {
    "WM": 0.126,
    "GM": 0.275,
    "CSF": 1.654,
    "Bone": 0.01,
    "Scalp": 0.465,
    "Eyes": 0.5,
    "CompactBone": 0.008,
    "SpongyBone": 0.025,
    "Blood": 0.6,
    "Muscle": 0.16,
}

# EEG Montage CSV 文件名常量
EEG10_10_CUTINI_2011 = "EEG10-10_Cutini_2011"
EEG10_10_NEUROELECTRICS = "EEG10-10_Neuroelectrics"
EEG10_10_UI_JURAK_2007 = "EEG10-10_UI_Jurak_2007"
EEG10_20_OKAMOTO_2004 = "EEG10-20_Okamoto_2004"
EEG10_20_EXTENDED_SPM12 = "EEG10-20_extended_SPM12"

# OSS 相关常量
DEFAULT_STS_TOKEN_DURATION_SECONDES = 3600
DEFAULT_STS_ROLE_SESSION_NAME = "simnibs_session"

# 内置头模相关常量
BUILT_IN_DIR_PATH = "m2m_ernie"
BUILT_IN_DTI_FILE_PATH = "DTI_coregT1_tensor.nii.gz"

# 调试模式：True 时跳过 output_dir 删除
DEBUG = False
