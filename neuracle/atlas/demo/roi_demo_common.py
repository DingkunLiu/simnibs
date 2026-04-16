"""
ROI demo 公共工具。

用于验证：
1. atlas ROI 经标准化后的单脑区 mask 是否可被 SimNIBS 正确读取
2. 直接输入 MNI 坐标的球形 ROI 是否可被 SimNIBS 正确映射到 subject
"""

from __future__ import annotations

from pathlib import Path

from neuracle.atlas import get_standardized_roi_path, load_atlas_registry
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT
from simnibs import RegionOfInterest

DEMO_OUTPUT_DIR = DATA_ROOT / "roi_demo_outputs"
SUBJECT_DIR = DATA_ROOT / "m2m_ernie"
T1_PATH = SUBJECT_DIR / "T1.nii.gz"


def setup_demo_environment(log_dir: str) -> None:
    """设置 demo 运行环境。

    Parameters
    ----------
    log_dir : str
        日志子目录名称，会被拼接到 PROJECT_ROOT / "log" / log_dir
    """
    setup_logging(str(PROJECT_ROOT / "log" / log_dir))
    DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not T1_PATH.exists():
        raise FileNotFoundError(f"demo T1 文件不存在: {T1_PATH}")


def _build_surface_roi() -> RegionOfInterest:
    # demo 统一使用 central surface，与 TI 优化中默认关注的皮层表面保持一致。
    roi = RegionOfInterest()
    roi.subpath = str(SUBJECT_DIR)
    roi.method = "surface"
    roi.surface_type = "central"
    return roi


def build_atlas_roi(atlas_name: str, area_name: str) -> tuple[RegionOfInterest, Path]:
    # 如果同名 component 有多个，这里会自动得到合并后的缓存 mask。
    load_atlas_registry()
    roi_path = get_standardized_roi_path(atlas_name, area_name)
    if not roi_path.exists():
        raise FileNotFoundError(
            f"标准化 ROI 不存在: {roi_path}。请先运行 atlas 标准化与 ROI 生成脚本。"
        )

    roi = _build_surface_roi()
    # atlas ROI 保留 MNI mask 语义，由 SimNIBS 负责完成 MNI -> subject 的最后一步。
    roi.mask_path = str(roi_path)
    roi.mask_space = "mni"
    roi.mask_value = 1
    return roi, roi_path


def build_mni_sphere_roi(center: list[float], radius: float) -> RegionOfInterest:
    roi = _build_surface_roi()
    # MNI 球形 ROI 不在业务层先做坐标转换，直接交给 SimNIBS 处理空间语义。
    roi.roi_sphere_center_space = "mni"
    roi.roi_sphere_center = center
    roi.roi_sphere_radius = radius
    return roi


def write_roi_demo(roi: RegionOfInterest, output_name: str) -> Path:
    output_dir = DEMO_OUTPUT_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    # 这里会输出 .msh / .geo / .msh.opt，便于在 Gmsh 中直接查看 ROI 可视化结果。
    roi.write_visualization(str(output_dir), output_name)
    return output_dir


def summarize_roi(roi: RegionOfInterest) -> int:
    # get_nodes() 返回的是映射到 surface 后的节点集合，不是体素数。
    return len(roi.get_nodes())
