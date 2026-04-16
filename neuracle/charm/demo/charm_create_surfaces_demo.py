"""
CHARM 步骤6: 皮层表面重建示例

演示如何使用 create_surfaces 函数从分割结果重建皮层表面。

数据来源: data/m2m_ernie/
- segmentation/tissue_labeling_upsampled.nii.gz
- segmentation/norm_image.nii.gz
"""

from neuracle.charm.create_surfaces import create_surfaces
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "create_surfaces"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"

    print("=" * 60)
    print("CHARM 步骤6: 皮层表面重建")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行表面重建
    # 这将:
    # 1. 如果提供 FreeSurfer 目录，从 FreeSurfer 表面加载
    # 2. 否则使用 TopoFit 方法进行皮层表面估计（默认）
    # 3. 可选地根据表面更新分割结果
    # 输出:
    # - surfaces/lh.white
    # - surfaces/rh.white
    # - surfaces/lh.pial
    # - surfaces/rh.pial
    # - surfaces/lh.central
    # - surfaces/rh.central
    # - surfaces/lh.sphere
    # - surfaces/rh.sphere
    # - surfaces/lh.sphere.reg
    # - surfaces/rh.sphere.reg
    create_surfaces(
        subject_dir=str(subject_dir),
        fs_dir=None,  # 可选：FreeSurfer subjects 目录
    )

    print("=" * 60)
    print("皮层表面重建完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
