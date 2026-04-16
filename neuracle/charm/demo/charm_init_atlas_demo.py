"""
CHARM 步骤4: Atlas 初始仿射配准与颈部校正示例

演示如何使用 init_atlas 函数将 MNI Atlas 仿射配准到输入图像空间。

数据来源: data/m2m_ernie/
- T1fs.nii.gz 或 T1fs_denoised.nii.gz
- T2_reg.nii.gz (可选)
"""

from neuracle.charm.init_atlas import init_atlas
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "charm_init_atlas"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"

    print("=" * 60)
    print("CHARM 步骤4: Atlas 初始仿射配准与颈部校正")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行 Atlas 配准
    # 这将:
    # 1. 设置 samseg 相关的 atlas 路径和参数
    # 2. 根据 init_type 设置决定初始化方式（atlas、mni 或 trega）
    # 3. 执行仿射配准
    # 4. 可选进行颈部校正
    # 输出:
    # - segmentation/template_coregistered.nii.gz
    init_atlas(
        subject_dir=str(subject_dir),
        use_transform=None,  # 可选：使用预先计算的变换矩阵
        init_transform=None,  # 可选：初始变换矩阵
        noneck=False,  # 设置为 True 跳过颈部校正
    )

    print("=" * 60)
    print("Atlas 仿射配准完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
