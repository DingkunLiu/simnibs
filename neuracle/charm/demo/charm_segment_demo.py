"""
CHARM 步骤5: 体积与表面分割示例

演示如何使用 run_segmentation 函数执行分割、偏置场校正，并生成上采样的组织标记图像。

数据来源: data/m2m_ernie/
- segmentation/template_coregistered.nii.gz
- T1fs.nii.gz / T2_reg.nii.gz (偏置校正输出)
"""

from neuracle.charm.segment import run_segmentation
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "charm_segment"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"

    print("=" * 60)
    print("CHARM 步骤5: 体积与表面分割")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行分割
    # 这将:
    # 1. 使用 samseg 进行参数估计和分割
    # 2. 输出偏置场校正后的图像
    # 3. 进行后处理（形态学操作）
    # 4. 生成上采样（1mm）的组织标签图像
    # 输出:
    # - segmentation/tissue_labeling_upsampled.nii.gz
    # - segmentation/tissue_labeling_upsampled_LUT.txt
    # - segmentation/norm_image.nii.gz
    # - segmentation/segmentation/BiasCorrectedT1.nii.gz
    run_segmentation(
        subject_dir=str(subject_dir),
        debug=False,  # 设置为 True 会保留中间结果
    )

    print("=" * 60)
    print("分割完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
