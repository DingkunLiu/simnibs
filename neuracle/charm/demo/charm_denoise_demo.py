"""
CHARM 步骤3: 图像降噪示例

演示如何使用 denoise_inputs 函数对输入图像进行 SANLM 降噪。

数据来源: data/m2m_ernie/
- T1fs.nii.gz 或 reference_volume
- T2_reg.nii.gz (如果存在)
"""

from neuracle.charm.denoise import denoise_inputs
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "charm_denoise"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"

    print("=" * 60)
    print("CHARM 步骤3: 图像降噪")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行降噪
    # 这将:
    # 1. 读取 settings.ini 中的降噪设置
    # 2. 如果 T1_denoised 不存在，对 T1 进行 SANLM 降噪
    # 3. 如果 T2_reg 存在且 T2_reg_denoised 不存在，对 T2 进行降噪
    # 输出:
    # - T1fs_denoised.nii.gz
    # - T2_reg_denoised.nii.gz (如果存在 T2)
    denoise_inputs(subject_dir=str(subject_dir))

    print("=" * 60)
    print("图像降噪完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
