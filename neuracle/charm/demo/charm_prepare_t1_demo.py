"""
CHARM 步骤1: T1 图像准备示例

演示如何使用 prepare_t1 函数将原始 T1 加权 MRI 图像转换为 SimNIBS 所需格式。

数据来源: data/m2m_ernie/T1.nii.gz
"""

from neuracle.charm.prepare_t1 import prepare_t1
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "charm_prepare_t1"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"
    t1_input = DATA_ROOT / "m2m_ernie" / "T1.nii.gz"

    print("=" * 60)
    print("CHARM 步骤1: T1 图像准备")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Input T1: {t1_input}")

    # 执行 T1 图像准备
    # 这将:
    # 1. 检查并修正 qform/sform 编码
    # 2. 去除单例维度
    # 3. 转换为 float32
    # 4. 保存到 subject_dir (使用 file_finder 的 reference_volume)
    prepare_t1(
        subject_dir=str(subject_dir),
        t1=str(t1_input),
        force_qform=False,
        force_sform=False,
    )

    print("=" * 60)
    print("T1 图像准备完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
