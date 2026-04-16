"""
CHARM 步骤7: 四面体网格生成示例

演示如何使用 create_mesh_step 函数从组织标签图像生成 tetrahedral 头模型网格。

数据来源: data/m2m_ernie/
- segmentation/tissue_labeling_upsampled.nii.gz
"""

from neuracle.charm.mesh import create_mesh_step
from neuracle.logger import setup_logging
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "charm_mesh"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"

    print("=" * 60)
    print("CHARM 步骤7: 四面体网格生成")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")

    # 执行网格生成
    # 这将:
    # 1. 加载上采样的组织标签图像
    # 2. 裁剪图像至感兴趣区域
    # 3. 使用 CGAL 进行四面体网格生成
    # 4. 重新标记内部空气边界
    # 5. 变换 EEG 电极位置到受试者空间
    # 6. 输出最终的 .msh 文件
    # 输出:
    # - {subid}.msh (头模型网格)
    # - eeg_positions/*.csv, *.geo (EEG 电极位置)
    # - mni_transf/final_labels.nii.gz, final_labels_MNI.nii.gz
    create_mesh_step(
        subject_dir=str(subject_dir),
        debug=False,  # 设置为 True 会保留中间结果
    )

    print("=" * 60)
    print("四面体网格生成完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
