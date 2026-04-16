"""
TI Focality Inverse Optimization Demo - 反向 focality 优化

演示如何使用步骤函数执行 TI Focality Inverse 优化。

数据来源: data/m2m_ernie/

默认配置：
- 电极对1: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- 电极对2: ElectrodeArrayPair, 半径 10mm, 电流 2mA
- ROI 中心: [-41.0, -13.0, 66.0] (subject space)
- ROI 半径: 20mm
- Non-ROI 半径: 25mm
- Focality 阈值: [0.1, 0.2] V/m
"""

from pathlib import Path

from neuracle.logger import setup_logging
from neuracle.ti_optimization import (
    init_optimization,
    run_optimization,
    setup_electrodes_and_roi,
    setup_goal,
)
from neuracle.utils import EEG10_20_EXTENDED_SPM12, find_montage_file
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT
from neuracle.utils.ti_export import export_ti_to_nifti
from simnibs.utils import file_finder


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "ti_focality_inv_optimize"))

    # 设置路径
    subject_dir = DATA_ROOT / "m2m_ernie"
    output_dir = DATA_ROOT / "TI_focality_inv_optimize_ernie"
    sub_files = file_finder.SubjectFiles(subpath=str(subject_dir))
    subid = sub_files.subid

    print("=" * 60)
    print("TI Focality Inverse Optimization: 反向 focality 优化")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")

    # 1. 初始化优化结构
    print("\n[1/6] 初始化优化结构...")
    opt = init_optimization(
        subject_dir=str(subject_dir),
        output_dir=str(output_dir),
        msh_file_path=sub_files.fnamehead,
    )

    # 2. 查找 EEG 电极帽文件
    print("[2/6] 查找 EEG 电极帽文件...")
    net_electrode_file = find_montage_file(subject_dir, EEG10_20_EXTENDED_SPM12)
    print(f"EEG 电极帽文件: {net_electrode_file}")

    # 3. 配置目标函数
    print("[3/6] 配置目标函数 (focality_inv)...")
    setup_goal(
        opt=opt,
        goal="focality_inv",
        focality_threshold=[0.1, 0.2],
        net_electrode_file=net_electrode_file,
    )

    # 4. 配置电极对和 ROI
    print("[4/6] 配置电极对和 ROI...")
    setup_electrodes_and_roi(
        opt=opt,
        goal="focality_inv",
        mesh_file_path=sub_files.fnamehead,
        electrode_pair1_center=[[0, 0]],
        electrode_pair2_center=[[0, 0]],
        electrode_radius=[10],
        electrode_current1=[0.002, -0.002],
        electrode_current2=[0.002, -0.002],
        roi_center=[-41.0, -13.0, 66.0],
        roi_radius=20.0,
        non_roi_center=[-41.0, -13.0, 66.0],
        non_roi_radius=25.0,
    )

    # 5. 运行优化
    print("[5/6] 运行优化算法...")
    output_folder = run_optimization(
        opt=opt,
        n_workers=24,
    )

    # 6. 导出 NIfTI 格式
    print("[6/6] 导出 NIfTI 格式...")
    msh_path = (
        Path(output_folder)
        / "mapped_electrodes_simulation"
        / f"{subid}_tes_mapped_opt_head_mesh.msh"
    )
    ti_nifti_path = export_ti_to_nifti(
        msh_path=str(msh_path),
        output_dir=str(output_dir),
        reference=str(subject_dir / "T1.nii.gz"),
        field_name="max_TI",
        prefix=f"{subid}_optimization",
    )

    print("=" * 60)
    print("TI Focality Inverse 优化完成!")
    print(f"输出目录: {output_folder}")
    print(f"TI NIfTI 文件: {ti_nifti_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
