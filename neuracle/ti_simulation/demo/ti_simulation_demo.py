"""
TI Simulation Demo - Temporal Interference 正向仿真示例

演示如何使用步骤函数执行 TI 正向仿真。

数据来源: data/m2m_ernie/

默认配置：
- 电极对1: F5-P5, 电流 1mA
- 电极对2: F6-P6, 电流 1mA
- 电极形状: 椭圆 40x40 mm
- 电极厚度: 2 mm
- n_workers: 24
"""

from neuracle.logger import setup_logging
from neuracle.ti_simulation import (
    calculate_ti,
    run_tdcs_simulation,
    setup_electrode_pair1,
    setup_electrode_pair2,
    setup_session,
)
from neuracle.utils.constants import DATA_ROOT, PROJECT_ROOT
from neuracle.utils.ti_export import export_ti_to_nifti
from simnibs.utils import file_finder


def main():
    """主函数"""
    # 启用日志
    setup_logging(str(PROJECT_ROOT / "log" / "ti_simulation"))

    # 设置路径
    subject_dir = str(DATA_ROOT / "m2m_ernie")
    output_dir = str(DATA_ROOT / "TI_ernie")
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    subid = sub_files.subid

    print("=" * 60)
    print("TI Simulation: Temporal Interference 正向仿真")
    print("=" * 60)
    print(f"Subject directory: {subject_dir}")
    print(f"Output directory: {output_dir}")

    # 1. 配置会话
    print("\n[1/6] 配置会话参数...")
    S = setup_session(
        subject_dir=subject_dir,
        output_dir=output_dir,
        msh_file_path=sub_files.fnamehead,
    )

    # 2. 配置第一个电极对
    print("[2/6] 配置第一个电极对 (F5-P5)...")
    setup_electrode_pair1(
        session=S,
        electrode_pair1=["F5", "P5"],
        current1=[0.001, -0.001],  # 1mA
    )

    # 3. 配置第二个电极对
    print("[3/6] 配置第二个电极对 (F6-P6)...")
    setup_electrode_pair2(
        session=S,
        electrode_pair2=["F6", "P6"],
        current2=[0.001, -0.001],  # 1mA
    )

    # 4. 运行 TDCS 仿真
    print("[4/6] 运行 TDCS 仿真...")
    mesh1_path, mesh2_path = run_tdcs_simulation(
        session=S,
        subject_dir=subject_dir,
        output_dir=output_dir,
        n_workers=24,
    )

    # 5. 计算 TI 场
    print("[5/6] 计算 TI 场...")
    ti_mesh_path = calculate_ti(
        mesh1_path=mesh1_path,
        mesh2_path=mesh2_path,
        output_dir=output_dir,
    )

    # 6. 导出 NIfTI 格式
    print("[6/6] 导出 NIfTI 格式...")
    ti_nifti_path = export_ti_to_nifti(
        msh_path=ti_mesh_path,
        output_dir=output_dir,
        reference=str(DATA_ROOT / "m2m_ernie" / "T1.nii.gz"),
        field_name="max_TI",
        prefix=f"{subid}_simulation",
    )

    print("=" * 60)
    print("TI 仿真完成!")
    print(f"TI 网格文件: {ti_mesh_path}")
    print(f"TI NIfTI 文件: {ti_nifti_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
