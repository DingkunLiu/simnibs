"""
TI Simulation 模块 - Temporal Interference 正向仿真

原理：
    1. 设置两个电极对，配置电流
    2. 运行两次 TDCS 仿真（每个电极对一次）
    3. 从仿真结果中提取电场数据
    4. 使用 TI_utils 计算 TI 最大调制振幅
    5. 生成可视化输出

用法：
    from neuracle.ti_simulation import (
        setup_session,
        setup_electrode_pair1,
        setup_electrode_pair2,
        run_tdcs_simulation,
        calculate_ti,
    )

    # 1. 配置会话
    S = setup_session(subject_dir, output_dir)

    # 2. 配置电极对
    setup_electrode_pair1(S, ["F5", "P5"], current1=[0.001, -0.001])
    setup_electrode_pair2(S, ["F6", "P6"], current2=[0.001, -0.001])

    # 3. 运行仿真
    mesh1_path, mesh2_path = run_tdcs_simulation(S, subject_dir, output_dir, n_workers=24)

    # 4. 计算 TI
    ti_mesh_path = calculate_ti(mesh1_path, mesh2_path, output_dir)
"""

from neuracle.ti_simulation.electrode import (
    setup_electrode_pair1,
    setup_electrode_pair2,
)
from neuracle.ti_simulation.run import run_tdcs_simulation
from neuracle.ti_simulation.session import setup_session
from neuracle.ti_simulation.ti_calc import calculate_ti

__all__ = [
    "setup_session",
    "setup_electrode_pair1",
    "setup_electrode_pair2",
    "run_tdcs_simulation",
    "calculate_ti",
]
