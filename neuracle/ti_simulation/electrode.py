"""
Electrode 模块 - 电极对配置

提供 TDCS 电极对配置功能，用于在 SimNIBS 会话中添加电极对阵列。

原理：
    TDCS (经颅直流电刺激) 仿真通过在头皮放置电极对来注入电流。
    每个电极对由阳极（anode）和阴极（cathode）组成，电流从阳极流入，阴极流出。
    在 TI (Temporal Interference) 仿真中，需要配置两对电极：
    - 第一对电极对产生第一个高频电场 (f1)
    - 第二对电极对产生第二个高频电场 (f2)
    两对电极的频率差 (Δf = |f1 - f2|) 决定了干涉产生的低频调制包络频率

用法：
    from neuracle.ti_simulation import setup_electrode_pair1, setup_electrode_pair2

    # 配置第一个电极对 (频率 f1)
    setup_electrode_pair1(
        session=S,
        electrode_pair1=["F5", "P5"],
        current1=[0.001, -0.001]  # 1mA，方向相反
    )

    # 配置第二个电极对 (频率 f2)
    setup_electrode_pair2(
        session=S,
        electrode_pair2=["F6", "P6"],
        current2=[0.001, -0.001]  # 1mA
    )
"""

import logging

from simnibs import sim_struct

logger = logging.getLogger(__name__)


def setup_electrode_pair1(
    session: sim_struct.SESSION,
    electrode_pair1: list[str],
    current1: list[float],
    electrode_shape: str = "ellipse",
    electrode_dimensions: list[float] | None = None,
    electrode_thickness: float = 2.0,
) -> sim_struct.TDCSLIST:
    """
    配置第一个电极对

    用于 TI 仿真的第一组电极对，通常对应频率 f1。

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象，通过 setup_session 创建
    electrode_pair1 : list[str]
        第一个电极对列表 [anode_name, cathode_name]，如 ["F5", "P5"]
        电极名称来自 EEG 电极帽配置或标准 10-20 系统
    current1 : list[float]
        第一组电极对电流列表 [anode_current, cathode_current]，单位 A
        通常设置为 [I, -I]，即阳极电流为正，阴极电流为负（大小相等方向相反）
    electrode_shape : str, optional
        电极形状，支持："ellipse"（椭圆）、"rect"（矩形）、"custom"（自定义）
        (default: "ellipse")
    electrode_dimensions : list[float], optional
        电极尺寸 [width, height]，单位 mm
        对于椭圆形状，为 [长轴, 短轴]
        对于矩形形状，为 [宽度, 高度]
        (default: [40, 40])
    electrode_thickness : float, optional
        电极厚度，单位 mm (default: 2.0)

    Returns
    -------
    sim_struct.TDCSLIST
        配置好的 TDCS 列表对象，包含电极对阵列配置

    See Also
    --------
    setup_electrode_pair2 : 配置第二个电极对
    setup_session : 创建 SimNIBS 会话
    """
    if electrode_dimensions is None:
        electrode_dimensions = [40, 40]

    logger.info("配置第一个电极对: %s", electrode_pair1)
    tdcs1 = _setup_electrode_pair(
        session=session,
        electrode_pair=electrode_pair1,
        currents=current1,
        electrode_shape=electrode_shape,
        electrode_dimensions=electrode_dimensions,
        electrode_thickness=electrode_thickness,
    )
    return tdcs1


def setup_electrode_pair2(
    session: sim_struct.SESSION,
    electrode_pair2: list[str],
    current2: list[float],
    electrode_shape: str = "ellipse",
    electrode_dimensions: list[float] | None = None,
    electrode_thickness: float = 2.0,
) -> sim_struct.TDCSLIST:
    """
    配置第二个电极对

    用于 TI 仿真的第二组电极对，通常对应频率 f2。
    第二组电极的频率应与第一组略有不同，以产生干涉效应。

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象，通过 setup_session 创建
    electrode_pair2 : list[str]
        第二个电极对列表 [anode_name, cathode_name]，如 ["F6", "P6"]
        电极名称来自 EEG 电极帽配置或标准 10-20 系统
    current2 : list[float]
        第二组电极对电流列表 [anode_current, cathode_current]，单位 A
        通常设置为 [I, -I]，即阳极电流为正，阴极电流为负（大小相等方向相反）
    electrode_shape : str, optional
        电极形状，支持："ellipse"（椭圆）、"rect"（矩形）、"custom"（自定义）
        (default: "ellipse")
    electrode_dimensions : list[float], optional
        电极尺寸 [width, height]，单位 mm
        对于椭圆形状，为 [长轴, 短轴]
        对于矩形形状，为 [宽度, 高度]
        (default: [40, 40])
    electrode_thickness : float, optional
        电极厚度，单位 mm (default: 2.0)

    Returns
    -------
    sim_struct.TDCSLIST
        配置好的 TDCS 列表对象，包含电极对阵列配置

    See Also
    --------
    setup_electrode_pair1 : 配置第一个电极对
    setup_session : 创建 SimNIBS 会话
    """
    if electrode_dimensions is None:
        electrode_dimensions = [40, 40]

    logger.info("配置第二个电极对: %s", electrode_pair2)
    tdcs2 = _setup_electrode_pair(
        session=session,
        electrode_pair=electrode_pair2,
        currents=current2,
        electrode_shape=electrode_shape,
        electrode_dimensions=electrode_dimensions,
        electrode_thickness=electrode_thickness,
    )
    return tdcs2


def _setup_electrode_pair(
    session: sim_struct.SESSION,
    electrode_pair: list[str],
    currents: list[float],
    electrode_shape: str,
    electrode_dimensions: list[float],
    electrode_thickness: float,
) -> sim_struct.TDCSLIST:
    """
    配置单个电极对（内部函数）

    Parameters
    ----------
    session : sim_struct.SESSION
        SimNIBS 会话对象
    electrode_pair : list[str]
        电极列表，支持多电极配置 [elec1_name, elec2_name, ...]
    currents : list[float]
        电流配置，长度应与 electrode_pair 一致
        同一通道的多个电极电流会合并
    electrode_shape : str
        电极形状
    electrode_dimensions : list[float]
        电极尺寸 [width, height]
    electrode_thickness : float
        电极厚度

    Returns
    -------
    TDCSLIST
        配置好的 TDCS 列表对象
    """
    tdcs = session.add_tdcslist()

    for i, elec_name in enumerate(electrode_pair):
        electrode = tdcs.add_electrode()
        electrode.channelnr = i + 1
        electrode.centre = elec_name
        electrode.shape = electrode_shape
        electrode.dimensions = electrode_dimensions
        electrode.thickness = electrode_thickness

    tdcs.currents = currents

    return tdcs
