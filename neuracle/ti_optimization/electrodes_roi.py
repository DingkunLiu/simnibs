"""
TI Optimization 电极和 ROI 配置模块

原理：
    1. 配置两组 ElectrodeArrayPair 电极对
    2. 使用球形或 mask 定义 ROI 区域
    3. focality 目标需要额外配置 Non-ROI 区域

用法：
    from neuracle.ti_optimization import setup_electrodes_and_roi

    setup_electrodes_and_roi(
        opt,
        goal="focality",
        roi_center=[-41.0, -13.0, 66.0],
        roi_radius=20.0,
    )
"""

import logging

from simnibs import opt_struct
from simnibs.utils.mesh_element_properties import ElementTags

logger = logging.getLogger(__name__)


def setup_electrodes_and_roi(
    opt: opt_struct.TesFlexOptimization,
    goal: str,
    mesh_file_path: str | None = None,
    electrode_pair1_center: list[list[float]] | None = None,
    electrode_pair2_center: list[list[float]] | None = None,
    electrode_radius: list[float] | None = None,
    electrode_current1: list[float] | None = None,
    electrode_current2: list[float] | None = None,
    roi_center: list[float] | None = None,
    roi_radius: float | None = None,
    roi_center_space: str = "subject",
    roi_mask_path: str | None = None,
    roi_mask_space: str | None = None,
    non_roi_center: list[float] | None = None,
    non_roi_radius: float | None = None,
) -> None:
    """
    配置电极对和 ROI

    Parameters
    ----------
    opt : opt_struct.TesFlexOptimization
        优化对象
    goal : str
        目标函数类型
    electrode_pair1_center : list[list[float]], optional
        第一组电极阵列中心位置
    electrode_pair2_center : list[list[float]], optional
        第二组电极阵列中心位置
    electrode_radius : list[float], optional
        电极半径
    electrode_current1 : list[float], optional
        第一组电极电流
    electrode_current2 : list[float], optional
        第二组电极电流
    roi_center : list[float], optional
        ROI 球形区域中心
    roi_radius : float, optional
        ROI 球形区域半径
    roi_center_space : str
        ROI 球形区域坐标空间，支持 "subject" 或 "mni"
    roi_mask_path : str, optional
        ROI mask 文件路径。提供后优先使用 mask ROI，而不是球形 ROI
    roi_mask_space : str, optional
        ROI mask 所在坐标空间，支持 "subject" 或 "mni"
    non_roi_center : list[float], optional
        Non-ROI 球形区域中心
    non_roi_radius : float, optional
        Non-ROI 球形区域半径
    """
    # 默认值
    if electrode_pair1_center is None:
        electrode_pair1_center = [[0, 0]]
    if electrode_pair2_center is None:
        electrode_pair2_center = [[0, 0]]
    if electrode_radius is None:
        electrode_radius = [10]
    if electrode_current1 is None:
        electrode_current1 = [0.002, -0.002]
    if electrode_current2 is None:
        electrode_current2 = [0.002, -0.002]
    if roi_center is None:
        roi_center = [-41.0, -13.0, 66.0]
    if roi_radius is None:
        roi_radius = 20.0

    # 配置第一组电极对
    logger.info("配置第一组电极对")
    electrode_layout1 = opt.add_electrode_layout("ElectrodeArrayPair")
    electrode_layout1.center = electrode_pair1_center
    electrode_layout1.radius = electrode_radius
    electrode_layout1.current = electrode_current1

    # 配置第二组电极对
    logger.info("配置第二组电极对")
    electrode_layout2 = opt.add_electrode_layout("ElectrodeArrayPair")
    electrode_layout2.center = electrode_pair2_center
    electrode_layout2.radius = electrode_radius
    electrode_layout2.current = electrode_current2

    # 配置 ROI
    logger.info("配置 ROI")
    roi = opt.add_roi()
    roi.method = "volume"
    roi.mesh = str(mesh_file_path)
    roi.subpath = opt.subpath
    roi.tissues = [ElementTags.WM, ElementTags.GM]  # 只保留白质和灰质
    if roi_mask_path:
        roi.mask_path = roi_mask_path
        roi.mask_space = roi_mask_space or "mni"
        roi.mask_value = 1
        logger.info("使用 atlas ROI mask: %s (space=%s)", roi_mask_path, roi.mask_space)
    else:
        roi.roi_sphere_center_space = roi_center_space
        roi.roi_sphere_center = roi_center
        roi.roi_sphere_radius = roi_radius
        logger.info(
            "使用体积球形 ROI: 中心=%s, 半径=%s, space=%s",
            roi_center,
            roi_radius,
            roi_center_space,
        )

    # focality 目标的第二个 ROI 表示"除目标 ROI 外的其余体积"
    if goal in ["focality", "focality_inv"]:
        non_roi = opt.add_roi()
        non_roi.method = "volume"
        non_roi.mesh = str(mesh_file_path)
        non_roi.subpath = opt.subpath
        non_roi.tissues = [ElementTags.WM, ElementTags.GM]  # 只保留白质和灰质
        if roi_mask_path:
            non_roi.mask_path = roi_mask_path
            non_roi.mask_space = roi_mask_space or "mni"
            non_roi.mask_value = 1
            non_roi.mask_operator = ["difference"]
            logger.info(
                "配置 Non-ROI: 使用 atlas ROI 差集 (mask=%s, space=%s)",
                roi_mask_path,
                non_roi.mask_space,
            )
        else:
            if non_roi_center is None:
                non_roi_center = roi_center
            if non_roi_radius is None:
                non_roi_radius = 25.0
            non_roi.roi_sphere_center_space = roi_center_space
            non_roi.roi_sphere_center = non_roi_center
            non_roi.roi_sphere_radius = non_roi_radius
            non_roi.roi_sphere_operator = ["difference"]
            logger.info(
                "配置 Non-ROI: 中心=%s, 半径=%s, space=%s",
                non_roi_center,
                non_roi_radius,
                roi_center_space,
            )

    logger.info("电极对和 ROI 配置完成")
