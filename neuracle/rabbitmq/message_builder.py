"""
RabbitMQ 消息构建辅助函数

提供便捷的消息构建接口，用于构造符合规范的 JSON 消息。

这些函数用于构建发送到 RabbitMQ 队列的消息。
消息格式统一为：
{
    "id": "任务唯一标识符",
    "type": "model | forward | inverse | ack_test",
    "params": { ... }  // 任务参数
}

构建函数列表
-----------
- build_model_message()：构建头模生成任务消息
- build_forward_message()：构建正向仿真任务消息
- build_inverse_message()：构建逆向仿真任务消息
- build_ack_test_message()：构建 ACK 测试任务消息
- build_progress_message()：构建进度更新消息（用于从 simnibs 发回后端）
"""

from typing import Literal


def build_model_message(
    id: str,
    dir_path: str,
    T1_file_path: str,
    T2_file_path: str | None = None,
    DTI_file_path: str | None = None,
) -> dict:
    """
    构建头模生成消息

    Parameters
    ----------
    id : str
        任务唯一标识符
    dir_path : str
        头模生成输出目录路径
    T1_file_path : str
        T1 加权 MRI 图像路径
    T2_file_path : str, optional
        T2 加权 MRI 图像路径
    DTI_file_path : str, optional
        DTI 扩散张量图像路径

    Returns
    -------
    dict
        符合规范的 JSON 消息字典
    """
    message = {
        "id": id,
        "type": "model",
        "params": {
            "dir_path": dir_path,
            "T1_file_path": T1_file_path,
        },
    }
    if T2_file_path:
        message["params"]["T2_file_path"] = T2_file_path
    if DTI_file_path:
        message["params"]["DTI_file_path"] = DTI_file_path
    return message


def build_forward_message(
    id: str,
    dir_path: str,
    T1_file_path: str,
    montage: str,
    electrode_A: list[dict],
    electrode_B: list[dict],
    conductivity_config: dict[str, float],
    anisotropy: bool,
    DTI_file_path: str | None = None,
) -> dict:
    """
    构建正向仿真消息

    Parameters
    ----------
    id : str
        任务唯一标识符
    dir_path : str
        头模所在文件夹路径
    T1_file_path : str
        T1 加权 MRI 图像路径
    montage : str
        电极导联名称
    electrode_A : list[dict]
        电极组 A 列表，每个元素为 {name: str, current_mA: number}
    electrode_B : list[dict]
        电极组 B 列表，每个元素为 {name: str, current_mA: number}
    conductivity_config : dict[str, float]
        组织电导率配置
    anisotropy : bool
        是否启用各向异性电导率
    DTI_file_path : str, optional
        DTI 张量文件路径

    Returns
    -------
    dict
        符合规范的 JSON 消息字典
    """
    message = {
        "id": id,
        "type": "forward",
        "params": {
            "dir_path": dir_path,
            "T1_file_path": T1_file_path,
            "montage": montage,
            "electrode_A": electrode_A,
            "electrode_B": electrode_B,
            "conductivity_config": conductivity_config,
            "anisotropy": anisotropy,
        },
    }
    if DTI_file_path:
        message["params"]["DTI_file_path"] = DTI_file_path
    return message


def build_inverse_message(
    id: str,
    dir_path: str,
    T1_file_path: str,
    montage: str,
    current_A: list[float],
    current_B: list[float],
    roi_type: Literal["atlas", "mni_pos"],
    roi_param: dict,
    target_threshold: float,
    conductivity_config: dict[str, float],
    anisotropy: bool,
    DTI_file_path: str | None = None,
) -> dict:
    """
    构建逆向仿真消息

    Parameters
    ----------
    id : str
        任务唯一标识符
    dir_path : str
        头模所在文件夹路径
    T1_file_path : str
        T1 加权 MRI 图像路径
    montage : str
        电极导联名称
    current_A : list[float]
        电极组 A 电流值列表
    current_B : list[float]
        电极组 B 电流值列表
    roi_type : str
        ROI 类型（"atlas" 或 "mni_pos"）
    roi_param : dict
        ROI 参数配置
    target_threshold : float
        目标电场强度阈值
    conductivity_config : dict[str, float]
        组织电导率配置
    anisotropy : bool
        是否启用各向异性电导率
    DTI_file_path : str, optional
        DTI 张量文件路径

    Returns
    -------
    dict
        符合规范的 JSON 消息字典
    """
    message = {
        "id": id,
        "type": "inverse",
        "params": {
            "dir_path": dir_path,
            "T1_file_path": T1_file_path,
            "montage": montage,
            "current_A": current_A,
            "current_B": current_B,
            "roi_type": roi_type,
            "roi_param": roi_param,
            "target_threshold": target_threshold,
            "conductivity_config": conductivity_config,
            "anisotropy": anisotropy,
        },
    }
    if DTI_file_path:
        message["params"]["DTI_file_path"] = DTI_file_path
    return message


def build_progress_message(
    id: str,
    msg_type: Literal["model", "forward", "inverse", "ack_test"],
    progress_rate: int,
    message: str | None = None,
    result: dict | None = None,
) -> dict:
    """
    构建返回消息

    Parameters
    ----------
    id : str
        对应任务的 ID
    msg_type : str
        任务类型
    progress_rate : int
        进度百分比 (0-100)
    message : str, optional
        错误消息或状态描述
    result : dict, optional
        完成时返回的结果

    Returns
    -------
    dict
        符合规范的 JSON 消息字典
    """
    progress_msg = {
        "id": id,
        "type": msg_type,
        "progress_rate": progress_rate,
        "message": message,
        "result": result,
    }
    return progress_msg


def build_ack_test_message(id: str, sleep_seconds: float = 30.0) -> dict:
    """
    构建 ack 时机验证消息

    Parameters
    ----------
    id : str
        任务唯一标识符
    sleep_seconds : float
        模拟长任务持续时间

    Returns
    -------
    dict
        符合规范的 JSON 消息字典
    """
    return {
        "id": id,
        "type": "ack_test",
        "params": {
            "sleep_seconds": sleep_seconds,
        },
    }
