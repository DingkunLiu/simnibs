"""
RabbitMQ 消息参数验证器

在接收消息时进行参数合法性检查，确保满足以下条件：
1. 必填字段存在
2. 字段类型正确
3. 值范围有效

验证函数列表
-----------
- validate_model_params()：验证头模生成参数
- validate_forward_params()：验证正向仿真参数
- validate_inverse_params()：验证逆向仿真参数
- validate_ack_test_params()：验证 ACK 测试参数
- validate_message()：验证消息整体结构

验证失败处理
-----------
当验证失败时，抛出 ValidationError 异常。
在 scheduler.py 的 handle_message() 中捕获并处理：
- 发送错误进度到后端
- 直接 ack 消息（丢弃，不重发）
- 记录错误日志

为什么验证失败要 ack 而不是 nack？
因为验证失败通常意味着消息格式错误，即使重发也会失败。
nack 并 requeue 会导致无限重发循环。
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """验证失败异常，仅用于 raise 和 try...except 捕获"""

    pass


def validate_model_params(params: dict[str, Any]) -> None:
    """
    验证头模生成参数

    验证规则
    -------
    - T1_file_path: 非空字符串
    - dir_path: 非空字符串
    - T2_file_path: 可选
    - DTI_file_path: 可选

    Notes
    -----
        id 在消息顶层验证，params 内不验证 id

    Raises
    ------
    ValidationError
        验证失败时抛出
    """
    t1_path = params.get("T1_file_path")
    if not t1_path or not isinstance(t1_path, str):
        logger.error("[T1_file_path] T1_file_path 必须是非空字符串")
        raise ValidationError("T1_file_path 验证失败")

    dir_path = params.get("dir_path")
    if not dir_path or not isinstance(dir_path, str):
        logger.error("[dir_path] dir_path 必须是字符串")
        raise ValidationError("dir_path 验证失败")

    t2_path = params.get("T2_file_path")
    if t2_path is not None and not isinstance(t2_path, str):
        logger.error("[T2_file_path] T2_file_path 必须是字符串")
        raise ValidationError("T2_file_path 必须是字符串")

    dti_path = params.get("DTI_file_path")
    if dti_path is not None and not isinstance(dti_path, str):
        logger.error("[DTI_file_path] DTI_file_path 必须是字符串")
        raise ValidationError("DTI_file_path 必须是字符串")


def validate_forward_params(params: dict[str, Any]) -> None:
    """
    验证正向仿真参数

    验证规则
    -------
    - dir_path: 非空字符串
    - montage: 非空字符串
    - electrode_A: 非空列表，元素为对象 {name: str, current_mA: number}
    - electrode_B: 非空列表，元素为对象 {name: str, current_mA: number}
    - electrode_A 中 current_mA 总和必须为 0
    - electrode_B 中 current_mA 总和必须为 0
    - conductivity_config: 非空字典，值为浮点数
    - anisotropy: 布尔值
    - DTI_file_path: 可选

    Notes
    -----
        id 在消息顶层验证，params 内不验证 id

    Raises
    ------
    ValidationError
        验证失败时抛出
    """
    dir_path = params.get("dir_path")
    if not dir_path or not isinstance(dir_path, str):
        logger.error("[dir_path] dir_path 必须是字符串")
        raise ValidationError("dir_path 验证失败")

    montage = params.get("montage")
    if not montage or not isinstance(montage, str):
        logger.error("[montage] montage 必须是字符串")
        raise ValidationError("montage 验证失败")

    electrode_A = params.get("electrode_A")
    if not electrode_A or not isinstance(electrode_A, list):
        logger.error("[electrode_A] electrode_A 必须是列表")
        raise ValidationError("electrode_A 验证失败")
    if not all(
        isinstance(e, dict)
        and isinstance(e.get("name"), str)
        and isinstance(e.get("current_mA"), (int, float))
        for e in electrode_A
    ):
        logger.error(
            "[electrode_A] electrode_A 元素必须是 {name: str, current_mA: number} 格式"
        )
        raise ValidationError(
            "electrode_A 元素必须是 {name: str, current_mA: number} 格式"
        )

    electrode_B = params.get("electrode_B")
    if not electrode_B or not isinstance(electrode_B, list):
        logger.error("[electrode_B] electrode_B 必须是列表")
        raise ValidationError("electrode_B 验证失败")
    if not all(
        isinstance(e, dict)
        and isinstance(e.get("name"), str)
        and isinstance(e.get("current_mA"), (int, float))
        for e in electrode_B
    ):
        logger.error(
            "[electrode_B] electrode_B 元素必须是 {name: str, current_mA: number} 格式"
        )
        raise ValidationError(
            "electrode_B 元素必须是 {name: str, current_mA: number} 格式"
        )

    current_A_values = [e["current_mA"] for e in electrode_A]
    if abs(sum(current_A_values)) > 1e-6:
        logger.error(
            "[electrode_A] 电流总和必须为 0，当前为: %s", sum(current_A_values)
        )
        raise ValidationError("electrode_A 电流总和必须为 0")

    current_B_values = [e["current_mA"] for e in electrode_B]
    if abs(sum(current_B_values)) > 1e-6:
        logger.error(
            "[electrode_B] 电流总和必须为 0，当前为: %s", sum(current_B_values)
        )
        raise ValidationError("electrode_B 电流总和必须为 0")

    conductivity_config = params.get("conductivity_config")
    if not conductivity_config or not isinstance(conductivity_config, dict):
        logger.error("[conductivity_config] conductivity_config 必须是字典")
        raise ValidationError("conductivity_config 验证失败")
    if not all(isinstance(v, (int, float)) for v in conductivity_config.values()):
        logger.error("[conductivity_config] conductivity_config 值必须是数字")
        raise ValidationError("conductivity_config 值必须是数字")

    anisotropy = params.get("anisotropy")
    if not isinstance(anisotropy, bool):
        logger.error("[anisotropy] anisotropy 必须是布尔值")
        raise ValidationError("anisotropy 必须是布尔值")

    dti_path = params.get("DTI_file_path")
    if dti_path is not None and not isinstance(dti_path, str):
        logger.error("[DTI_file_path] DTI_file_path 必须是字符串")
        raise ValidationError("DTI_file_path 必须是字符串")


def validate_inverse_params(params: dict[str, Any]) -> None:
    """
    验证逆向仿真参数

    验证规则
    -------
    - dir_path: 非空字符串
    - montage: 非空字符串
    - current_A: 非空列表，元素为浮点数，总和必须为 0
    - current_B: 非空列表，元素为浮点数，总和必须为 0
    - cond: 非空字典，值为浮点数
    - anisotropy: 布尔值
    - roi_type: 字符串，必须为 "atlas" 或 "mni_pos"
    - roi_param: 字典，根据 roi_type 验证对应参数
    - roi_type=atlas 时，atlas_param 必填（name, area）
    - roi_type=mni_pos 时，mni_param 必填（center, radius）
    - target_threshold: 浮点数，>= 0
    - DTI_file_path: 可选

    Notes
    -----
        逆向仿真与正向仿真不同，没有 electrode_A 和 electrode_B 参数

    Raises
    ------
    ValidationError
        验证失败时抛出
    """
    dir_path = params.get("dir_path")
    if not dir_path or not isinstance(dir_path, str):
        logger.error("[dir_path] dir_path 必须是字符串")
        raise ValidationError("dir_path 验证失败")

    montage = params.get("montage")
    if not montage or not isinstance(montage, str):
        logger.error("[montage] montage 必须是字符串")
        raise ValidationError("montage 验证失败")

    current_A = params.get("current_A")
    if not current_A or not isinstance(current_A, list):
        logger.error("[current_A] current_A 必须是列表")
        raise ValidationError("current_A 验证失败")
    if not all(isinstance(c, (int, float)) for c in current_A):
        logger.error("[current_A] current_A 元素必须是数字")
        raise ValidationError("current_A 元素必须是数字")

    current_B = params.get("current_B")
    if not current_B or not isinstance(current_B, list):
        logger.error("[current_B] current_B 必须是列表")
        raise ValidationError("current_B 验证失败")
    if not all(isinstance(c, (int, float)) for c in current_B):
        logger.error("[current_B] current_B 元素必须是数字")
        raise ValidationError("current_B 元素必须是数字")

    if current_A and abs(sum(current_A)) > 1e-6:
        logger.error("[current_A] 电流总和必须为 0，当前为: %s", sum(current_A))
        raise ValidationError("current_A 电流总和必须为 0")

    if current_B and abs(sum(current_B)) > 1e-6:
        logger.error("[current_B] 电流总和必须为 0，当前为: %s", sum(current_B))
        raise ValidationError("current_B 电流总和必须为 0")

    conductivity_config = params.get("conductivity_config")
    if not conductivity_config or not isinstance(conductivity_config, dict):
        logger.error("[conductivity_config] conductivity_config 必须是字典")
        raise ValidationError("conductivity_config 验证失败")
    if not all(isinstance(v, (int, float)) for v in conductivity_config.values()):
        logger.error("[conductivity_config] conductivity_config 值必须是数字")
        raise ValidationError("conductivity_config 值必须是数字")

    anisotropy = params.get("anisotropy")
    if not isinstance(anisotropy, bool):
        logger.error("[anisotropy] anisotropy 必须是布尔值")
        raise ValidationError("anisotropy 必须是布尔值")

    roi_type = params.get("roi_type")
    if not roi_type or roi_type not in ("atlas", "mni_pos"):
        logger.error('[roi_type] roi_type 必须是 "atlas" 或 "mni_pos"')
        raise ValidationError('roi_type 必须是 "atlas" 或 "mni_pos"')

    roi_param = params.get("roi_param", {})
    if not isinstance(roi_param, dict):
        logger.error("[roi_param] roi_param 必须是字典")
        raise ValidationError("roi_param 必须是字典")

    if roi_type == "atlas":
        atlas_param = roi_param.get("atlas_param")
        if not atlas_param or not isinstance(atlas_param, dict):
            logger.error("[roi_param.atlas_param] atlas_param 必须是字典")
            raise ValidationError("atlas_param 必须是字典")
        if not atlas_param.get("name"):
            logger.error("[roi_param.atlas_param.name] name 不能为空")
            raise ValidationError("atlas_param.name 不能为空")
        if not atlas_param.get("area"):
            logger.error("[roi_param.atlas_param.area] area 不能为空")
            raise ValidationError("atlas_param.area 不能为空")

    elif roi_type == "mni_pos":
        mni_param = roi_param.get("mni_param")
        if not mni_param or not isinstance(mni_param, dict):
            logger.error("[roi_param.mni_param] mni_param 必须是字典")
            raise ValidationError("mni_param 必须是字典")
        center = mni_param.get("center")
        if not center or not isinstance(center, list) or len(center) != 3:
            logger.error(
                "[roi_param.mni_param.center] center 必须是 [x, y, z] 格式的列表"
            )
            raise ValidationError("center 必须是 [x, y, z] 格式的列表")
        if not all(isinstance(c, (int, float)) for c in center):
            logger.error("[roi_param.mni_param.center] center 元素必须是数字")
            raise ValidationError("center 元素必须是数字")
        radius = mni_param.get("radius")
        if not isinstance(radius, (int, float)) or radius <= 0:
            logger.error("[roi_param.mni_param.radius] radius 必须是正数")
            raise ValidationError("radius 必须是正数")

    threshold = params.get("target_threshold")
    if not isinstance(threshold, (int, float)):
        logger.error("[target_threshold] target_threshold 必须是数字")
        raise ValidationError("target_threshold 必须是数字")
    if threshold < 0:
        logger.error("[target_threshold] target_threshold 必须 >= 0")
        raise ValidationError("target_threshold 必须 >= 0")

    dti_path = params.get("DTI_file_path")
    if dti_path is not None and not isinstance(dti_path, str):
        logger.error("[DTI_file_path] DTI_file_path 必须是字符串")
        raise ValidationError("DTI_file_path 必须是字符串")


def validate_ack_test_params(params: dict[str, Any]) -> None:
    """
    验证 ack 时机测试参数

    验证规则
    -------
    - sleep_seconds: 数字且 > 0

    Raises
    ------
    ValidationError
        验证失败时抛出
    """
    sleep_seconds = params.get("sleep_seconds", 30.0)
    if not isinstance(sleep_seconds, (int, float)):
        logger.error("[sleep_seconds] sleep_seconds 必须是数字")
        raise ValidationError("sleep_seconds 必须是数字")
    if sleep_seconds <= 0:
        logger.error("[sleep_seconds] sleep_seconds 必须大于 0")
        raise ValidationError("sleep_seconds 必须大于 0")


def validate_message(message: dict[str, Any]) -> None:
    """
    验证消息完整性

    验证规则
    -------
    - message 必须是字典
    - id: 非空字符串
    - type: 非空字符串，必须为 "model", "forward", "inverse" 或 "ack_test"
    - params: 字典，根据 type 验证对应参数

    Parameters
    ----------
    message : dict
        消息字典

    Raises
    ------
    ValidationError
        验证失败时抛出
    """
    if not isinstance(message, dict):
        logger.error("[message] 消息必须是字典")
        raise ValidationError("消息必须是字典")

    msg_id = message.get("id")
    if not msg_id or not isinstance(msg_id, str):
        logger.error("[id] id 必须是字符串")
        raise ValidationError("id 必须是字符串")

    msg_type = message.get("type")
    if not msg_type or msg_type not in ("model", "forward", "inverse", "ack_test"):
        logger.error('[type] type 必须是 "model", "forward", "inverse" 或 "ack_test"')
        raise ValidationError('type 必须是 "model", "forward", "inverse" 或 "ack_test"')

    params = message.get("params")
    if not params or not isinstance(params, dict):
        logger.error("[params] params 必须是字典")
        raise ValidationError("params 必须是字典")

    if msg_type == "model":
        validate_model_params(params)
    elif msg_type == "forward":
        validate_forward_params(params)
    elif msg_type == "inverse":
        validate_inverse_params(params)
    elif msg_type == "ack_test":
        validate_ack_test_params(params)
