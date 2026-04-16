"""
RabbitMQ 消息模块

提供 RabbitMQ 消息中枢、任务调度、进度汇报等功能。

模块架构
-------
scheduler.py      - 任务调度器（核心），处理 model/forward/inverse/ack_test 任务
connection.py    - 连接管理，建立和管理 RabbitMQ 连接
consumer.py      - 消息消费者，监听队列并触发回调
sender.py        - 消息发送器，发送消息到指定队列
publisher.py     - 发送线程，从队列取消息并调用 sender 发送
message_builder.py - 消息构建器，构造各类型消息的 dict
params.py        - 参数转换，将 dict 转为 dataclass
validator.py     - 参数验证，验证必填字段、类型、值范围
progress.py      - 进度管理，定义进度枚举和进度文件读写
schemas.py        - 数据结构，定义参数和结果 dataclass
task_handlers.py - 任务处理工具，如 schedule_ack()

对外接口
-------
使用 from neuracle.rabbitmq import * 可导入：
- RabbitMQConsumer, RabbitMQPublisher, RabbitMQSender, RabbitMQConnection
- 各种 Params dataclass（ModelParams, ForwardParams, InverseParams, AckTestParams）
- 各种 Result dataclass（ModelResult, ForwardResult, InverseResult）
- 验证函数（validate_model_params, validate_forward_params, ...）
- 消息构建函数（build_model_message, build_forward_message, ...）
- 进度枚举（ModelProgress, ForwardProgress, InverseProgress）
"""

from neuracle.rabbitmq.connection import RabbitMQConnection
from neuracle.rabbitmq.consumer import RabbitMQConsumer
from neuracle.rabbitmq.message_builder import (
    build_ack_test_message,
    build_forward_message,
    build_inverse_message,
    build_model_message,
    build_progress_message,
)
from neuracle.rabbitmq.params import (
    dict_to_ack_test_params,
    dict_to_forward_params,
    dict_to_inverse_params,
    dict_to_model_params,
)
from neuracle.rabbitmq.progress import (
    ForwardProgress,
    InverseProgress,
    ModelProgress,
    load_progress,
    save_progress,
)
from neuracle.rabbitmq.publisher import RabbitMQPublisher
from neuracle.rabbitmq.schemas import (
    AckTestParams,
    AtlasParam,
    ElectrodeWithCurrent,
    ForwardParams,
    ForwardResult,
    InverseParams,
    InverseResult,
    MNIParam,
    ModelParams,
    ModelResult,
    ReturnMessage,
    ROIParam,
)
from neuracle.rabbitmq.sender import RabbitMQSender
from neuracle.rabbitmq.validator import (
    ValidationError,
    validate_ack_test_params,
    validate_forward_params,
    validate_inverse_params,
    validate_message,
    validate_model_params,
)

__all__ = [
    "RabbitMQConnection",
    "RabbitMQConsumer",
    "RabbitMQPublisher",
    "RabbitMQSender",
    "AtlasParam",
    "AckTestParams",
    "ElectrodeWithCurrent",
    "MNIParam",
    "ROIParam",
    "ModelParams",
    "ForwardParams",
    "InverseParams",
    "ModelResult",
    "ForwardResult",
    "InverseResult",
    "ReturnMessage",
    "ValidationError",
    "validate_model_params",
    "validate_forward_params",
    "validate_inverse_params",
    "validate_ack_test_params",
    "validate_message",
    "build_model_message",
    "build_forward_message",
    "build_inverse_message",
    "build_ack_test_message",
    "build_progress_message",
    "ModelProgress",
    "ForwardProgress",
    "InverseProgress",
    "load_progress",
    "save_progress",
    "dict_to_model_params",
    "dict_to_forward_params",
    "dict_to_inverse_params",
    "dict_to_ack_test_params",
]
