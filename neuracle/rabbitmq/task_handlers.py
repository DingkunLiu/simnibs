"""
任务处理工具函数

提供 RabbitMQ 任务处理相关的辅助函数，包括：
- ACK 调度

ACK 机制说明
-----------
RabbitMQ 的消息确认（ACK）机制：
1. 消费者收到消息后，RabbitMQ 会将该消息标记为"已投递"（unacked）
2. 消费者处理完消息后，必须调用 basic_ack() 确认
3. 若不确认，消息会一直处于 unacked 状态
4. 若消费者崩溃，RabbitMQ 会将 unacked 消息重新投递给其他消费者

线程安全说明
---------
由于任务在工作线程中执行，而 basic_ack() 必须在消费线程中调用，
因此使用 connection.add_callback_threadsafe() 将 ack 操作投递回消费线程执行。
"""

import logging
from functools import partial
from typing import Any

logger = logging.getLogger(__name__)


def _ack_in_consumer_thread(
    channel: Any,
    delivery_tag: int,
    task_id: str,
    msg_type: str,
    reason: str,
) -> None:
    """
    在 RabbitMQ 消费线程中实际执行 ack

    Parameters
    ----------
    channel : Any
        RabbitMQ 通道
    delivery_tag : int
        消息 delivery tag
    task_id : str
        任务 ID
    msg_type : str
        任务类型
    reason : str
        ack 原因
    """
    try:
        if channel is None or not getattr(channel, "is_open", False):
            logger.warning(
                "ack_sent 跳过，channel 不可用: task_id=%s type=%s delivery_tag=%s reason=%s",
                task_id,
                msg_type,
                delivery_tag,
                reason,
            )
            return
        channel.basic_ack(delivery_tag=delivery_tag)
        logger.info(
            "ack_sent: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )
    except Exception:
        logger.error(
            "发送 ack 失败: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )


def schedule_ack(
    channel: Any,
    delivery_tag: int,
    task_id: str,
    msg_type: str,
    reason: str,
) -> None:
    """
    将 ack 请求从工作线程投递回消费线程执行

    Parameters
    ----------
    channel : Any
        RabbitMQ 通道
    delivery_tag : int
        消息 delivery tag
    task_id : str
        任务 ID
    msg_type : str
        任务类型
    reason : str
        ack 原因
    """
    connection = getattr(channel, "connection", None)
    if connection is None or not getattr(connection, "is_open", False):
        logger.warning(
            "ack_scheduled 跳过，connection 不可用: task_id=%s type=%s delivery_tag=%s reason=%s",
            task_id,
            msg_type,
            delivery_tag,
            reason,
        )
        return
    logger.info(
        "ack_scheduled: task_id=%s type=%s delivery_tag=%s reason=%s",
        task_id,
        msg_type,
        delivery_tag,
        reason,
    )
    connection.add_callback_threadsafe(
        partial(
            _ack_in_consumer_thread,
            channel,
            delivery_tag,
            task_id,
            msg_type,
            reason,
        )
    )
