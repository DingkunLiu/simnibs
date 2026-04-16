"""
SimNIBS 端 RabbitMQ 测试示例

持续运行，接收来自 Backend 的消息，验证参数，发送进度和结果消息。

启动方式
--------
.. code-block:: bash

    python -m neuracle.rabbitmq.demo.simnibs_side_demo
"""

import json
import logging
import re
import threading
import time
from functools import partial
from queue import Queue
from typing import Any

from neuracle.config import get_rabbitmq_config, load_env
from neuracle.logger import setup_logging
from neuracle.rabbitmq import RabbitMQConsumer, RabbitMQPublisher
from neuracle.rabbitmq.message_builder import build_progress_message
from neuracle.rabbitmq.validator import ValidationError, validate_message
from neuracle.utils import PROJECT_ROOT

logger = logging.getLogger(__name__)


def ack_in_consumer_thread(
    channel: Any, delivery_tag: int, msg_id: str, msg_type: str, reason: str
) -> None:
    """在消费线程中发送 ack，和 main.py 的线程模型保持一致。"""
    channel.basic_ack(delivery_tag=delivery_tag)
    logger.info(
        "ack_sent: id=%s type=%s delivery_tag=%s reason=%s",
        msg_id,
        msg_type,
        delivery_tag,
        reason,
    )


def schedule_ack(
    channel: Any, delivery_tag: int, msg_id: str, msg_type: str, reason: str
) -> None:
    """把 ack 请求从工作线程投递回消费线程。"""
    logger.info(
        "ack_scheduled: id=%s type=%s delivery_tag=%s reason=%s",
        msg_id,
        msg_type,
        delivery_tag,
        reason,
    )
    channel.connection.add_callback_threadsafe(
        partial(
            ack_in_consumer_thread,
            channel,
            delivery_tag,
            msg_id,
            msg_type,
            reason,
        )
    )


def process_message_async(
    message_queue: Queue,
    channel: Any,
    delivery_tag: int,
    msg_id: str,
    msg_type: str,
    params: dict[str, Any],
) -> None:
    """在线程中模拟任务处理，结束后再由消费线程 ack。"""
    reason = "task_finished"
    try:
        logger.info("worker_started: id=%s type=%s", msg_id, msg_type)
        time.sleep(1.0)

        progress_msg = build_progress_message(
            id=msg_id,
            msg_type=msg_type,
            progress_rate=50,
            message="Processing...",
            result=None,
        )
        message_queue.put(progress_msg)
        logger.info("已发送进度消息: progress_rate=50")

        dir_path = params.get("dir_path", "")
        subid = re.search("m2m_(.+)", dir_path).group(1) if dir_path else msg_id

        if msg_type == "model":
            result = {"msh_file_path": f"/data/{msg_id}/{subid}.msh"}
        elif msg_type == "forward":
            result = {
                "TI_file": f"/data/{msg_id}/{subid}_simulation_max_TI.nii.gz",
            }
        else:
            result = {
                "TI_file": f"/data/{msg_id}/{subid}_optimization_max_TI.nii.gz",
                "electrode_A": ["F3", "FC5"],
                "electrode_B": ["P3", "PO7"],
            }

        complete_msg = build_progress_message(
            id=msg_id,
            msg_type=msg_type,
            progress_rate=100,
            message=None,
            result=result,
        )
        message_queue.put(complete_msg)
        logger.info("已发送完成消息: progress_rate=100")
        logger.info("worker_finished: id=%s type=%s", msg_id, msg_type)
    except Exception as e:
        reason = "task_exception"
        logger.error("异步处理消息时发生错误: %s", e)
        error_msg = build_progress_message(
            id=msg_id,
            msg_type=msg_type,
            progress_rate=0,
            message=str(e),
            result=None,
        )
        message_queue.put(error_msg)
    finally:
        schedule_ack(channel, delivery_tag, msg_id, msg_type, reason)


def handle_message(message_queue: Queue, channel, method, properties, body):
    """
    处理接收到的消息，验证并异步发送响应。

    Parameters
    ----------
    message_queue : Queue
        消息队列，用于发送响应消息
    channel : pika.channel.Channel
        RabbitMQ 通道对象
    method : pika.spec.Basic.Deliver
        消息传递信息
    properties : pika.spec.BasicProperties
        消息属性
    body : bytes
        消息内容
    """
    try:
        message = json.loads(body.decode("utf-8"))
        msg_id = message.get("id")
        msg_type = message.get("type")
        params = message.get("params", {})

        logger.info("=" * 60)
        logger.info("收到消息:")
        logger.info("  ID: %s", msg_id)
        logger.info("  Type: %s", msg_type)
        logger.info(
            "  Params: %s",
            json.dumps(params, indent=4, ensure_ascii=False),
        )
        logger.info("=" * 60)

        try:
            validate_message(message)
            logger.info("消息验证通过")

            worker = threading.Thread(
                target=process_message_async,
                args=(
                    message_queue,
                    channel,
                    method.delivery_tag,
                    msg_id,
                    msg_type,
                    params,
                ),
                daemon=True,
                name=f"simnibs-side-{msg_type}-{msg_id}",
            )
            worker.start()
            logger.info("任务已提交到工作线程: %s", worker.name)

        except ValidationError as e:
            logger.warning("消息验证失败: %s", e)

            error_msg = build_progress_message(
                id=msg_id,
                msg_type=msg_type,
                progress_rate=0,
                message=str(e),
                result=None,
            )
            message_queue.put(error_msg)
            logger.info("已发送错误消息")
            ack_in_consumer_thread(
                channel,
                method.delivery_tag,
                msg_id,
                msg_type,
                "sync_validation_error",
            )

    except json.JSONDecodeError as e:
        logger.error("JSON 解析失败: %s", e)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error("处理消息时发生错误: %s", e)
        channel.basic_ack(delivery_tag=method.delivery_tag)


def run_simnibs_side(config):
    """
    启动 SimNIBS 端，持续监听消息

    Parameters
    ----------
    config : dict
        RabbitMQ 配置字典
    """
    logger.info("启动 SimNIBS 端...")
    logger.info("监听队列: %s", config["send_queue_name"])
    logger.info("发送队列: %s", config["listen_queue_name"])

    message_queue = Queue()

    sender_thread = RabbitMQPublisher(
        host=config["host"],
        port=config["port"],
        queue_name=config["listen_queue_name"],
        message_queue=message_queue,
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )
    sender_thread.start()

    listener = RabbitMQConsumer(
        host=config["host"],
        port=config["port"],
        queue_name=config["send_queue_name"],
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )

    logger.info("SimNIBS 端已启动，持续监听消息（Ctrl+C 退出）...")

    try:
        listener.consume_forever(partial(handle_message, message_queue))
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    finally:
        sender_thread.stop()
        sender_thread.join(timeout=5)
        listener.close()
        logger.info("SimNIBS 端已停止")


if __name__ == "__main__":
    load_env()
    setup_logging(str(PROJECT_ROOT / "log" / "simnibs_side_demo"))
    config = get_rabbitmq_config()
    run_simnibs_side(config)
