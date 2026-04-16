"""
RabbitMQ 监听器使用示例

演示如何使用 RabbitMQConsumer 监听来自后端服务器的消息，
并通过专门的发送线程发送响应消息。
配置从 .env 文件读取。

启动方式
--------
.. code-block:: bash

    python -m neuracle.rabbitmq.demo.rabbitmq_example
"""

import json
import logging
import threading
import time
from functools import partial
from queue import Queue
from typing import Any

from neuracle.config import get_rabbitmq_config, load_env
from neuracle.config.env import mask_rabbitmq_config
from neuracle.logger import setup_logging
from neuracle.rabbitmq import RabbitMQConsumer, RabbitMQPublisher
from neuracle.utils.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def ack_in_consumer_thread(channel: Any, delivery_tag: int, message_type: str) -> None:
    """在消费线程中实际发送 ack，避免工作线程直接操作 channel。"""
    channel.basic_ack(delivery_tag=delivery_tag)
    logger.info("ack_sent: type=%s delivery_tag=%s", message_type, delivery_tag)


def schedule_ack(
    channel: Any, delivery_tag: int, message_type: str, reason: str
) -> None:
    """把 ack 请求投递回 RabbitMQ 消费线程执行。"""
    logger.info(
        "ack_scheduled: type=%s delivery_tag=%s reason=%s",
        message_type,
        delivery_tag,
        reason,
    )
    channel.connection.add_callback_threadsafe(
        partial(ack_in_consumer_thread, channel, delivery_tag, message_type)
    )


def process_message_async(
    message: dict[str, Any],
    message_queue: Queue,
    channel: Any,
    delivery_tag: int,
) -> None:
    """模拟一个异步长任务，完成后再回到消费线程 ack。"""
    message_type = str(message.get("type", "unknown"))
    try:
        logger.info("worker_started: type=%s", message_type)
        time.sleep(1.0)
        response = {
            "type": "response",
            "original_type": message_type,
            "status": "processed",
            "message": "已处理消息",
        }
        message_queue.put(response)
        logger.info("响应已加入发送队列")
        logger.info("worker_finished: type=%s", message_type)
        schedule_ack(channel, delivery_tag, message_type, "task_finished")
    except Exception:
        logger.error("处理消息时发生错误")
        schedule_ack(channel, delivery_tag, message_type, "task_exception")


def message_callback(channel, method, properties, body, message_queue: Queue):
    """
    消息处理回调函数

    原理
    ----
        收到消息后解析并放入发送队列，由专门的发送线程处理。
        这样可以避免阻塞主线程的监听循环，同时复用发送连接。

    Parameters
    ----------
    channel : pika.channel.Channel
        RabbitMQ 通道对象
    method : pika.spec.Basic.Deliver
        消息传递信息
    properties : pika.spec.BasicProperties
        消息属性
    body : bytes
        消息内容
    message_queue : Queue
        用于把结果作为消息返回给发送消息的线程
    """
    try:
        message = json.loads(body.decode("utf-8"))
        logger.info("收到消息: %s", message)
        worker = threading.Thread(
            target=process_message_async,
            args=(message, message_queue, channel, method.delivery_tag),
            daemon=True,
            name=f"rabbitmq-example-{message.get('type', 'unknown')}",
        )
        worker.start()
        logger.info("消息已交给工作线程: %s", worker.name)
    except json.JSONDecodeError as e:
        logger.error("无法解析 JSON 消息: %s", e)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        logger.error("处理消息时发生错误: %s", e)
        channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    """主函数"""
    setup_logging(str(PROJECT_ROOT / "log" / "rabbitmq_example"))
    load_env()
    config = get_rabbitmq_config()
    logger.info("RabbitMQ 配置: %s", mask_rabbitmq_config(config))
    message_queue = Queue()

    sender_thread = RabbitMQPublisher(
        host=config["host"],
        port=config["port"],
        queue_name=config["send_queue_name"],
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
        queue_name=config["listen_queue_name"],
        username=config["username"],
        password=config["password"],
        virtual_host=config["virtual_host"],
        heartbeat=config["heartbeat"],
        blocked_connection_timeout=config["blocked_connection_timeout"],
        socket_timeout=config["socket_timeout"],
        connection_attempts=config["connection_attempts"],
        retry_delay=config["retry_delay"],
    )

    callback_with_args = partial(message_callback, message_queue=message_queue)
    logger.info("开始监听消息...")
    try:
        listener.consume_forever(callback_with_args)
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止...")
    sender_thread.stop()
    sender_thread.join(timeout=5)
    logger.info("程序退出")


if __name__ == "__main__":
    main()
