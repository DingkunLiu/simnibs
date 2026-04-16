"""
RabbitMQ ack 时机验证 demo

用途
----
1. 向 main.py 监听的任务队列连续发送两条 ack_test 消息
2. 每条消息都模拟一个可控的长任务
3. 配合 main.py 日志中的 received / worker_started / worker_finished / ack_scheduled / ack_sent
   验证是否真的做到"任务完成后才 ack"

启动方式
--------
.. code-block:: bash

    # 先启动 RabbitMQ
    # 再启动 SimNIBS 服务
    python -m neuracle.rabbitmq.scheduler
    # 最后运行本 demo
    python -m neuracle.rabbitmq.demo.rabbitmq_ack_timing_demo
"""

import json
import logging
import threading
import time
from queue import Empty, Queue
from typing import Any

from neuracle.config import get_rabbitmq_config, load_env
from neuracle.logger import setup_logging
from neuracle.rabbitmq import (
    RabbitMQConsumer,
    RabbitMQSender,
    build_ack_test_message,
)
from neuracle.utils.constants import PROJECT_ROOT

logger = logging.getLogger("neuracle.rabbitmq.demo.rabbitmq_ack_timing_demo")


def progress_callback(
    channel: Any, method: Any, properties: Any, body: bytes, message_queue: Queue
) -> None:
    """接收 main.py 发送回来的进度消息。"""
    try:
        message = json.loads(body.decode("utf-8"))
        message["_received_at"] = time.strftime("%H:%M:%S")
        logger.info("收到进度消息: %s", json.dumps(message, ensure_ascii=False))
        message_queue.put(message)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception:
        logger.error("处理进度消息失败")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def start_progress_listener(
    config: dict[str, Any], message_queue: Queue
) -> tuple[RabbitMQConsumer, threading.Thread]:
    """在后台线程监听进度队列。"""
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
    callback = lambda ch, meth, props, body: progress_callback(
        ch, meth, props, body, message_queue
    )
    thread = threading.Thread(
        target=listener.consume_forever,
        args=(callback,),
        daemon=True,
        name="ack-timing-progress-listener",
    )
    thread.start()
    return listener, thread


def send_ack_test_messages(config: dict[str, Any], sleep_seconds: float) -> list[str]:
    """连续发送两条 ack_test 消息。"""
    sender = RabbitMQSender(
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
    if not sender.connect():
        raise RuntimeError("发送器连接失败，无法发送 ack_test 消息")

    try:
        task_ids = [
            f"ack_test_{int(time.time())}_1",
            f"ack_test_{int(time.time())}_2",
        ]
        for task_id in task_ids:
            message = build_ack_test_message(task_id, sleep_seconds=sleep_seconds)
            logger.info(
                "发送 ack_test 消息: task_id=%s sleep_seconds=%.1f",
                task_id,
                sleep_seconds,
            )
            sender.send_message(message)
        return task_ids
    finally:
        sender.close()


def wait_for_tasks(
    message_queue: Queue, task_ids: list[str], timeout: float
) -> dict[str, dict[str, Any]]:
    """等待两条 ack_test 都完成。"""
    results: dict[str, dict[str, Any]] = {}
    start = time.time()
    while len(results) < len(task_ids):
        if time.time() - start > timeout:
            raise TimeoutError(f"等待 ack_test 超时，当前结果: {results}")
        try:
            message = message_queue.get(timeout=1)
        except Empty:
            continue
        task_id = message.get("id")
        if task_id not in task_ids:
            continue
        if message.get("progress_rate") == 100:
            results[task_id] = message
    return results


def main() -> None:
    """主函数"""
    setup_logging(str(PROJECT_ROOT / "log" / "rabbitmq_ack_timing_demo"))
    load_env()
    config = get_rabbitmq_config()
    sleep_seconds = 15.0
    timeout = 120.0

    logger.info(
        "ack timing demo 启动: send_queue=%s recv_queue=%s sleep_seconds=%.1f",
        config["listen_queue_name"],
        config["send_queue_name"],
        sleep_seconds,
    )
    logger.info(
        "请同时观察 SimNIBS 服务日志中的 received / worker_started / worker_finished / ack_scheduled / ack_sent"
    )

    progress_queue: Queue = Queue()
    listener, listener_thread = start_progress_listener(config, progress_queue)
    time.sleep(1)

    try:
        task_ids = send_ack_test_messages(config, sleep_seconds=sleep_seconds)
        results = wait_for_tasks(progress_queue, task_ids, timeout=timeout)
        logger.info("ack timing demo 完成，任务结果如下：")
        for task_id in task_ids:
            result = results[task_id]
            logger.info(
                "task_id=%s progress=%s received_at=%s result=%s",
                task_id,
                result.get("progress_rate"),
                result.get("_received_at"),
                result.get("result"),
            )
    finally:
        listener.stop_consume()
        listener_thread.join(timeout=5)
        listener.close()


if __name__ == "__main__":
    main()
