"""
RabbitMQ Sender 测试示例

用于测试 main.py 服务的演示脚本。
使用 RabbitMQSender 发送测试消息到 backend_to_simnibs 队列，
并监听 simnibs_to_backend 队列接收进度和结果。

启动方式: python -m neuracle.rabbitmq.demo.main_sender_demo
"""

import json
import logging
import os
import threading
import time
from queue import Empty, Queue

from neuracle.config import get_rabbitmq_config, load_env
from neuracle.logger import setup_logging
from neuracle.rabbitmq import (
    RabbitMQConsumer,
    RabbitMQSender,
    build_ack_test_message,
)
from neuracle.utils import BUILT_IN_DIR_PATH, PROJECT_ROOT, STANDARD_COND

logger = logging.getLogger(__name__)


def create_model_message(task_id: str, with_dti: bool = False) -> dict:
    """创建 model 任务消息"""
    params = {
        "T1_file_path": f"{BUILT_IN_DIR_PATH}/T1.nii.gz",
        "T2_file_path": f"{BUILT_IN_DIR_PATH}/T2_reg.nii.gz",
        "dir_path": BUILT_IN_DIR_PATH,
    }
    if with_dti:
        params["DTI_file_path"] = f"{BUILT_IN_DIR_PATH}/DTI_coregT1_tensor.nii.gz"
    return {"id": task_id, "type": "model", "params": params}


def create_forward_message(
    task_id: str,
    montage: str,
    anisotropy: bool = False,
    electrode_a: list[dict] | None = None,
    electrode_b: list[dict] | None = None,
) -> dict:
    """创建 forward 任务消息"""
    if electrode_a is None:
        electrode_a = [
            {"name": "F5", "current_mA": 2.0},
            {"name": "P5", "current_mA": -2.0},
        ]
    if electrode_b is None:
        electrode_b = [
            {"name": "F6", "current_mA": 1.0},
            {"name": "P6", "current_mA": -1.0},
        ]
    params = {
        "dir_path": BUILT_IN_DIR_PATH,
        "T1_file_path": f"{BUILT_IN_DIR_PATH}/T1.nii.gz",
        "montage": montage,
        "electrode_A": electrode_a,
        "electrode_B": electrode_b,
        "conductivity_config": STANDARD_COND,
        "anisotropy": anisotropy,
    }
    if anisotropy:
        params["DTI_file_path"] = f"{BUILT_IN_DIR_PATH}/DTI_coregT1_tensor.nii.gz"
    return {"id": task_id, "type": "forward", "params": params}


def create_inverse_message(
    task_id: str, montage: str, anisotropy: bool = False
) -> dict:
    """创建 inverse 任务消息"""
    params = {
        "dir_path": BUILT_IN_DIR_PATH,
        "T1_file_path": f"{BUILT_IN_DIR_PATH}/T1.nii.gz",
        "montage": montage,
        "current_A": [0.002, -0.002],
        "current_B": [0.001, -0.001],
        "roi_type": "mni_pos",
        "roi_param": {
            "mni_param": {
                "center": [-38.5, -22.5, 58.3],
                "radius": 15.0,
            },
            "atlas_param": None,
        },
        "target_threshold": 0.5,
        "conductivity_config": STANDARD_COND,
        "anisotropy": anisotropy,
    }
    if anisotropy:
        params["DTI_file_path"] = f"{BUILT_IN_DIR_PATH}/DTI_coregT1_tensor.nii.gz"
    return {"id": task_id, "type": "inverse", "params": params}


def create_inverse_atlas_message(
    task_id: str,
    montage: str,
    atlas_name: str = "BN",
    area_name: str = "A8m_L",
    anisotropy: bool = False,
) -> dict:
    """创建 inverse atlas 任务消息。

    注意：
        atlas ROI 依赖离线标准化产物，首次使用前需要先运行
        neuracle/docs/atlas_roi_offline_preprocessing_plan.md 中记录的预处理脚本。
    """
    params = {
        "dir_path": BUILT_IN_DIR_PATH,
        "T1_file_path": f"{BUILT_IN_DIR_PATH}/T1.nii.gz",
        "montage": montage,
        "current_A": [0.002, -0.002],
        "current_B": [0.001, -0.001],
        "roi_type": "atlas",
        "roi_param": {
            "mni_param": None,
            "atlas_param": {
                "name": atlas_name,
                "area": area_name,
            },
        },
        "target_threshold": 0.5,
        "conductivity_config": STANDARD_COND,
        "anisotropy": anisotropy,
    }
    if anisotropy:
        params["DTI_file_path"] = f"{BUILT_IN_DIR_PATH}/DTI_coregT1_tensor.nii.gz"
    return {"id": task_id, "type": "inverse", "params": params}


def create_ack_test_message(task_id: str, sleep_seconds: float = 10.0) -> dict:
    """创建 ack 时机验证消息。"""
    return build_ack_test_message(task_id, sleep_seconds=sleep_seconds)


def progress_callback(channel, method, properties, body, message_queue: Queue):
    """进度消息回调函数"""
    try:
        message = json.loads(body.decode("utf-8"))
        logger.info("收到进度消息: %s", json.dumps(message, ensure_ascii=False))
        message_queue.put(message)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except json.JSONDecodeError as e:
        logger.error("无法解析 JSON 消息: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as e:
        logger.error("处理消息时发生错误: %s", e)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def run_consumer_thread(
    host: str,
    port: int,
    queue_name: str,
    message_queue: Queue,
    username: str = "guest",
    password: str = "guest",
    virtual_host: str = "/",
    heartbeat: int = 60,
    blocked_connection_timeout: int = 300,
    socket_timeout: int = 10,
    connection_attempts: int = 5,
    retry_delay: int = 5,
) -> tuple[RabbitMQConsumer, threading.Thread]:
    """启动监听线程"""
    consumer = RabbitMQConsumer(
        host=host,
        port=port,
        queue_name=queue_name,
        username=username,
        password=password,
        virtual_host=virtual_host,
        heartbeat=heartbeat,
        blocked_connection_timeout=blocked_connection_timeout,
        socket_timeout=socket_timeout,
        connection_attempts=connection_attempts,
        retry_delay=retry_delay,
    )

    callback = lambda ch, meth, props, body: progress_callback(  # noqa: E731
        ch, meth, props, body, message_queue
    )
    thread = threading.Thread(
        target=consumer.consume_forever,
        args=(callback,),
        daemon=True,
    )
    thread.start()
    logger.info("监听线程已启动，监听队列: %s", queue_name)
    return consumer, thread


def wait_for_completion(
    message_queue: Queue, task_id: str, timeout: int | None = None
) -> dict | None:
    """等待任务完成"""
    last_progress = 0
    start_time = time.time()

    while True:
        if timeout is not None and time.time() - start_time >= timeout:
            logger.warning("等待任务完成超时，task_id: %s", task_id)
            return None
        try:
            message = message_queue.get(timeout=1)
            if message.get("id") == task_id:
                progress_rate = message.get("progress_rate", 0)
                status_message = message.get("message", "")
                if progress_rate > last_progress:
                    logger.info(
                        "任务进度: %d%% - %s",
                        progress_rate,
                        status_message,
                    )
                    last_progress = progress_rate
                if (
                    progress_rate == 0
                    and status_message
                    and status_message not in ("任务开始",)
                ):
                    logger.error("任务失败，返回消息: %s", status_message)
                    return message
                if progress_rate == 100:
                    logger.info("任务完成，结果: %s", message.get("result"))
                    return message
        except Empty:
            continue
        except Exception as e:
            logger.exception("等待任务完成时处理消息队列发生错误: %s", e)
            continue


def send_message_and_wait(
    host: str,
    port: int,
    send_queue: str,
    recv_queue: str,
    message: dict,
    username: str = "guest",
    password: str = "guest",
    virtual_host: str = "/",
    heartbeat: int = 60,
    blocked_connection_timeout: int = 300,
    socket_timeout: int = 10,
    connection_attempts: int = 5,
    retry_delay: int = 5,
    timeout: int | None = None,
) -> dict | None:
    """发送消息并等待结果"""
    message_queue: Queue = Queue()

    consumer, consumer_thread = run_consumer_thread(
        host,
        port,
        recv_queue,
        message_queue,
        username=username,
        password=password,
        virtual_host=virtual_host,
        heartbeat=heartbeat,
        blocked_connection_timeout=blocked_connection_timeout,
        socket_timeout=socket_timeout,
        connection_attempts=connection_attempts,
        retry_delay=retry_delay,
    )
    if consumer_thread is None:
        return None

    sender = RabbitMQSender(
        host=host,
        port=port,
        queue_name=send_queue,
        username=username,
        password=password,
        virtual_host=virtual_host,
        heartbeat=heartbeat,
        blocked_connection_timeout=blocked_connection_timeout,
        socket_timeout=socket_timeout,
        connection_attempts=connection_attempts,
        retry_delay=retry_delay,
    )
    if not sender.connect():
        logger.error("发送器连接失败")
        return None

    task_id = message.get("id", "unknown")
    logger.info("发送消息，task_id: %s, type: %s", task_id, message.get("type"))
    sender.send_message(message)

    result = wait_for_completion(message_queue, task_id, timeout)

    try:
        sender.close()
    except Exception as e:
        logger.warning("关闭发送器连接时发生错误: %s", e)
    try:
        consumer.stop_consume()
    except Exception as e:
        logger.warning("停止监听器时发生错误: %s", e)
    consumer_thread.join(timeout=5)
    try:
        consumer.close()
    except Exception as e:
        logger.warning("关闭监听器连接时发生错误: %s", e)
    return result


def main():
    """主函数"""
    setup_logging(
        os.path.join(
            PROJECT_ROOT,
            "log",
            "main_sendor_demo",
        )
    )
    load_env()

    config = get_rabbitmq_config()
    masked_config = dict(config)
    if masked_config.get("password"):
        masked_config["password"] = "***"
    logger.info("RabbitMQ 配置: %s", masked_config)

    host = config["host"]
    port = config["port"]

    to_main_queue = config["listen_queue_name"]
    from_main_queue = config["send_queue_name"]

    logger.info(
        "发送队列(给main.py): %s，接收队列(来自main.py): %s",
        to_main_queue,
        from_main_queue,
    )

    model_msg = create_model_message("test_model_001", with_dti=True)
    forward_msg = create_forward_message(
        "test_forward_001", "EEG10-10_UI_Jurak_2007", anisotropy=False
    )
    forward_aniso_msg = create_forward_message(
        "test_forward_002",
        "EEG10-20_Okamoto_2004",
        anisotropy=True,
        electrode_a=[
            {"name": "F3", "current_mA": 1.0},
            {"name": "P3", "current_mA": -1.0},
        ],
        electrode_b=[
            {"name": "F4", "current_mA": 1.0},
            {"name": "P4", "current_mA": -1.0},
        ],
    )
    inverse_msg = create_inverse_message(
        "test_inverse_001", "EEG10-10_Cutini_2011", anisotropy=False
    )
    inverse_aniso_msg = create_inverse_message(
        "test_inverse_002", "EEG10-20_extended_SPM12", anisotropy=True
    )
    inverse_atlas_msg = create_inverse_atlas_message(
        "test_inverse_003",
        "EEG10-10_Cutini_2011",
        atlas_name="BN",
        area_name="A8m_L",
        anisotropy=False,
    )
    ack_test_msg = create_ack_test_message("test_ack_001", sleep_seconds=10.0)

    menu = {
        "1": ("model 任务（头模生成）", model_msg),
        "2": ("forward 任务（标量电导率）", forward_msg),
        "3": ("forward 任务（各向异性电导率）", forward_aniso_msg),
        "4": ("inverse 任务（标量电导率）", inverse_msg),
        "5": ("inverse 任务（各向异性电导率）", inverse_aniso_msg),
        "6": ("inverse 任务（Atlas ROI，需先离线预处理）", inverse_atlas_msg),
        "7": ("ack_test 任务（验证完成后 ack）", ack_test_msg),
    }

    while True:
        print("\n请选择要发送的消息：")
        for key, (desc, _) in menu.items():
            print(f"  {key}. {desc}")
        print("  q. 退出")

        choice = input("\n请输入选项: ").strip()

        if choice.lower() == "q":
            logger.info("退出程序")
            break

        if choice not in menu:
            logger.warning("无效的选项: %s", choice)
            continue

        desc, msg = menu[choice]
        logger.info("=" * 60)
        logger.info("发送 %s", desc)
        logger.info("=" * 60)
        result = send_message_and_wait(
            host,
            port,
            to_main_queue,
            from_main_queue,
            msg,
            username=config["username"],
            password=config["password"],
            virtual_host=config["virtual_host"],
            heartbeat=config["heartbeat"],
            blocked_connection_timeout=config["blocked_connection_timeout"],
            socket_timeout=config["socket_timeout"],
            connection_attempts=config["connection_attempts"],
            retry_delay=config["retry_delay"],
        )

        if result is None:
            logger.error("任务执行失败或超时")
        else:
            logger.info("任务完成")


if __name__ == "__main__":
    main()
