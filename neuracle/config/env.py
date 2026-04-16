"""
环境变量配置读取模块

提供从 .env 文件读取配置的函数，包括 RabbitMQ 和阿里云 OSS 配置。
"""

import logging
import os
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv

from neuracle.utils.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)

# .env 文件路径
_ENV_FILE: Final[Path] = PROJECT_ROOT / "neuracle" / "config" / ".env"


def load_env() -> None:
    """
    加载 .env 文件。

    Notes
    -----
    使用 python-dotenv 库加载 .env 文件中的环境变量到 os.environ。
    如果 .env 文件不存在，静默失败。
    """
    load_dotenv(_ENV_FILE)


def get_rabbitmq_config() -> dict:
    """
    获取 RabbitMQ 配置。

    Returns
    -------
    dict
        包含 host、port、username、password、virtual_host、heartbeat、
        blocked_connection_timeout、socket_timeout、connection_attempts、
        retry_delay、listen_queue_name、send_queue_name 的配置字典

    Notes
    -----
    配置项从环境变量读取，如环境变量不存在则使用默认值。
    """
    return {
        "host": os.getenv("RABBITMQ_HOST", "localhost"),
        "port": int(os.getenv("RABBITMQ_PORT", "5672")),
        "username": os.getenv("RABBITMQ_USERNAME", "guest"),
        "password": os.getenv("RABBITMQ_PASSWORD", "guest"),
        "virtual_host": os.getenv("RABBITMQ_VHOST", "/"),
        "heartbeat": int(os.getenv("RABBITMQ_HEARTBEAT", "60")),
        "blocked_connection_timeout": int(
            os.getenv("RABBITMQ_BLOCKED_CONNECTION_TIMEOUT", "300")
        ),
        "socket_timeout": int(os.getenv("RABBITMQ_SOCKET_TIMEOUT", "10")),
        "connection_attempts": int(os.getenv("RABBITMQ_CONNECTION_ATTEMPTS", "5")),
        "retry_delay": int(os.getenv("RABBITMQ_RETRY_DELAY", "5")),
        "listen_queue_name": os.getenv("RABBITMQ_LISTEN_QUEUE_NAME", ""),
        "send_queue_name": os.getenv("RABBITMQ_SEND_QUEUE_NAME", ""),
    }


def get_aliyun_config() -> dict:
    """
    获取阿里云 OSS 配置。

    Returns
    -------
    dict
        包含 access_key_id、access_key_secret、sts_role_arn、sts_endpoint、
        oss_endpoint、bucket_name、bucket_target 的配置字典

    Notes
    -----
    配置项从环境变量读取，如环境变量不存在则使用默认值。
    """
    return {
        "access_key_id": os.getenv("ALI_ACCESS_KEY_ID", ""),
        "access_key_secret": os.getenv("ALI_ACCESS_KEY_SECRET", ""),
        "sts_role_arn": os.getenv("ALI_STS_ROLE_ARN", ""),
        "sts_endpoint": os.getenv("ALI_STS_ENDPOINT", "sts.cn-shanghai.aliyuncs.com"),
        "oss_endpoint": os.getenv(
            "ALI_OSS_ENDPOINT", "https://oss-cn-shanghai.aliyuncs.com"
        ),
        "bucket_name": os.getenv("ALI_BUCKET_NAME", ""),
        "bucket_target": os.getenv("ALI_BUCKET_TARGET", ""),
    }


def mask_rabbitmq_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    隐藏 RabbitMQ 配置中的敏感信息，避免密码写入日志

    Parameters
    ----------
    config : dict[str, Any]
        原始配置字典

    Returns
    -------
    dict[str, Any]
        密码被掩码处理后的配置字典
    """
    masked = dict(config)
    if masked.get("password"):
        masked["password"] = "***"
    return masked
