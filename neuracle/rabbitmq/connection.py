"""
RabbitMQ 连接管理模块

提供 RabbitMQ 连接管理功能，支持连接参数的构建、重连机制。

该模块主要被 sender.py 和 consumer.py 内部使用，
用于建立与 RabbitMQ 服务器的连接。

主要组件
-------
- RabbitMQConnection：连接管理类，负责建立和管理 RabbitMQ 连接
  - connect()：建立连接
  - close()：关闭连接
  - is_connected()：检查连接状态
  - from_config()：从配置文件创建连接实例

连接参数说明
-----------
- heartbeat：客户端和服务器之间的心跳间隔，用于检测连接是否存活
- blocked_connection_timeout：连接被阻塞时的超时时间
- socket_timeout：socket 通信超时时间
- connection_attempts：连接失败时的重试次数
- retry_delay：两次连接重试之间的间隔
"""

import logging
from typing import Any

from pika import (
    BlockingConnection,
    ConnectionParameters,
    PlainCredentials,
)
from pika.exceptions import AMQPConnectionError

from neuracle.config import get_rabbitmq_config

logger = logging.getLogger(__name__)


class RabbitMQConnection:
    """
    RabbitMQ 连接管理类

    封装 RabbitMQ 连接的生命周期管理，包括连接建立、参数配置和关闭。
    使用 pika 库的 BlockingConnection 模式（同步阻塞式连接）。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        virtual_host: str = "/",
        heartbeat: int = 60,
        blocked_connection_timeout: int = 300,
        socket_timeout: int = 10,
        connection_attempts: int = 5,
        retry_delay: int = 5,
    ):
        """
        初始化 RabbitMQ 连接管理器

        Parameters
        ----------
        host : str
            RabbitMQ 服务器地址
        port : int
            RabbitMQ 服务器端口
        username : str
            用户名
        password : str
            密码
        virtual_host : str
            虚拟主机（类似于命名空间，用于隔离不同应用的队列）
        heartbeat : int
            心跳间隔（秒），0 表示禁用心跳
        blocked_connection_timeout : int
            阻塞连接超时（秒），当连接被阻塞时的最大等待时间
        socket_timeout : int
            socket 超时（秒），socket 读写操作的超时时间
        connection_attempts : int
            连接重试次数，连接失败时的最大重试次数
        retry_delay : int
            重试间隔（秒），两次连接重试之间的等待时间
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.heartbeat = heartbeat
        self.blocked_connection_timeout = blocked_connection_timeout
        self.socket_timeout = socket_timeout
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        self.connection: BlockingConnection | None = None  # RabbitMQ 连接对象
        self.channel: Any = None  # RabbitMQ 通道对象

    def _build_connection_parameters(self) -> ConnectionParameters:
        """
        构造 RabbitMQ 连接参数

        将实例的配置属性转换为 pika 库所需的 ConnectionParameters 对象。
        使用 PlainCredentials 进行用户名密码认证。

        Returns
        -------
        ConnectionParameters
            pika 连接参数对象
        """
        # PlainCredentials：简单的用户名/密码认证方式
        credentials = PlainCredentials(self.username, self.password)
        return ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.virtual_host,
            credentials=credentials,
            heartbeat=self.heartbeat,
            blocked_connection_timeout=self.blocked_connection_timeout,
            socket_timeout=self.socket_timeout,
            connection_attempts=self.connection_attempts,
            retry_delay=self.retry_delay,
        )

    def connect(self) -> bool:
        """
        连接到 RabbitMQ 服务器

        使用 BlockingConnection 建立同步连接，建立成功后创建 channel。

        Returns
        -------
        bool
            连接是否成功
        """
        try:
            parameters = self._build_connection_parameters()
            # BlockingConnection：同步阻塞式连接，会等待服务器响应
            self.connection = BlockingConnection(parameters)
            # channel：通道，用于发送和接收消息
            self.channel = self.connection.channel()
            logger.info(
                "成功连接到 RabbitMQ 服务器: host=%s port=%s vhost=%s user=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
            )
            return True
        except AMQPConnectionError as e:
            # AMQP 连接错误（认证失败、服务器不可达等）
            logger.error(
                "连接 RabbitMQ 失败: host=%s port=%s vhost=%s user=%s error_type=%s error=%r",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                type(e).__name__,
                e,
            )
            return False
        except Exception as e:
            # 其他未知错误
            logger.error(
                "连接时发生未知错误: host=%s port=%s vhost=%s user=%s error_type=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                type(e).__name__,
            )
            return False

    def close(self) -> None:
        """
        关闭 RabbitMQ 连接

        安全关闭连接，如果连接已关闭或不存在则忽略。
        """
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("RabbitMQ 连接已关闭")

    def is_connected(self) -> bool:
        """
        检查连接是否处于打开状态

        Returns
        -------
        bool
            连接是否处于打开状态
        """
        return self.connection is not None and self.connection.is_open

    @staticmethod
    def from_config() -> "RabbitMQConnection":
        """
        从配置文件创建连接管理器

        从 neuracle.config 模块读取 RabbitMQ 配置，创建一个连接实例。

        Returns
        -------
        RabbitMQConnection
            根据配置创建的连接管理器实例
        """
        config = get_rabbitmq_config()
        return RabbitMQConnection(
            host=config["host"],
            port=config["port"],
            username=config["username"],
            password=config["password"],
            virtual_host=config["virtual_host"],
            heartbeat=config["heartbeat"],
            blocked_connection_timeout=config["blocked_connection_timeout"],
            socket_timeout=config["socket_timeout"],
            connection_attempts=config["connection_attempts"],
            retry_delay=config["retry_delay"],
        )
