"""
RabbitMQ 消息发送模块

用于向 RabbitMQ 队列发送消息。

该模块是发送功能的核心组件，被 publisher.py 中的 RabbitMQPublisher 线程使用。
每个消息都会创建一个临时的 Sender 实例，发送完成后关闭连接。

消息发送流程
---------
1. RabbitMQPublisher 从 message_queue 取消息
2. 创建 RabbitMQSender 并 connect()
3. 调用 send_message() 发送消息
4. 关闭连接

特点
----
- 每次发送都创建新连接（避免连接复用导致的复杂状态问题）
- 支持发送失败后自动重连
- 使用持久化消息（delivery_mode=2）
"""

import json
import logging
from typing import Any

from pika import (
    BasicProperties,
    BlockingConnection,
    ConnectionParameters,
    DeliveryMode,
    PlainCredentials,
)
from pika.exceptions import AMQPConnectionError

logger = logging.getLogger(__name__)


class RabbitMQSender:
    """
    RabbitMQ 发送器类

    负责向 RabbitMQ 队列发送消息。
    与 RabbitMQConsumer 不同，Sender 是"推"消息到队列，而 Consumer 是"拉"消息。

    每次发送消息时，都会创建新的 Sender 实例（见 publisher.py），
    发送完成后立即关闭连接。这种设计简化了连接管理。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        queue_name: str = "",
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
        初始化 RabbitMQ 发送器

        Parameters
        ----------
        host : str
            RabbitMQ 服务器地址
        port : int
            RabbitMQ 服务器端口
        queue_name : str
            队列名称（消息将发送到这个队列）
        username : str
            用户名
        password : str
            密码
        virtual_host : str
            虚拟主机
        heartbeat : int
            心跳间隔（秒）
        blocked_connection_timeout : int
            阻塞连接超时（秒）
        socket_timeout : int
            socket 超时（秒）
        connection_attempts : int
            连接重试次数
        retry_delay : int
            重试间隔（秒）
        """
        self.host = host
        self.port = port
        self.queue_name = queue_name
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.heartbeat = heartbeat
        self.blocked_connection_timeout = blocked_connection_timeout
        self.socket_timeout = socket_timeout
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        self.connection: BlockingConnection | None = None
        self.channel = None

    def _build_connection_parameters(self) -> ConnectionParameters:
        """
        构造 RabbitMQ 连接参数

        Returns
        -------
        ConnectionParameters
            pika 连接参数对象
        """
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
        连接到 RabbitMQ 服务器并声明队列

        Returns
        -------
        bool
            连接是否成功
        """
        try:
            parameters = self._build_connection_parameters()
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            if self.queue_name:
                # 声明队列（确保队列存在）
                self.channel.queue_declare(
                    queue=self.queue_name,
                    durable=True,  # 持久化队列
                    arguments={
                        "x-queue-type": "quorum",  # Quorum 队列
                        "x-consumer-timeout": 86400000,  # 消费者超时 24 小时
                    },
                )
            logger.info(
                "发送器成功连接到 RabbitMQ 服务器: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            return True
        except AMQPConnectionError as e:
            logger.error(
                "发送器连接 RabbitMQ 失败: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s error=%r",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
                e,
            )
            return False
        except Exception as e:
            logger.error(
                "发送器连接时发生未知错误: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
            )
            return False

    def send_message(self, message: Any) -> bool:
        """
        发送消息

        原理
        ----
            1. 将消息对象序列化为 JSON 格式的 UTF-8 bytes
            2. 设置消息属性（持久化模式）
            3. 通过 basic_publish 发送到指定队列
            4. 若发送失败，尝试重连后重试

        消息持久化
        ----
            delivery_mode=DeliveryMode.Persistent (2)
            表示消息会写入磁盘，服务器重启后消息不会丢失

        Parameters
        ----------
        message : Any
            消息内容（可序列化为 JSON 的对象，通常是 dict）

        Returns
        -------
        bool
            发送是否成功
        """
        if not self.channel:
            logger.error("通道未连接，无法发送消息")
            return False
        try:
            # 将消息序列化为 JSON bytes（ensure_ascii=False 支持中文）
            body = json.dumps(message, ensure_ascii=False).encode("utf-8")
            # 设置消息持久化属性
            properties = BasicProperties(
                delivery_mode=DeliveryMode.Persistent,  # 持久化消息
            )
            # basic_publish：发送到默认交换机（exchange=""），路由键为队列名
            self.channel.basic_publish(
                exchange="",  # 默认交换机
                routing_key=self.queue_name,  # 队列名称作为路由键
                body=body,  # 消息体
                properties=properties,  # 消息属性
            )
            logger.info("消息已发送: %s", message)
            return True
        except Exception as e:
            # 发送失败，尝试重连后重发
            logger.warning("发送消息失败: %s，尝试重连...", e)
            if self._reconnect():
                try:
                    self.channel.basic_publish(
                        exchange="",
                        routing_key=self.queue_name,
                        body=body,
                        properties=properties,
                    )
                    logger.info("重连后消息发送成功: %s", message)
                    return True
                except Exception as e2:
                    logger.error("重连后发送消息仍然失败: %s", e2)
                    return False
            return False

    def _reconnect(self) -> bool:
        """
        重新建立连接

        在发送失败后尝试重新连接，如果当前连接存在则先关闭。

        Returns
        -------
        bool
            重连是否成功
        """
        try:
            logger.info(
                "尝试重新连接 RabbitMQ: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            # 先尝试关闭现有连接（忽略可能的错误）
            if self.connection and self.connection.is_open:
                try:
                    self.connection.close()
                except Exception:
                    pass
            # 创建新连接
            parameters = self._build_connection_parameters()
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # 重新声明队列
            if self.queue_name:
                self.channel.queue_declare(
                    queue=self.queue_name,
                    durable=True,
                    arguments={"x-queue-type": "quorum"},
                )
            logger.info(
                "重连成功: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            return True
        except Exception as e:
            logger.error(
                "重连失败: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
            )
            return False

    def close(self) -> None:
        """
        关闭 RabbitMQ 连接

        安全关闭连接，处理可能的重复关闭情况
        """
        if self.connection and self.connection.is_open:
            self.connection.close()
            logger.info("发送器连接已关闭")
        else:
            logger.warning("发送器还未连接时就调用close")
