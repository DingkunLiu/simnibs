"""
RabbitMQ 消息消费者模块

用于监听 RabbitMQ 队列，接收来自后端服务器的消息。

该模块是服务架构中的"监听组件"，负责：
1. 连接到 RabbitMQ 服务器的指定队列
2. 持续监听队列中的新消息
3. 收到消息后触发回调函数进行处理
4. 支持断线重连

队列配置
-------
- 使用 durable=True 确保队列持久化，服务器重启后队列仍存在
- 使用 x-queue-type=quorum 选择 Quorum 队列类型，提供更高的可靠性
- 使用 x-consumer-timeout=86400000（24小时）设置消费者超时时间

消息确认机制
-----------
- auto_ack=False：关闭自动确认
- 消费者必须在处理完消息后手动调用 basic_ack()
- 如果处理失败，消息会被重新投递（redelivered=True）
"""

import logging
import time
from typing import Any, Callable

from pika import BlockingConnection, ConnectionParameters, PlainCredentials
from pika.exceptions import AMQPConnectionError, ConnectionWrongStateError

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    """
    RabbitMQ 消费者类

    负责从 RabbitMQ 队列接收消息并触发回调处理。
    支持自动重连，确保服务的可靠性。

    使用方式
    ------
    consumer = RabbitMQConsumer(host="localhost", port=5672, queue_name="my_queue")
    consumer.consume_forever(my_callback)  # 阻塞运行
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
        prefetch_count: int = 1,
    ):
        """
        初始化 RabbitMQ 消费者

        Parameters
        ----------
        host : str
            RabbitMQ 服务器地址
        port : int
            RabbitMQ 服务器端口
        queue_name : str
            队列名称（要监听的队列）
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
        prefetch_count : int
            预取消息数量（QoS 设置），表示同时确认的最大消息数
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
        self.prefetch_count = prefetch_count
        self.connection: BlockingConnection | None = None
        self.channel = None
        self._stopped = False  # 用于控制停止消费循环

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

        连接成功后，会创建 channel 并声明要监听的队列。
        队列配置：
        - durable=True：队列持久化，服务器重启后队列仍存在
        - x-queue-type=quorum：使用 Quorum 队列，提供更高可靠性
        - x-consumer-timeout=86400000：消费者超时时间 24 小时

        Returns
        -------
        bool
            连接是否成功
        """
        try:
            parameters = self._build_connection_parameters()
            self.connection = BlockingConnection(parameters)
            self.channel = self.connection.channel()
            # 声明队列（若不存在则创建）
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True,  # 持久化队列
                arguments={
                    "x-queue-type": "quorum",  # Quorum 队列类型
                    "x-consumer-timeout": 86400000,  # 消费者超时 24 小时
                },
            )
            logger.info(
                "成功连接到 RabbitMQ 服务器: host=%s port=%s vhost=%s user=%s queue=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
            )
            return True
        except AMQPConnectionError as e:
            logger.error(
                "连接 RabbitMQ 失败: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s error=%r",
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
                "连接时发生未知错误: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                self.host,
                self.port,
                self.virtual_host,
                self.username,
                self.queue_name,
                type(e).__name__,
            )
            return False

    def start_consume(self, callback: Callable[[Any, Any, Any, bytes], None]) -> None:
        """
        开始消费消息

        原理
        ----
            使用 pika 的 basic_consume 方法注册回调函数，然后调用 start_consuming
            进入阻塞循环，持续接收并处理消息。

        参数 Quality of Service（QoS）
        ----
            prefetch_count=1：每次只预取 1 条消息，必须等前一条 ack 后才推送下一条
            这样可以确保即使有多个任务同时运行，消息也不会被重复投递

        Parameters
        ----------
        callback : callable
            消息回调函数，签名为 callback(channel, method, properties, body)
            - channel: pika.Channel 通道对象
            - method: pika.spec.Basic.Deliver 消息传递信息（包含 delivery_tag）
            - properties: pika.spec.BasicProperties 消息属性
            - body: bytes 消息内容（JSON 格式的 bytes）
        """
        # 设置 QoS（服务质量），prefetch_count 表示每次确认的最大消息数
        self.channel.basic_qos(prefetch_count=self.prefetch_count)
        # 注册消费者，auto_ack=False 表示手动确认
        self.channel.basic_consume(
            queue=self.queue_name, on_message_callback=callback, auto_ack=False
        )
        # 开始阻塞式消费，持续调用回调函数直到 stop_consume() 被调用
        self.channel.start_consuming()

    def consume_forever(self, callback: Callable[[Any, Any, Any, bytes], None]) -> None:
        """
        持续消费消息，断线后自动重连

        工作机制
        ----
            1. 如果连接失败，等待 retry_delay 秒后重试
            2. 如果消费中断（非用户主动停止），关闭连接并等待重试
            3. 支持 KeyboardInterrupt（Ctrl+C）优雅退出
            4. _stopped 标志控制是否继续循环

        Parameters
        ----------
        callback : callable
            消息回调函数
        """
        self._stopped = False
        while not self._stopped:
            # 尝试连接，如果失败则等待重试
            if not self.connect():
                if self._stopped:
                    break
                logger.warning(
                    "监听器连接失败，%s 秒后重试: queue=%s",
                    self.retry_delay,
                    self.queue_name,
                )
                time.sleep(self.retry_delay)
                continue
            try:
                # 开始消费（阻塞）
                self.start_consume(callback)
            except KeyboardInterrupt:
                # 用户按下 Ctrl+C，优雅退出
                self._stopped = True
                raise
            except Exception as e:
                if not self._stopped:
                    # 消费中断（可能是网络问题），记录日志准备重连
                    logger.error(
                        "监听器消费中断，准备重连: host=%s port=%s vhost=%s user=%s queue=%s error_type=%s",
                        self.host,
                        self.port,
                        self.virtual_host,
                        self.username,
                        self.queue_name,
                        type(e).__name__,
                    )
            finally:
                # 关闭连接
                self.close()
            # 非主动停止时，等待后重试
            if not self._stopped:
                time.sleep(self.retry_delay)

    def stop_consume(self) -> None:
        """
        停止消费消息

        设置 _stopped 标志为 True，然后调用 channel.stop_consuming() 退出消费循环
        """
        self._stopped = True
        if self.channel:
            self.channel.stop_consuming()

    def close(self) -> None:
        """
        关闭 RabbitMQ 连接

        安全关闭连接，处理可能的异常情况
        """
        if self.connection and not self.connection.is_closed:
            try:
                self.connection.close()
            except ConnectionWrongStateError:
                # 连接已处于关闭流程，忽略重复关闭
                logger.warning("RabbitMQ 连接已处于关闭流程，忽略重复关闭")
            except Exception as e:
                logger.warning("关闭 RabbitMQ 监听连接时发生错误: %s", e)
        self.channel = None
