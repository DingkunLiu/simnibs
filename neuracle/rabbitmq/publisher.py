"""
RabbitMQ 消息发布者模块

提供专门的发送线程，用于从队列中获取消息并发送。

该模块是服务架构中的"发送组件"，独立运行在后台线程中。
工作流程：
1. 从共享的 message_queue（Python Queue）中取消息
2. 为每个消息创建 RabbitMQSender 实例并发送
3. 发送完成后关闭连接，准备处理下一条消息

设计目的
-------
- 将消息发送与业务处理解耦
- 避免在任务处理线程中直接发送消息导致的阻塞
- 实现进度更新的异步发送
"""

import logging
import threading
from queue import Queue

from neuracle.rabbitmq.sender import RabbitMQSender

logger = logging.getLogger(__name__)


class RabbitMQPublisher(threading.Thread):
    """
    专门用于发送消息的线程

    继承自 threading.Thread，作为 daemon 线程运行。
    从 message_queue 中取出消息，调用 RabbitMQSender 发送到 RabbitMQ。

    线程生命周期
    ----
    1. run() 被调用 -> 启动线程
    2. 循环从 message_queue 取消息并发送
    3. stop() 被调用 -> 设置 running=False，线程退出

    与 RabbitMQSender 的关系
    ----
    - RabbitMQPublisher 管理线程生命周期和消息队列
    - RabbitMQSender 负责实际的发送操作
    - 每条消息都创建新的 RabbitMQSender 实例
    """

    def __init__(
        self,
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
    ):
        """
        初始化发送线程

        Parameters
        ----------
        host : str
            RabbitMQ 服务器地址
        port : int
            RabbitMQ 服务器端口
        queue_name : str
            发送队列名称（simnibs_to_backend）
        message_queue : Queue
            Python Queue，存放待发送的消息（从工作线程放入，发送线程取出）
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
        super().__init__(daemon=True)  # daemon=True 表示主进程退出时线程自动终止
        self.host = host
        self.port = port
        self.queue_name = queue_name
        self.message_queue = message_queue
        self.username = username
        self.password = password
        self.virtual_host = virtual_host
        self.heartbeat = heartbeat
        self.blocked_connection_timeout = blocked_connection_timeout
        self.socket_timeout = socket_timeout
        self.connection_attempts = connection_attempts
        self.retry_delay = retry_delay
        self.running = True  # 控制线程是否继续运行
        self.sender: RabbitMQSender | None = None

    def run(self) -> None:
        """
        线程运行逻辑（在线程启动时自动调用）

        工作流程
        ----
        1. 进入运行循环
        2. 从 message_queue 阻塞等待消息（queue.get()）
        3. 收到消息后，创建 RabbitMQSender 发送
        4. 发送完成后关闭连接
        5. 调用 task_done() 通知 Queue 消息已处理
        6. 若收到 None 消息或 running=False，退出循环
        """
        logger.info("发送线程启动")
        while self.running:
            try:
                # 从队列获取消息（阻塞直到有消息或线程停止）
                message = self.message_queue.get()
                # 若线程已停止，放弃当前消息并退出
                if not self.running:
                    self.message_queue.task_done()
                    break

                # 为当前消息创建发送器并发送
                self.sender = RabbitMQSender(
                    host=self.host,
                    port=self.port,
                    queue_name=self.queue_name,
                    username=self.username,
                    password=self.password,
                    virtual_host=self.virtual_host,
                    heartbeat=self.heartbeat,
                    blocked_connection_timeout=self.blocked_connection_timeout,
                    socket_timeout=self.socket_timeout,
                    connection_attempts=self.connection_attempts,
                    retry_delay=self.retry_delay,
                )

                # 连接并发送
                if not self.sender.connect():
                    logger.error("发送器连接失败，当前消息发送失败")
                else:
                    self.sender.send_message(message)
            except Exception as e:
                if self.running:
                    logger.error("发送线程处理消息时发生错误: %s", e)
            finally:
                # 关闭连接并清理
                if self.sender:
                    self.sender.close()
                    self.sender = None
                if self.running:
                    self.message_queue.task_done()  # 通知 Queue 消息已处理完毕

        # 线程退出前的清理
        if self.sender:
            self.sender.close()
        logger.info("发送线程退出")

    def stop(self) -> None:
        """
        停止发送线程

        设置 running=False，并向队列放入 None 唤醒阻塞的 get() 调用
        """
        self.running = False
        # 放入 None 唤醒阻塞的 queue.get()
        self.message_queue.put(None)
