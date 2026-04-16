"""
SimNIBS RabbitMQ 服务主入口

启动方式: python -m neuracle.main

该服务接收来自后端的任务请求，执行 CHARM 头模生成、TI 正向仿真、
TI 逆向仿真等任务，并通过 RabbitMQ 实时上报进度。

架构说明：
    1. 入口点（main.py）负责初始化日志和配置
    2. RabbitMQ 调度器（scheduler.py）负责任务分发和进度上报
    3. 各模块（charm, ti_simulation, ti_optimization）提供任务处理函数

任务类型：
    - model: CHARM 头模生成任务（支持断点续传）
    - forward: TI 正向仿真任务
    - inverse: TI 逆向仿真任务（优化）
    - ack_test: ACK 时机验证任务
"""

import logging
from datetime import datetime

from neuracle.config import get_rabbitmq_config, load_env, mask_rabbitmq_config
from neuracle.logger import setup_logging
from neuracle.rabbitmq.scheduler import run_service
from neuracle.utils.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def main() -> None:
    """主入口函数

    初始化日志、加载配置、启动 RabbitMQ 服务。
    不同的 worker 使用不同的日志目录，防止多进程问题。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    log_dir = PROJECT_ROOT / "log" / f"worker_{timestamp}"
    setup_logging(str(log_dir))
    load_env()
    config = get_rabbitmq_config()
    logger.info("RabbitMQ 配置: %s", mask_rabbitmq_config(config))
    run_service(config)


if __name__ == "__main__":
    main()
