"""
Neuracle 日志模块

提供统一的日志配置和管理功能，支持多级别文件日志输出和自动回滚。

使用方式：
    import logging
    from neuracle.logger import setup_logging
    setup_logging()  # 使用默认路径

    # 或指定日志目录
    setup_logging(log_dir='/path/to/logs')

    # 之后在任何地方使用标准 logging
    logger = logging.getLogger('neuracle')
    logger.info("message")

日志文件：
    - debug.log:   记录所有级别日志（包含 DEBUG）
    - info.log:    记录 INFO 及以上级别
    - warning.log: 记录 WARNING 及以上级别
    - error.log:   记录 ERROR 及以上级别

模块结构：
    - formatters: 日志格式化相关
    - handlers:   日志处理器相关
    - setup:      日志配置函数
"""

from neuracle.logger.setup import setup_logging

__all__ = ["setup_logging"]
