"""
日志格式化模块

提供日志格式定义和相关工具函数。
"""

import logging
from typing import Final

# 日志格式配置
_LOG_FORMAT: Final[str] = (
    "%(asctime)s - %(processName)s - %(threadName)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def get_formatter(
    fmt: str = _LOG_FORMAT, datefmt: str = _DATE_FORMAT
) -> logging.Formatter:
    """
    获取日志格式化器

    原理：
        创建一个Formatter实例，用于将日志记录格式化为字符串。
        格式化字符串遵循logging模块的标准格式语法。

    Args:
        fmt: 日志格式字符串，默认为 _LOG_FORMAT
            - %(asctime)s: 可读时间戳
            - %(processName)s: 进程名
            - %(threadName)s: 线程名
            - %(name)s: logger名称
            - %(levelname)s: 日志级别
            - %(filename)s: 源文件名
            - %(lineno)d: 源文件行号
            - %(message)s: 日志消息
        datefmt: 日期格式字符串，默认为 _DATE_FORMAT

    Returns:
        logging.Formatter: 配置好的格式化器
    """
    return logging.Formatter(fmt, datefmt)
