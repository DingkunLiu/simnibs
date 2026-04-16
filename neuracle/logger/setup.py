"""
日志配置模块

提供日志系统配置函数，设置日志记录器和处理器。
"""

import logging
import os
from typing import Optional

from neuracle.logger.formatters import get_formatter
from neuracle.logger.handlers import create_console_handler, create_level_handler
from neuracle.utils.constants import PROJECT_ROOT


def _configure_logger(
    logger_name: str, log_dir: str, console_level: int = logging.INFO
) -> logging.Logger:
    """
    配置指定名称的 logger

    原理：
        为指定的 logger 配置处理器。logger 采用单例模式，配置后所有子 logger
        都会继承此配置。通过为不同级别设置不同的文件处理器，可以实现日志分级存储。

        处理器说明：
        - 控制台处理器：输出 INFO 及以上级别到 stderr
        - debug.log：记录所有级别日志
        - info.log：记录 INFO 及以上级别
        - warning.log：记录 WARNING 及以上级别
        - error.log：记录 ERROR 及以上级别

    Args:
        logger_name: logger 名称
        log_dir: 日志目录路径
        console_level: 控制台日志级别，默认为 logging.INFO

    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    logger = logging.getLogger(logger_name)
    # 移除已有的 handlers（避免重复配置）
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG)
    # 获取格式化器
    formatter = get_formatter()
    # 添加控制台处理器
    console_handler = create_console_handler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # 添加各级别文件处理器
    debug_handler = create_level_handler(logging.DEBUG, "debug.log", log_dir)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)
    info_handler = create_level_handler(logging.INFO, "info.log", log_dir)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)
    warning_handler = create_level_handler(logging.WARNING, "warning.log", log_dir)
    warning_handler.setFormatter(formatter)
    logger.addHandler(warning_handler)
    error_handler = create_level_handler(logging.ERROR, "error.log", log_dir)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    return logger


def setup_logging(log_dir: Optional[str] = None) -> None:
    """
    配置 neuracle 日志系统

    原理：
        配置 'neuracle' logger 的处理器。由于 logging 模块的单例机制，
        所有以 'neuracle' 开头的子 logger 都会继承此配置，不会创建重复的处理器。

        simnibs 的 logger 在 simnibs_logger.py 中已有配置（仅控制台），
        此处不再配置，避免多进程文件日志竞争问题。

        日志输出：
        - neuracle: 控制台（INFO）+ 文件（debug/info/warning/error.log）
        - simnibs: 仅控制台

    Args:
        log_dir: 日志目录路径，默认为 neuracle/log/

    用法:
        >>> import logging
        >>> from neuracle.logger import setup_logging
        >>> setup_logging()  # 使用默认路径
        >>>
        >>> # 或指定日志目录
        >>> setup_logging(log_dir='/path/to/logs')
        >>>
        >>> # 在任何模块中使用
        >>> logger = logging.getLogger('neuracle.module')
        >>> logger.info("message")
    """
    # 默认日志目录
    if log_dir is None:
        log_dir = os.path.join(PROJECT_ROOT, "log")
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 配置 neuracle logger
    _configure_logger("neuracle", log_dir)
