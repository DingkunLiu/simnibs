"""
日志处理器模块

提供各种日志处理器，用于将日志输出到不同的目标。
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Final


class Utf8StreamHandler(logging.StreamHandler):
    """
    UTF-8 编码的控制台处理器

    原理：
        标准 StreamHandler 在 Windows 下默认使用 GBK 编码写入 stderr，
        会导致中文等非 ASCII 字符显示为乱码。该类覆写 emit() 方法，
        确保输出使用 UTF-8 编码。

    用法：
        handler = Utf8StreamHandler()
        handler.setLevel(logging.INFO)
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            # 写入前确保使用 UTF-8 编码
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


# 日志文件配置
_MAX_BYTES: Final[int] = 1 * 1024 * 1024  # 1MB，单个日志文件最大大小
_BACKUP_COUNT: Final[int] = 20  # 保留的备份文件数量


def create_level_handler(
    level: int, filename: str, log_dir: str
) -> RotatingFileHandler:
    """
    创建指定级别的文件处理器

    原理：
        使用 RotatingFileHandler 实现日志文件自动回滚。当日志文件达到
        指定大小时，会自动关闭当前文件，并将其重命名为带数字后缀的备份文件，
        然后创建一个新的日志文件继续写入。备份文件数量超过限制时，最旧的
        备份文件会被删除。

        RotatingFileHandler 是线程安全的，适用于多进程环境。

    Args:
        level: 日志级别（logging.DEBUG, logging.INFO 等）
        filename: 日志文件名，例如 "debug.log"
        log_dir: 日志目录路径

    Returns:
        RotatingFileHandler: 配置好的文件处理器

    Raises:
        IOError: 当日志目录不存在且无法创建时
    """
    log_path = os.path.join(log_dir, filename)
    handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setLevel(level)
    return handler


def create_console_handler() -> Utf8StreamHandler:
    """
    创建控制台处理器

    原理：
        使用 StreamHandler 输出到标准错误流（stderr），支持彩色输出（在某些终端中）。
        使用 Utf8StreamHandler 避免 Windows GBK 编码问题，确保中文正常输出。

        控制台默认设置为 INFO 级别，避免过多调试信息干扰。

    Returns:
        logging.StreamHandler: 配置好的控制台处理器
    """
    console_handler = Utf8StreamHandler()
    console_handler.setLevel(logging.INFO)
    return console_handler
