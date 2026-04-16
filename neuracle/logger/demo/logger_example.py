"""
Logger 使用示例

演示如何使用标准 logging 库记录日志。
"""

import logging

from neuracle.logger import setup_logging
from neuracle.utils.constants import PROJECT_ROOT

logger = logging.getLogger("neuracle.demo")


def demo_basic_logging():
    """演示基本日志功能"""
    logger.debug("这是一条 DEBUG 级别的日志")
    logger.info("这是一条 INFO 级别的日志")
    logger.warning("这是一条 WARNING 级别的日志")
    logger.error("这是一条 ERROR 级别的日志")


def demo_custom_logger():
    """演示自定义 logger"""
    # 获取自定义名称的 logger
    custom_logger = logging.getLogger("neuracle.custom")
    custom_logger.info("使用自定义 logger 记录日志")


def demo_logging_with_exception():
    """演示记录异常信息"""
    try:
        # 模拟一个错误
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.error("发生除零错误: %s", e, exc_info=True)


def demo_logging_in_loop():
    """演示循环中记录日志"""
    for i in range(5):
        logger.info("处理第 %s 条数据", i + 1)
    logger.info("所有数据处理完成")


def main():
    """主函数"""
    setup_logging(str(PROJECT_ROOT / "log" / "logger_example"))
    separator = "=" * 50
    logger.info(separator)
    logger.info("Logger 示例程序启动")
    logger.info(separator)
    # 演示基本日志功能
    logger.info("\n--- 基本日志功能 ---")
    demo_basic_logging()
    # 演示自定义 logger
    logger.info("\n--- 自定义 Logger ---")
    demo_custom_logger()
    # 演示异常记录
    logger.info("\n--- 异常记录 ---")
    demo_logging_with_exception()
    # 演示循环日志
    logger.info("\n--- 循环日志 ---")
    demo_logging_in_loop()
    logger.info("\n%s", separator)
    logger.info("Logger 示例程序结束")
    logger.info(separator)
    logger.info("\n请查看 neuracle/log/ 目录下的日志文件：")
    logger.info("  - debug.log   (所有级别)")
    logger.info("  - info.log    (INFO 及以上)")
    logger.info("  - warning.log (WARNING 及以上)")
    logger.info("  - error.log   (ERROR 及以上)")


if __name__ == "__main__":
    main()
