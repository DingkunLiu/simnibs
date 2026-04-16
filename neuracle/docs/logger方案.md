# 日志系统改造方案

## 需求分析

1. 创建通用的 neuracle 级别日志系统（不局限于 rabbitmq）
2. 将 logger 改为模块级别（非类内部）
3. 增加 logger 保存到文件功能
4. 日志路径可配置，默认为 `neuracle/log/`
5. 四个级别各自保存到不同文件：
   - `debug.log` - DEBUG 级别
   - `info.log` - INFO 级别
   - `warning.log` - WARNING 级别
   - `error.log` - ERROR 级别
6. 自动回滚：每个文件最大 1MB，最多保留 20 个备份文件

## 实现方案

### 1. 创建日志配置模块 `neuracle/logger/`

使用 Python 标准库的 `logging.handlers.RotatingFileHandler` 实现自动回滚功能。
模块拆分清晰，职责明确：
- `formatters.py`: 日志格式化器
- `handlers.py`: 日志处理器
- `setup.py`: 日志配置函数

### 2. 日志配置策略

| 级别    | 文件名      | 过滤级别       |
| ------- | ----------- | -------------- |
| DEBUG   | debug.log   | DEBUG 及以上   |
| INFO    | info.log    | INFO 及以上    |
| WARNING | warning.log | WARNING 及以上 |
| ERROR   | error.log   | ERROR 及以上   |

### 3. 日志格式

```
%(asctime)s - %(processName)s - %(threadName)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s
```

## 使用方法

```python
import logging
from neuracle.logger import setup_logging

# 方式一：使用默认配置
setup_logging()

# 方式二：手动指定日志目录
setup_logging(log_dir='/path/to/logs')

# 在任何模块中使用
logger = logging.getLogger('neuracle.mymodule')
logger.info("message: %s", var)
```

## 设计原理

使用 Python 标准 logging 模块的单例机制：
1. 调用 `setup_logging()` 配置日志系统
2. 可手动调用 `setup_logging(log_dir=xxx)` 指定日志目录
3. 所有以 'neuracle' 开头的子 logger 自动继承配置
4. 使用标准的 `logging.getLogger()` 获取 logger 实例
5. 不会创建重复的文件处理器

## 模块说明

### formatters.py - 日志格式化器

提供日志格式定义和格式化器创建函数。

主要函数：
- `get_formatter()`: 获取标准日志格式化器

### handlers.py - 日志处理器

提供各种日志处理器，用于将日志输出到不同目标。

主要函数：
- `create_level_handler()`: 创建指定级别的文件处理器（支持自动回滚）
- `create_console_handler()`: 创建控制台处理器（使用 Utf8StreamHandler 避免 Windows 编码问题）

### setup.py - 日志配置

提供日志系统配置函数，设置日志记录器和处理器。

主要函数：
- `_configure_logger()`: 配置指定名称的 logger
- `setup_logging()`: 配置 neuracle 日志系统（主入口）
