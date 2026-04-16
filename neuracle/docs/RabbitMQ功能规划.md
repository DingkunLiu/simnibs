# RabbitMQ 功能规划与实现

本文档记录 RabbitMQ 消费机制的设计决策和最终实现状态。

## 一、消息确认（ACK）模型

### 1.1 整体原则

确认语义为：

- 主消费线程：只负责"接收消息、轻量校验、启动工作线程"
- 工作线程：负责执行实际任务
- 消费连接线程：负责真正调用 `basic_ack`

特别说明：

- `pika.BlockingConnection` / `channel` 不是随便在哪个线程都可以直接安全调用的
- 如果任务在线程池或后台线程里跑完后要确认消息，**不直接在工作线程里调用 `channel.basic_ack(...)`**
- 通过 `connection.add_callback_threadsafe(...)` 把 ack 操作投递回 RabbitMQ 消费连接所在的线程执行

### 1.2 推荐流程

#### A. 正常路径

1. `handle_message()` 收到消息
2. 在消费线程中完成：
   - `json.loads(body)`
   - `task_id` / `msg_type` 提取
   - 参数校验
   - 参数对象构造
3. 如果上述步骤都成功：
   - 创建工作线程
   - 将 `channel` / `delivery_tag` / `task_id` / `msg_type` 等必要上下文交给工作线程
4. `handle_message()` 立即返回，但**此时不 ack**
5. 工作线程执行实际任务
6. 任务成功完成后：
   - 通过 `connection.add_callback_threadsafe(...)`
   - 在消费线程里执行 `channel.basic_ack(delivery_tag=...)`

#### B. 异常路径

以下情况应视为"消息已经处理完毕，但处理结果是失败"，因此仍然 ack：

1. 消息 JSON 解析失败
2. `type` 非法
3. 参数校验失败
4. 参数对象构造失败
5. 工作线程启动失败
6. 工作线程内部任务执行抛异常并已发出失败进度

策略不是"异常就重新入队"，而是：

- 失败结果通过进度消息返回后端
- RabbitMQ 侧仍 ack，避免同一坏消息无限重试

## 二、prefetch_count 的要求

在监听器中显式设置：

```python
channel.basic_qos(prefetch_count=1)
```

目标语义是：

- 当前这条消息没有被 ack 之前
- RabbitMQ 不再向该 consumer 投递下一条消息

这样对项目的意义：

1. 保证单实例服务同一时刻只占用一条未确认消息
2. 避免在前一个任务还没完成时又收到下一个任务
3. 和"完成后 ack"的设计天然匹配
4. 降低内存、CPU、网格文件 IO 的并发压力

## 三、ACK 时机矩阵

| 场景 | 是否 ack | 说明 |
|---|---|---|
| JSON 解析失败 | 是 | 坏消息不应反复重投 |
| `msg_type` 非法 | 是 | 业务无效消息，直接确认并回报失败 |
| 参数校验失败 | 是 | 当前项目走失败回传，不做队列重试 |
| 参数构造失败 | 是 | 同上 |
| 工作线程创建失败 | 是 | 任务未能进入执行态，但消息已处理为失败 |
| 工作线程内任务成功 | 是 | 正常完成后确认 |
| 工作线程内任务异常 | 是 | 失败进度已回传后确认 |

不推荐把这些异常分支改成 `basic_nack(requeue=True)`，否则容易造成：

- 同一坏消息无限重试
- 后端持续收到重复失败
- 单机服务被错误任务卡死

## 四、工作线程与发送线程模型

当前项目已经有独立的 `RabbitMQPublisher`：

- 主任务线程通过 `message_queue.put(...)` 推送进度
- `RabbitMQPublisher` 再独立连接 RabbitMQ 发送到后端队列

这个模型和"完成后 ack"是互补的：

1. 消费线程不负责发送进度
2. 工作线程也不直接持有发送连接
3. 发送与消费解耦
4. ack 与进度回传可以分别管理

因此架构保持：

- 独立发送线程不变
- 消费线程只负责接收和连接生命周期
- 工作线程只负责执行任务和触发最终 ack 请求

## 五、最终架构

```text
RabbitMQ Server
    |
    v
RabbitMQConsumer / BlockingConnection / consume thread
    |
    |-- 解析消息
    |-- 参数校验
    |-- 启动工作线程
    |-- 不立即 ack
    |
    v
Worker Thread
    |
    |-- 执行长任务
    |-- 发送进度到本地 Queue
    |-- 成功/失败后调用 schedule_ack()
    |
    v
connection.add_callback_threadsafe(ack_callback)
    |
    v
consume thread 中执行 basic_ack(delivery_tag)

RabbitMQPublisher
    |
    v
RabbitMQ Server (后端进度队列)
```

同时监听器设置：

```text
basic_qos(prefetch_count=1)
```

## 六、ACK 验证机制

### 6.1 ack_test 任务类型

为验证 ACK 时序正确性，实现了 `ack_test` 任务类型：

- 位于 [`neuracle/rabbitmq/demo/rabbitmq_ack_timing_demo.py`](../../neuracle/rabbitmq/demo/rabbitmq_ack_timing_demo.py)
- 在指定秒数内执行 `sleep`，模拟长任务
- 在关键时间点输出日志：
  - `worker_started` - 工作线程开始
  - `worker_finished` - 工作线程结束
  - `ack_scheduled` - ack 请求已投递
  - `ack_sent` - ack 实际发送

### 6.2 预期日志顺序

正确实现时应看到：

```text
[10:00:00] received task-1
[10:00:00] worker_started task-1
[10:00:30] worker_finished task-1
[10:00:30] ack_scheduled task-1
[10:00:30] ack_sent task-1
```

如果看到：

```text
[10:00:00] received task-1
[10:00:00] ack_sent task-1
[10:00:30] worker_finished task-1
```

就说明仍然是提前 ack。

### 6.3 验证步骤

1. 启动 RabbitMQ 服务端和 management 界面
2. 启动本项目的监听服务
3. 确保监听器已设置 `prefetch_count=1`
4. 运行 `rabbitmq_ack_timing_demo.py`
5. 连续发送两条长任务消息
6. 同时观察：
   - 服务端日志
   - RabbitMQ 管理界面的 `Unacked`

### 6.4 预期现象

如果实现正确，应当看到：

1. 第一条消息开始执行后，队列 `Unacked = 1`
2. 在第一条任务完成前，第二条消息不会开始执行
3. 第一条任务完成后，才出现 `ack_sent`
4. 第一条 `ack_sent` 之后，第二条消息才开始进入执行态

## 七、关键设计决策

1. **完成后才 ack**：任务执行完成（包括成功和失败）后才执行 ACK
2. **线程安全 ACK**：通过 `add_callback_threadsafe` 回到消费连接线程执行
3. **prefetch_count=1**：确保未确认前不接收下一条消息
4. **失败也 ack**：避免坏消息无限重试，失败通过进度消息回传
5. **独立发送线程**：进度发送不阻塞消费线程
