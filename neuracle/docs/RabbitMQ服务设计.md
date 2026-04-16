# RabbitMQ 服务设计

本文档描述 `python -m neuracle.rabbitmq.scheduler` 当前代码已经实现的服务结构。

## 一、服务目标

服务从 RabbitMQ 接收四类任务：

- `model`：CHARM 头模生成
- `forward`：TI 正向仿真
- `inverse`：TI 逆向优化
- `ack_test`：ACK 时机验证（用于测试 RabbitMQ 确认机制）

服务执行完成后，将进度和结果消息发送回另一个队列。

## 二、实际架构

```text
Backend
  |
  | 任务消息
  v
listen_queue_name
  |
  v
RabbitMQConsumer.consume_forever()
  |
  v
handle_message()
  |
  +--> 为每个任务创建一个 daemon 工作线程
          |
          +--> execute_task()
                  |
                  +--> handle_model_task()
                  +--> handle_forward_task()
                  +--> handle_inverse_task()
                  +--> handle_ack_test_task()
                              |
                              v
                        Queue message_queue
                              |
                              v
                        RabbitMQPublisher.run()
                              |
                              v
                       RabbitMQSender.send_message()
                              |
                              v
                       send_queue_name
```

### 2.1 消息确认（ACK）机制

任务完成后的 ACK 通过以下机制实现：

```text
工作线程 execute_task()
  |
  +--> try: 执行 task_handler
  +--> except: 记录失败原因
  +--> finally: 调用 schedule_ack()
                    |
                    v
            connection.add_callback_threadsafe(ack_callback)
                    |
                    v
            回调在消费线程中执行 channel.basic_ack(delivery_tag)
```

关键点：

- **不在工作线程直接调 ack**，而是通过 `add_callback_threadsafe` 投递回消费连接线程
- `prefetch_count=1` 确保未确认前不投递下一条消息
- 校验/解析失败在 `handle_message()` 中直接 ack，不进入工作线程
- 任务执行中的异常也在 `execute_task()` 的 finally 中统一 schedule_ack

## 三、核心模块

### 3.1 `neuracle/rabbitmq/scheduler.py`

职责：

- 加载 `.env`
- 读取 RabbitMQ 配置
- 启动 `RabbitMQPublisher`
- 启动 `RabbitMQConsumer`
- 解析消息并分发到对应任务处理函数

### 3.2 `RabbitMQConsumer`

位于 [`neuracle/rabbitmq/consumer.py`](../../neuracle/rabbitmq/consumer.py)。

实际行为：

- 使用 `pika.BlockingConnection`
- 声明 quorum 队列（`x-queue-type: quorum`）
- `auto_ack=False`
- `prefetch_count=1`（默认，可配置）
- `consume_forever()` 在连接失败或消费异常时自动重连
- 解析/校验失败时直接在 `handle_message()` 中 `basic_ack`
- 正常任务派发到工作线程后，由 `execute_task()` 的 finally 通过 `schedule_ack()` 延迟 ack

### 3.3 `RabbitMQPublisher`

位于 [`neuracle/rabbitmq/publisher.py`](../../neuracle/rabbitmq/publisher.py)。

实际行为：

- 从 Python `Queue` 中阻塞读取待发送消息
- 每处理一条消息就新建一个 `RabbitMQSender`
- 发送完成后立即关闭连接
- 调用 `stop()` 时向队列塞入 `None` 作为停止信号

这意味着当前实现优先简化连接生命周期，而不是复用长连接。

### 3.4 `RabbitMQSender`

位于 [`neuracle/rabbitmq/sender.py`](../../neuracle/rabbitmq/sender.py)。

实际行为：

- 发送到默认交换机 `""`
- `routing_key = queue_name`
- 消息持久化：`delivery_mode=Persistent`
- 发送失败时尝试重连并重发一次

### 3.5 `neuracle/rabbitmq/task_handlers.py`

负责 ACK 调度的核心函数：

- `schedule_ack()`：将 ack 请求从工作线程投递回消费线程
- `_ack_in_consumer_thread()`：在消费线程中实际执行 `basic_ack`

## 四、服务启动流程

`main()` 的启动顺序如下：

1. `setup_logging()`
2. `load_env()`
3. `get_rabbitmq_config()`
4. `run_service(config)`
5. `run_service()` 内部先启动 `RabbitMQPublisher`
6. 再创建 `RabbitMQConsumer`
7. 使用 `partial(handle_message, message_queue=message_queue)` 注册回调
8. `listener.consume_forever(callback)` 进入常驻监听

## 五、消息处理流程

### 5.1 回调入口

`handle_message(channel, method, properties, body, message_queue)` 的主要逻辑：

1. `json.loads(body)`
2. 读取 `id`、`type`、`params`
3. 按类型调用校验器
4. 将 `params` 转换为 dataclass
5. 创建后台线程执行 `execute_task(...)`
6. **不立即 ack**（ACK 由工作线程 `execute_task()` 的 finally 统一处理）

### 5.2 校验与转换

校验器：

- `validate_model_params()`
- `validate_forward_params()`
- `validate_inverse_params()`
- `validate_ack_test_params()`

转换器：

- `dict_to_model_params()`
- `dict_to_forward_params()`
- `dict_to_inverse_params()`
- `dict_to_ack_test_params()`

### 5.3 异常处理

两层异常处理都会调用 `send_progress(..., progress_rate=0, message=str(e))`：

- `handle_message()` 负责解析和校验阶段异常（直接 ack）
- `execute_task()` 负责具体任务执行异常（通过 `schedule_ack()` ack）

因此外部看到的失败消息统一为：

- `progress_rate = 0`
- `message = 异常文本`
- `result = null`

## 六、四类任务的真实执行逻辑

### 6.1 `model`

处理函数：`handle_model_task()`

执行顺序：

1. `prepare_t1(dir_path, T1_file_path)`
2. 如果存在 `T2_file_path`
3. `prepare_t2(dir_path, T2_file_path)`
4. 若失败则再次调用 `prepare_t2(..., force_sform=True)`
5. `denoise_inputs(dir_path)`
6. `init_atlas(dir_path)`
7. `run_segmentation(dir_path)`
8. `create_surfaces(dir_path)`
9. `create_mesh(dir_path)`
10. 返回 `msh_file_path = dir_path/model.msh`

说明：

- `DTI_file_path` 不参与 `model` 任务执行
- 未传 `T2_file_path` 时不会发送 20% 进度

### 6.2 `forward`

处理函数：`handle_forward_task()`

执行前准备：

- 输出目录固定为 `Dir_path/TI_simulation/{task_id}`
- 若目录已存在，先删除旧目录
- `anisotropy=True` 时传入 `anisotropy_type="vn"`，否则为 `"scalar"`
- `montage` 通过 `find_montage_file()` 解析为 CSV 路径

执行顺序：

1. `setup_session(...)`
2. `setup_electrode_pair1(...)`
3. `setup_electrode_pair2(...)`
4. `run_tdcs_simulation(...)`
5. `calculate_ti(...)`
6. `export_ti_to_nifti(...)`
7. 返回 `{"TI_file": ti_nifti_path}`

说明：

- `electrode_A/B` 现在是 `ElectrodeWithCurrent` 对象数组，从对象中提取 `name` 和 `current_mA`
- `conductivity_config` 会被 `cond_dict_to_list()` 转成 SimNIBS 所需列表
- `N_WORKERS` 当前固定为 `8`

### 6.3 `inverse`

处理函数：`handle_inverse_task()`

执行前准备：

- 输出目录固定为 `Dir_path/TI_optimization/{task_id}`
- 若目录已存在，先删除旧目录
- `montage` 同样通过 `find_montage_file()` 解析

执行顺序：

1. 根据 `roi_type` 生成 ROI 参数
2. `init_optimization(...)`
3. `setup_goal(..., goal="focality", focality_threshold=[target_threshold, NON_ROI_THRESHOLD])`
4. `setup_electrodes_and_roi(...)`
5. `run_optimization(...)`
6. `get_electrode_mapping(output_dir)`
7. `optimize_export_mz3(output_dir, surface_type="central")`
8. 返回 `TI_file`、`electrode_A`、`electrode_B`

当前实现限制：

- 目标函数固定为 `focality`
- `current_A`、`current_B` 仅参与请求校验，不直接传给优化器
- 若未传 `electrode_pair1_center` 等可选字段，则使用 `ti_optimization` 模块中的默认值

### 6.4 `ack_test`

处理函数：`handle_ack_test_task()`

用于验证 RabbitMQ ACK 机制是否在任务完成后才执行确认：

1. 发送进度 0（任务开始）
2. 按指定秒数 `sleep`
3. 发送进度 100（任务完成）

此任务不执行任何实际仿真计算，仅用于验证 ACK 时序。

## 七、进度上报

统一入口：`send_progress(message_queue, task_id, msg_type, progress_rate, message, result)`

实现方式：

1. 通过 `build_progress_message()` 构造字典
2. 放入 `message_queue`
3. `RabbitMQPublisher` 读取并发送

### 7.1 `model` 进度

| 进度 | 文案 |
|---|---|
| 0 | 任务开始 |
| 10 | 已完成: T1 图像准备与格式转换 |
| 20 | 已完成: T2 图像配准与准备 |
| 35 | 已完成: 输入图像降噪 |
| 50 | 已完成: Atlas 初始仿射配准与颈部校正 |
| 70 | 已完成: 体积与表面分割 |
| 85 | 已完成: 皮层表面重建 |
| 100 | 已完成: 四面体网格生成 |

### 7.2 `forward` 进度

| 进度 | 文案 |
|---|---|
| 0 | 任务开始 |
| 10 | 已完成: 配置会话参数 |
| 20 | 已完成: 配置第一个电极对 |
| 35 | 已完成: 配置第二个电极对 |
| 70 | 已完成: TDCS 仿真计算 |
| 85 | 已完成: TI 场计算 |
| 95 | 已完成: 导出 NIfTI 格式 |
| 100 | 已完成: 仿真完成 |

### 7.3 `inverse` 进度

| 进度 | 文案 |
|---|---|
| 0 | 任务开始 |
| 10 | 已完成: 初始化优化结构 |
| 20 | 已完成: 配置目标函数 |
| 35 | 已完成: 配置电极对和 ROI |
| 85 | 已完成: 优化算法执行 |
| 90 | 已完成: 获取电极映射结果 |
| 95 | 已完成: 导出 NIfTI 格式 |
| 100 | 已完成: 优化完成 |

## 八、配置约定

配置文件来自 `neuracle/config/.env`，由 `load_env()` 加载。

关键环境变量：

| 变量名 | 说明 |
|---|---|
| `RABBITMQ_HOST` | 服务地址 |
| `RABBITMQ_PORT` | 服务端口 |
| `RABBITMQ_USERNAME` | 用户名 |
| `RABBITMQ_PASSWORD` | 密码 |
| `RABBITMQ_VHOST` | 虚拟主机 |
| `RABBITMQ_HEARTBEAT` | 心跳 |
| `RABBITMQ_BLOCKED_CONNECTION_TIMEOUT` | 阻塞连接超时 |
| `RABBITMQ_SOCKET_TIMEOUT` | Socket 超时 |
| `RABBITMQ_CONNECTION_ATTEMPTS` | 连接重试次数 |
| `RABBITMQ_RETRY_DELAY` | 重试间隔秒数 |
| `RABBITMQ_LISTEN_QUEUE_NAME` | 服务监听队列 |
| `RABBITMQ_SEND_QUEUE_NAME` | 服务发送队列 |

## 九、示例与测试

现有联调示例位于：

- [`neuracle/rabbitmq/demo/backend_side_demo.py`](../../neuracle/rabbitmq/demo/backend_side_demo.py)
- [`neuracle/rabbitmq/demo/simnibs_side_demo.py`](../../neuracle/rabbitmq/demo/simnibs_side_demo.py)
- [`neuracle/rabbitmq/demo/rabbitmq_example.py`](../../neuracle/rabbitmq/demo/rabbitmq_example.py)
- [`neuracle/rabbitmq/demo/rabbitmq_ack_timing_demo.py`](../../neuracle/rabbitmq/demo/rabbitmq_ack_timing_demo.py)

`rabbitmq_ack_timing_demo.py` 用于验证 ACK 时机是否符合预期（任务完成后才 ack）。
