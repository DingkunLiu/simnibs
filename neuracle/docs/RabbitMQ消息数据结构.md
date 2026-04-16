# RabbitMQ 消息数据结构

本文档描述 `neuracle/rabbitmq/scheduler.py` 当前实现实际使用的消息结构。

## 一、队列与连接

### 1.1 队列方向

| 配置项 | 方向 | 用途 |
|---|---|---|
| `listen_queue_name` | Backend -> SimNIBS | 发送任务请求 |
| `send_queue_name` | SimNIBS -> Backend | 返回进度与结果 |

默认运行方式下：

- SimNIBS 服务监听 `listen_queue_name`
- SimNIBS 服务通过 `RabbitMQPublisher` 发送到 `send_queue_name`

### 1.2 RabbitMQ 连接参数

配置由 [`neuracle/config/env.py`](../../neuracle/config/env.py) 的 `get_rabbitmq_config()` 提供：

| 字段 | 默认值 |
|---|---|
| `host` | `localhost` |
| `port` | `5672` |
| `username` | `guest` |
| `password` | `guest` |
| `virtual_host` | `/` |
| `heartbeat` | `60` |
| `blocked_connection_timeout` | `300` |
| `socket_timeout` | `10` |
| `connection_attempts` | `5` |
| `retry_delay` | `5` |
| `listen_queue_name` | `""` |
| `send_queue_name` | `""` |

### 1.3 队列声明方式

`RabbitMQConsumer` 和 `RabbitMQSender` 都会声明队列，声明参数固定为：

- `durable=True`
- `arguments={"x-queue-type": "quorum", "x-consumer-timeout": 86400000}`
- 使用默认交换机 `""`
- `routing_key` 直接等于队列名

## 二、顶层消息格式

所有任务请求消息都使用 JSON，对应验证入口为 [`neuracle/rabbitmq/validator.py`](../../neuracle/rabbitmq/validator.py)。

```json
{
  "id": "task_id",
  "type": "model | forward | inverse | ack_test",
  "params": {}
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | 是 | 任务 ID，不能为空 |
| `type` | string | 是 | 支持 `model`、`forward`、`inverse`、`ack_test` |
| `params` | object | 是 | 按任务类型区分 |

## 三、`model` 请求

对应数据类：`ModelParams`

```json
{
  "id": "model_001",
  "type": "model",
  "params": {
    "T1_file_path": "C:/data/subj/T1.nii.gz",
    "T2_file_path": "C:/data/subj/T2.nii.gz",
    "DTI_file_path": "C:/data/subj/DTI_coregT1_tensor.nii.gz",
    "dir_path": "C:/data/subj"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `params.T1_file_path` | string | 是 | T1 文件路径 |
| `params.dir_path` | string | 是 | 输出目录，也是 CHARM 工作目录 |
| `params.T2_file_path` | string | 否 | T2 文件路径 |
| `params.DTI_file_path` | string | 否 | 当前消息结构保留该字段，但 `handle_model_task()` 未使用 |

### 3.1 `model` 完成结果

```json
{
  "id": "model_001",
  "type": "model",
  "progress_rate": 100,
  "message": "已完成: 四面体网格生成",
  "result": {
    "msh_file_path": "C:/data/subj/model.msh"
  }
}
```

注意：

- 结果文件名被代码固定为 `model.msh`
- 路径由 `os.path.join(dir_path, "model.msh")` 生成

## 四、`forward` 请求

对应数据类：`ForwardParams`

```json
{
  "id": "forward_001",
  "type": "forward",
  "params": {
    "dir_path": "C:/data/m2m_ernie",
    "T1_file_path": "C:/data/m2m_ernie/T1.nii.gz",
    "montage": "EEG10-10_UI_Jurak_2007",
    "electrode_A": [
      { "name": "F5", "current_mA": 2.0 },
      { "name": "P5", "current_mA": -2.0 }
    ],
    "electrode_B": [
      { "name": "F6", "current_mA": 1.0 },
      { "name": "P6", "current_mA": -1.0 }
    ],
    "conductivity_config": {
      "White Matter": 0.126,
      "Gray Matter": 0.275,
      "CSF": 1.654,
      "Bone": 0.01,
      "Scalp": 0.465,
      "Eye balls": 0.5,
      "Compact Bone": 0.008,
      "Spongy Bone": 0.025,
      "Blood": 0.6,
      "Muscle": 0.16
    },
    "anisotropy": false
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `params.dir_path` | string | 是 | 头模目录 |
| `params.T1_file_path` | string | 是 | T1 文件路径 |
| `params.montage` | string | 是 | Montage 名称或绝对路径 |
| `params.electrode_A` | list[dict] | 是 | 第一组电极，元素为 `{"name": str, "current_mA": number}`，current_mA 总和必须为 0 |
| `params.electrode_B` | list[dict] | 是 | 第二组电极，元素为 `{"name": str, "current_mA": number}`，current_mA 总和必须为 0 |
| `params.conductivity_config` | dict | 是 | 电导率字典，值必须为数字 |
| `params.anisotropy` | bool | 是 | 是否启用各向异性 |
| `params.DTI_file_path` | string | 否 | 各向异性场景可传 |

### 4.1 `montage` 的实际解析规则

`montage` 通过 `find_montage_file(dir_path, montage)` 解析：

- 如果是绝对路径，直接使用该路径
- 否则依次查找：
  - `Dir_path/eeg_positions/{montage}`
  - `Dir_path/eeg_positions/{montage}.csv`

### 4.2 `forward` 完成结果

```json
{
  "id": "forward_001",
  "type": "forward",
  "progress_rate": 100,
  "message": "已完成: 仿真完成",
  "result": {
    "TI_file": "C:/data/m2m_ernie/TI_simulation/forward_001/TI.nii.gz"
  }
}
```

注意：

- 输出目录固定为 `Dir_path/TI_simulation/{task_id}`
- 开始执行前会删除同名旧目录
- 中间产物包括 `TI.msh`

## 五、`inverse` 请求

对应数据类：`InverseParams`

```json
{
  "id": "inverse_001",
  "type": "inverse",
  "params": {
    "dir_path": "C:/data/m2m_ernie",
    "T1_file_path": "C:/data/m2m_ernie/T1.nii.gz",
    "montage": "EEG10-10_Cutini_2011",
    "current_A": [0.002, -0.002],
    "current_B": [0.001, -0.001],
    "roi_type": "mni_pos",
    "roi_param": {
      "atlas_param": null,
      "mni_param": {
        "center": [-38.5, -22.5, 58.3],
        "radius": 15.0
      }
    },
    "target_threshold": 0.5,
    "conductivity_config": {
      "White Matter": 0.126,
      "Gray Matter": 0.275,
      "CSF": 1.654,
      "Bone": 0.01,
      "Scalp": 0.465,
      "Eye balls": 0.5,
      "Compact Bone": 0.008,
      "Spongy Bone": 0.025,
      "Blood": 0.6,
      "Muscle": 0.16
    },
    "anisotropy": false
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `params.dir_path` | string | 是 | 头模目录 |
| `params.T1_file_path` | string | 是 | T1 文件路径 |
| `params.montage` | string | 是 | EEG 网格文件名称或绝对路径 |
| `params.current_A` | list[float] | 是 | 仅做校验，要求总和为 0 |
| `params.current_B` | list[float] | 是 | 仅做校验，要求总和为 0 |
| `params.roi_type` | string | 是 | `atlas` 或 `mni_pos` |
| `params.roi_param` | dict | 是 | ROI 参数 |
| `params.target_threshold` | number | 是 | 必须 `>= 0` |
| `params.conductivity_config` | dict | 是 | 电导率字典 |
| `params.anisotropy` | bool | 是 | 是否启用各向异性 |
| `params.DTI_file_path` | string | 否 | 各向异性场景可传 |

### 5.1 `roi_param` 结构

`roi_type=atlas` 时：

```json
{
  "roi_type": "atlas",
  "roi_param": {
    "atlas_param": {
      "name": "AAL3",
      "area": "Precentral_L"
    },
    "mni_param": null
  }
}
```

`roi_type=mni_pos` 时：

```json
{
  "roi_type": "mni_pos",
  "roi_param": {
    "atlas_param": null,
    "mni_param": {
      "center": [-38.5, -22.5, 58.3],
      "radius": 15.0
    }
  }
}
```

### 5.2 `inverse` 完成结果

```json
{
  "id": "inverse_001",
  "type": "inverse",
  "progress_rate": 100,
  "message": "已完成: 优化完成",
  "result": {
    "TI_file": "C:/data/m2m_ernie/TI_optimization/inverse_001/result.nii.gz",
    "electrode_A": ["F3", "FC5"],
    "electrode_B": ["P3", "PO7"]
  }
}
```

注意：

- 输出目录固定为 `Dir_path/TI_optimization/{task_id}`
- 开始执行前会删除同名旧目录
- `electrode_A` 和 `electrode_B` 取自 `electrode_mapping.json` 的 `mapped_labels`

## 六、`ack_test` 请求

对应数据类：`AckTestParams`

用于验证 RabbitMQ ACK 时机，不执行任何实际仿真计算。

```json
{
  "id": "ack_test_001",
  "type": "ack_test",
  "params": {
    "sleep_seconds": 10.0
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `params.sleep_seconds` | float | 否 | 模拟长任务持续时间，默认 30.0 秒 |

### 6.1 `ack_test` 完成结果

```json
{
  "id": "ack_test_001",
  "type": "ack_test",
  "progress_rate": 100,
  "message": null,
  "result": {
    "sleep_seconds": 10.0
  }
}
```

## 七、返回消息格式

返回消息由 `build_progress_message()` 构造：

```json
{
  "id": "task_id",
  "type": "model | forward | inverse | ack_test",
  "progress_rate": 0,
  "message": "任务开始",
  "result": null
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | string | 是 | 任务 ID |
| `type` | string | 是 | 任务类型 |
| `progress_rate` | integer | 是 | 进度百分比 |
| `message` | string or null | 否 | 状态或错误信息 |
| `result` | object or null | 否 | 完成时才返回 |

### 7.1 失败消息

当前实现没有单独的 `status` 字段。失败通过以下模式表达：

- `progress_rate = 0`
- `message = 异常信息`
- `result = null`

示例：

```json
{
  "id": "forward_001",
  "type": "forward",
  "progress_rate": 0,
  "message": "montage 文件不存在: EEG10-10_UI_Jurak_2007",
  "result": null
}
```

## 八、进度节点

### 8.1 `model`

| 进度 | 说明 |
|---|---|
| 0 | 任务开始 |
| 10 | 已完成: T1 图像准备与格式转换 |
| 20 | 已完成: T2 图像配准与准备，仅在传入 `T2_file_path` 时发送 |
| 35 | 已完成: 输入图像降噪 |
| 50 | 已完成: Atlas 初始仿射配准与颈部校正 |
| 70 | 已完成: 体积与表面分割 |
| 85 | 已完成: 皮层表面重建 |
| 100 | 已完成: 四面体网格生成 |

### 8.2 `forward`

| 进度 | 说明 |
|---|---|
| 0 | 任务开始 |
| 10 | 已完成: 配置会话参数 |
| 20 | 已完成: 配置第一个电极对 |
| 35 | 已完成: 配置第二个电极对 |
| 70 | 已完成: TDCS 仿真计算 |
| 85 | 已完成: TI 场计算 |
| 95 | 已完成: 导出 NIfTI 格式 |
| 100 | 已完成: 仿真完成 |

### 8.3 `inverse`

| 进度 | 说明 |
|---|---|
| 0 | 任务开始 |
| 10 | 已完成: 初始化优化结构 |
| 20 | 已完成: 配置目标函数 |
| 35 | 已完成: 配置电极对和 ROI |
| 85 | 已完成: 优化算法执行 |
| 90 | 已完成: 获取电极映射结果 |
| 95 | 已完成: 导出 NIfTI 格式 |
| 100 | 已完成: 优化完成 |

## 九、相关代码位置

| 文件 | 作用 |
|---|---|
| [`neuracle/rabbitmq/scheduler.py`](../../neuracle/rabbitmq/scheduler.py) | 服务入口与任务执行 |
| [`neuracle/rabbitmq/schemas.py`](../../neuracle/rabbitmq/schemas.py) | dataclass 定义 |
| [`neuracle/rabbitmq/validator.py`](../../neuracle/rabbitmq/validator.py) | 请求校验 |
| [`neuracle/rabbitmq/message_builder.py`](../../neuracle/rabbitmq/message_builder.py) | 消息构造 |
| [`neuracle/rabbitmq/params.py`](../../neuracle/rabbitmq/params.py) | dict -> dataclass 转换 |
| [`neuracle/rabbitmq/progress.py`](../../neuracle/rabbitmq/progress.py) | 进度枚举定义 |
| [`neuracle/rabbitmq/task_handlers.py`](../../neuracle/rabbitmq/task_handlers.py) | ACK 调度逻辑 |
| [`neuracle/rabbitmq/demo/rabbitmq_ack_timing_demo.py`](../../neuracle/rabbitmq/demo/rabbitmq_ack_timing_demo.py) | ACK 时机验证示例 |
