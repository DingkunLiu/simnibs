"""
RabbitMQ 消息数据结构定义

定义所有消息类型的参数和结果数据类（使用 dataclass）。

这些数据类用于：
1. 定义消息参数的结构（ModelParams, ForwardParams, InverseParams, AckTestParams）
2. 定义消息结果的结构（ModelResult, ForwardResult, InverseResult）
3. 定义 ROI 参数结构（ROIParam, AtlasParam, MNIParam）

dataclass 的优势
--------------
- 提供类型提示，方便 IDE 和静态分析
- 相比 dict 更安全，访问不存在的字段会在运行时报错
- 可以设置默认值，简化对象创建
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class AtlasParam:
    """Atlas ROI 参数"""

    name: str
    area: str


@dataclass
class MNIParam:
    """MNI 位置 ROI 参数"""

    center: list[float]
    radius: float


@dataclass
class ROIParam:
    """ROI 参数"""

    atlas_param: AtlasParam | None = None
    mni_param: MNIParam | None = None


@dataclass
class ModelParams:
    """头模生成参数"""

    id: str
    T1_file_path: str
    dir_path: str
    T2_file_path: str | None = None
    DTI_file_path: str | None = None


@dataclass
class ElectrodeWithCurrent:
    """带电流的电极参数"""

    name: str
    current_mA: float


@dataclass
class ForwardParams:
    """正向仿真参数"""

    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    electrode_A: list[ElectrodeWithCurrent]
    electrode_B: list[ElectrodeWithCurrent]
    conductivity_config: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None


@dataclass
class InverseParams:
    """逆向仿真参数"""

    id: str
    dir_path: str
    T1_file_path: str
    montage: str
    current_A: list[float]
    current_B: list[float]
    roi_type: Literal["atlas", "mni_pos"]
    roi_param: ROIParam
    target_threshold: float
    conductivity_config: dict[str, float]
    anisotropy: bool
    DTI_file_path: str | None = None


@dataclass
class AckTestParams:
    """ack 时机验证参数"""

    id: str
    sleep_seconds: float = 30.0


@dataclass
class ModelResult:
    """头模生成结果"""

    msh_file_path: str | None = None


@dataclass
class ForwardResult:
    """正向仿真结果"""

    TI_file: str | None = None


@dataclass
class InverseResult:
    """逆向仿真结果"""

    TI_file: str | None = None
    electrode_A: list[str] | None = None
    electrode_B: list[str] | None = None


@dataclass
class ReturnMessage:
    """返回消息"""

    id: str
    type: Literal["model", "forward", "inverse", "ack_test"]
    progress_rate: int
    message: str | None = None
    result: ModelResult | ForwardResult | InverseResult | None = None
