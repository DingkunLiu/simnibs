"""
CHARM Pipeline 步骤脚本包

将 SimNIBS charm 分割流程拆分为独立的步骤脚本。
每个模块对应分割流程的一个步骤：

步骤 1 - prepare_t1 : T1 图像准备
步骤 2 - prepare_t2 : T2 图像配准与准备
步骤 3 - denoise : 输入图像降噪
步骤 4 - init_atlas : Atlas 初始仿射配准与颈部校正
步骤 5 - segment : 体积与表面分割
步骤 6 - create_surfaces : 皮层表面重建
步骤 7 - mesh : 四面体网格生成

用法：
    from neuracle.charm import prepare_t1, prepare_t2, denoise_inputs
    from neuracle.charm import init_atlas, run_segmentation, create_surfaces
    from neuracle.charm import create_mesh
"""

from .create_surfaces import create_surfaces
from .denoise import denoise_inputs
from .init_atlas import init_atlas
from .mesh import create_mesh_step as create_mesh
from .prepare_t1 import prepare_t1
from .prepare_t2 import prepare_t2
from .segment import run_segmentation

__all__ = [
    "prepare_t1",
    "prepare_t2",
    "denoise_inputs",
    "init_atlas",
    "run_segmentation",
    "create_surfaces",
    "create_mesh",
]
