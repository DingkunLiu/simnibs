"""
本地路径管理模块

提供任务目录路径的规范化和解析功能。
"""

import logging
import os
import re
import shutil
from pathlib import Path

from neuracle.utils.constants import (
    BUILT_IN_DIR_PATH,
    BUILT_IN_DTI_FILE_PATH,
    DATA_ROOT,
)

logger = logging.getLogger(__name__)


def normalize_dir_path(dir_path: str) -> str:
    """
    规范化相对目录路径，禁止绝对路径和父级跳转。

    Parameters
    ----------
    dir_path : str
        原始目录路径

    Returns
    -------
    str
        规范化后的目录路径

    Raises
    ------
    ValueError
        当路径为绝对路径或包含父级跳转 (..) 时抛出
    """
    normalized = dir_path.replace("\\", "/").strip().strip("/")
    path_obj = Path(normalized)
    if path_obj.is_absolute() or ".." in path_obj.parts:
        logger.error("非法路径访问: %s", dir_path)
        raise ValueError(f"非法 dir_path: {dir_path}")
    return normalized


def get_subject_dir(dir_path: str) -> Path:
    """
    获取 subject 目录的完整路径。

    Parameters
    ----------
    dir_path : str
        subject 目录名（如 m2m_ernie）

    Returns
    -------
    Path
        完整的 subject 目录路径
    """
    return DATA_ROOT / Path(normalize_dir_path(dir_path))


def get_task_output_dir(
    dir_path: str,
    suffix: str,
    task_id: str | None = None,
) -> Path:
    """获取任务输出目录的完整路径

    原理：
        根据 subject 目录名、任务类型后缀和可选的任务 ID 生成输出目录路径。
        格式：DATA_ROOT/{dir_path}_{suffix}_{task_id}/

    Parameters
    ----------
    dir_path : str
        subject 目录名
    suffix : str
        任务类型后缀（如 TI_simulation, TI_optimization）
    task_id : str, optional
        任务 ID

    Returns
    -------
    Path
        任务输出目录路径
    """
    normalized = normalize_dir_path(dir_path)
    if task_id:
        return DATA_ROOT / f"{normalized}_{suffix}_{task_id}"
    return DATA_ROOT / f"{normalized}_{suffix}"


def get_model_mesh_path(dir_path: str) -> Path:
    """获取头模网格文件路径

    原理：
        返回 subject 目录下的 {subid}.msh 文件路径，这是 TI 仿真的核心输入。
        文件名根据 dir_path (m2m_xxx) 动态提取 subid。

    Parameters
    ----------
    dir_path : str
        subject 目录名 (m2m_xxx)

    Returns
    -------
    Path
        {subid}.msh 文件路径
    """

    subid = re.search("m2m_(.+)", dir_path).group(1)
    return get_subject_dir(dir_path) / f"{subid}.msh"


def resolve_local_dti_path(
    dir_path: str,
    dti_file_path: str | None,
) -> str | None:
    """解析本地 DTI 文件路径

    原理：
        对于内置头模，DTI 文件路径是固定的；对于其他头模，需要外部传入。

    Parameters
    ----------
    dir_path : str
        subject 目录名
    dti_file_path : str, optional
        DTI 文件路径

    Returns
    -------
    str or None
        本地 DTI 文件完整路径，仅对内置头模有效
    """
    if dir_path == BUILT_IN_DIR_PATH:
        dti_file_path = BUILT_IN_DTI_FILE_PATH
        logger.info("使用内置头模，DTI使用内置路径: %s", dti_file_path)
        return str(get_subject_dir(dir_path) / dti_file_path)
    return None


def ensure_data_root() -> None:
    """确保数据根目录存在，不存在则创建"""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def reset_task_output_dir(output_dir: str) -> None:
    """删除任务输出目录，避免重跑时被残留结果污染

    Parameters
    ----------
    output_dir : str
        任务输出目录路径
    """
    if os.path.isdir(output_dir):
        logger.info("清理旧任务输出目录: %s", output_dir)
        shutil.rmtree(output_dir)
