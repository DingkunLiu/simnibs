"""
OSS 上传模块

提供文件和数据上传到 OSS 的功能。
"""

import logging
import re
from pathlib import Path
from typing import List

import oss2

from neuracle.storage.oss.client import get_bucket

logger = logging.getLogger(__name__)


def upload_file_to_oss(
    bucket: oss2.Bucket,
    local_path: Path,
    oss_key: str,
) -> None:
    """上传单个本地文件到 OSS

    原理：
        将本地文件内容上传到 OSS 指定路径。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    local_path : Path
        本地文件路径
    oss_key : str
        OSS 中的目标路径（包含文件名）

    Raises
    ------
    FileNotFoundError
        本地文件不存在时抛出
    oss2.exceptions.OssError
        OSS 操作失败时抛出
    Exception
        其他上传错误时抛出
    """
    if not local_path.is_file():
        logger.error("本地文件不存在: %s", local_path)
        raise FileNotFoundError(f"本地文件不存在: {local_path}")
    try:
        logger.info("正在上传文件: %s -> %s", local_path, oss_key)
        bucket.put_object_from_file(oss_key, str(local_path))
        logger.info("文件上传成功: %s", oss_key)
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
        raise
    except Exception as e:
        logger.error("上传文件失败: %s [%s]", e, oss_key)
        raise


def upload_folder_to_oss(
    bucket: oss2.Bucket,
    local_dir: Path,
    oss_prefix: str,
) -> List[str]:
    """递归上传本地目录到 OSS 前缀

    原理：
        遍历本地目录下的所有文件，保持目录结构上传到 OSS。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    local_dir : Path
        本地目录路径
    oss_prefix : str
        OSS 目标前缀路径

    Returns
    -------
    List[str]
        上传成功的文件 OSS key 列表

    Raises
    ------
    FileNotFoundError
        本地目录不存在时抛出
    """
    if not local_dir.is_dir():
        logger.error("本地目录不存在: %s", local_dir)
        raise FileNotFoundError(f"本地目录不存在: {local_dir}")

    uploaded_keys: List[str] = []
    normalized_prefix = oss_prefix.strip("/")
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(local_dir).as_posix()
        oss_key = f"{normalized_prefix}/{rel_path}" if normalized_prefix else rel_path
        upload_file_to_oss(bucket, file_path, oss_key)
        uploaded_keys.append(oss_key)

    return uploaded_keys


def upload_bytes_to_oss(
    bucket: oss2.Bucket,
    data: bytes,
    oss_key: str,
) -> None:
    """上传字节数据到 OSS

    原理：
        将内存中的字节数据直接上传到 OSS，支持任意格式内容。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    data : bytes
        要上传的字节数据
    oss_key : str
        OSS 中的目标路径（包含文件名）

    Raises
    ------
    oss2.exceptions.OssError
        OSS 操作失败时抛出
    Exception
        其他上传错误时抛出
    """
    try:
        logger.info("正在上传文件: %s", oss_key)
        bucket.put_object(oss_key, data)
        logger.info("文件上传成功: %s", oss_key)
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
        raise
    except Exception as e:
        logger.error("上传文件失败: %s [%s]", e, oss_key)
        raise


def upload_model_outputs(
    dir_path: str,
    subject_dir: Path,
    normalized: str,
    subid: str,
) -> dict[str, str]:
    """上传模型输出到 OSS

    原理：
        将头模生成任务的输出文件上传到 OSS，包括电极帽定位、表面重建、MNI变换等结果。

    注意：每次上传前重新获取 STS Token，避免 Token 过期导致上传失败。

    Parameters
    ----------
    dir_path : str
        头模目录名
    subject_dir : Path
        本地 subject 目录路径
    normalized : str
        OSS 根路径前缀
    subid : str
        Subject ID，用于构造动态文件名

    Returns
    -------
    dict[str, str]
        上传文件映射，key 为本地文件名，value 为 OSS 路径
    """
    if not subid:
        subid = re.search("m2m_(.+)", dir_path).group(1)

    bucket = get_bucket()
    uploaded: dict[str, str] = {}
    folder_mappings = {
        f"{normalized}/eeg_positions": subject_dir / "eeg_positions",
        f"{normalized}/label_prep": subject_dir / "label_prep",
        f"{normalized}/surfaces": subject_dir / "surfaces",
        f"{normalized}/toMNI": subject_dir / "toMNI",
    }
    for oss_prefix, local_dir in folder_mappings.items():
        if local_dir.is_dir():
            logger.info("上传目录到 OSS: %s -> %s", local_dir, oss_prefix)
            upload_folder_to_oss(bucket, local_dir, build_storage_key(oss_prefix))
            uploaded[local_dir.name] = f"{oss_prefix}/"

    file_mappings = {
        f"{normalized}/{subid}.msh": subject_dir / f"{subid}.msh",
        f"{normalized}/{subid}.msh.opt": subject_dir / f"{subid}.msh.opt",
        f"{normalized}/T2_reg.nii.gz": subject_dir / "T2_reg.nii.gz",
    }
    for oss_key, local_file in file_mappings.items():
        if local_file.is_file():
            logger.info("上传文件到 OSS: %s -> %s", local_file, oss_key)
            upload_file_to_oss(bucket, local_file, build_storage_key(oss_key))
            uploaded[local_file.name] = oss_key

    return uploaded


def upload_task_result(
    local_file: Path,
    oss_key: str,
) -> str:
    """上传任务结果到 OSS

    原理：
        将任务生成的最终结果文件上传到 OSS。

    注意：每次上传前重新获取 STS Token，避免 Token 过期导致上传失败。

    Parameters
    ----------
    local_file : Path
        本地结果文件路径
    oss_key : str
        OSS 目标路径

    Returns
    -------
    str
        上传后的 OSS 路径
    """
    bucket = get_bucket()
    logger.info("上传任务结果到 OSS: %s -> %s", local_file, oss_key)
    upload_file_to_oss(bucket, local_file, build_storage_key(oss_key))
    return oss_key


def build_storage_key(path: str) -> str:
    """将逻辑 OSS key 映射到真实存储 key

    原理：
        统一处理 OSS 路径格式，去除多余的前导斜杠，保持存储键一致性。

    Parameters
    ----------
    path : str
        原始 OSS 路径

    Returns
    -------
    str
        标准化后的 OSS 存储 key
    """
    logical_key = path.strip("/")
    return logical_key
