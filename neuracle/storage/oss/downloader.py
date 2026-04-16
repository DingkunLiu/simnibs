"""
OSS 下载模块

提供从 OSS 下载文件和数据的功能。
"""

import logging
from pathlib import Path
from typing import List

import oss2

from neuracle.storage.oss.client import get_bucket
from neuracle.storage.oss.uploader import build_storage_key

logger = logging.getLogger(__name__)


def get_file_keys_from_oss(
    bucket: oss2.Bucket,
    oss_prefix: str = "",
    save_path: Path | None = None,
    show_progress: bool = False,
) -> List[str]:
    """获取 OSS Bucket 下对应前缀的所有文件路径

    原理：
        递归遍历指定前缀下的所有文件，返回文件路径列表。可选保存到本地文件。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    oss_prefix : str, optional
        需要查看的 OSS 前缀路径, by default ""
    save_path : Path | None, optional
        保存路径, by default None
    show_progress : bool, optional
        是否打印获取到的文件信息, by default False

    Returns
    -------
    List[str]
        对应前缀的所有文件路径(key)列表
    """
    all_file_keys = []

    def recursive_list_files(prefix: str) -> None:
        """递归列举指定前缀下的所有文件"""
        for obj in oss2.ObjectIterator(bucket, prefix=prefix):
            if obj.is_prefix():
                recursive_list_files(obj.key)
            else:
                if show_progress:
                    logger.info("Listed file: %s", obj.key)
                all_file_keys.append(obj.key)

    recursive_list_files(oss_prefix)

    if save_path:
        with open(save_path, "w", encoding="utf8") as f:
            for path in all_file_keys:
                f.write(path + "\n")
    return all_file_keys


def download_folder_from_oss(
    oss_prefix: str,
    local_dir: Path,
) -> None:
    """下载 OSS 中某个前缀下的所有文件到本地

    原理：
        根据 OSS 前缀获取所有文件列表，逐一下载并保留原有目录结构。
        每次下载前重新获取 STS Token，避免 Token 过期。

    Parameters
    ----------
    oss_prefix : str
        需要下载的 OSS 前缀
    local_dir : Path
        保存的本地路径

    Raises
    ------
    oss2.exceptions.OssError
        OSS 操作失败时抛出
    Exception
        其他下载错误时抛出
    """
    bucket = get_bucket()
    target_file_keys = get_file_keys_from_oss(bucket, oss_prefix)
    for file_key in target_file_keys:
        oss_rel_path = Path(file_key).relative_to(oss_prefix)
        target_path = local_dir / oss_rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("正在下载文件: %s", file_key)
            bucket.get_object_to_file(file_key, target_path)
            logger.info("文件下载成功: %s", file_key)
        except oss2.exceptions.OssError as e:
            logger.error("OSS错误: %s [%s]", e.message, oss_rel_path)
            raise
        except Exception as e:
            logger.error("下载文件失败: %s [%s]", e, oss_rel_path)
            raise


def download_file_from_oss(
    oss_key: str,
    local_path: Path,
) -> None:
    """下载单个 OSS 文件到本地

    原理：
        将 OSS 中的单个文件下载到本地指定路径。
        每次下载前重新获取 STS Token，避免 Token 过期。

    Parameters
    ----------
    oss_key : str
        OSS 中的文件路径
    local_path : Path
        本地保存路径

    Raises
    ------
    oss2.exceptions.OssError
        OSS 操作失败时抛出
    Exception
        其他下载错误时抛出
    """
    bucket = get_bucket()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("正在下载文件: %s -> %s", oss_key, local_path)
        bucket.get_object_to_file(oss_key, str(local_path))
        logger.info("文件下载成功: %s", oss_key)
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
        raise
    except Exception as e:
        logger.error("下载文件失败: %s [%s]", e, oss_key)
        raise


def download_bytes_from_oss(
    bucket: oss2.Bucket,
    oss_key: str,
) -> bytes:
    """从 OSS 下载字节数据到内存

    原理：
        将 OSS 中的文件内容读取为字节数据并返回，不落盘。

    Parameters
    ----------
    bucket : oss2.Bucket
        OSS Bucket 实例
    oss_key : str
        OSS 中的文件路径

    Returns
    -------
    bytes
        文件的字节数据

    Raises
    ------
    oss2.exceptions.OssError
        OSS 操作失败时抛出
    Exception
        其他下载错误时抛出
    """
    try:
        logger.info("正在下载文件: %s", oss_key)
        result = bucket.get_object(oss_key)
        data = result.read()
        logger.info("文件下载成功: %s", oss_key)
        return data
    except oss2.exceptions.OssError as e:
        logger.error("OSS错误: %s [%s]", e.message, oss_key)
        raise
    except Exception as e:
        logger.error("下载文件失败: %s [%s]", e, oss_key)
        raise


def download_input_file(
    oss_key: str,
    local_path: Path,
) -> Path:
    """下载单个输入文件并返回本地路径

    原理：
        将 OSS 中的输入文件下载到本地指定路径，返回本地文件路径。

    Parameters
    ----------
    oss_key : str
        OSS 中的文件路径
    local_path : Path
        本地保存路径

    Returns
    -------
    Path
        本地文件路径

    Raises
    ------
    FileNotFoundError
        下载失败时抛出
    """
    download_file_from_oss(build_storage_key(oss_key), local_path)
    return local_path
