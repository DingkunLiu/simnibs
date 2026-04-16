"""
OSS 存储模块

提供阿里云 OSS 的上传、下载、认证等功能。

示例
----
>>> from neuracle.storage.oss import get_bucket, upload_file_to_oss
>>> bucket = get_bucket()
>>> upload_file_to_oss(bucket, Path("local.txt"), "oss/key.txt")
"""

from neuracle.storage.oss.auth import get_assume_role
from neuracle.storage.oss.client import get_bucket
from neuracle.storage.oss.downloader import (
    download_bytes_from_oss,
    download_file_from_oss,
    download_folder_from_oss,
    download_input_file,
    get_file_keys_from_oss,
)
from neuracle.storage.oss.uploader import (
    build_storage_key,
    upload_bytes_to_oss,
    upload_file_to_oss,
    upload_folder_to_oss,
    upload_model_outputs,
    upload_task_result,
)

__all__ = [
    "get_assume_role",
    "get_bucket",
    "get_file_keys_from_oss",
    "download_file_from_oss",
    "download_folder_from_oss",
    "upload_file_to_oss",
    "upload_folder_to_oss",
    "upload_bytes_to_oss",
    "download_bytes_from_oss",
    "download_input_file",
    "build_storage_key",
    "upload_model_outputs",
    "upload_task_result",
]
