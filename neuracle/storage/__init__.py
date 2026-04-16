"""
Storage 存储模块

提供文件存储和 OSS 操作功能，包括：
- 阿里云 OSS 上传下载
- 本地路径管理
- 任务输出目录管理

依赖
----
- config: 配置模块，提供环境变量读取

示例
----
>>> from neuracle.storage import get_bucket, get_subject_dir
>>> bucket = get_bucket()
>>> subject_dir = get_subject_dir("m2m_ernie")
"""

from neuracle.storage.oss import (
    build_storage_key,
    download_bytes_from_oss,
    download_file_from_oss,
    download_folder_from_oss,
    download_input_file,
    get_assume_role,
    get_bucket,
    get_file_keys_from_oss,
    upload_bytes_to_oss,
    upload_file_to_oss,
    upload_folder_to_oss,
    upload_model_outputs,
    upload_task_result,
)
from neuracle.storage.paths import (
    DATA_ROOT,
    ensure_data_root,
    get_model_mesh_path,
    get_subject_dir,
    get_task_output_dir,
    normalize_dir_path,
    reset_task_output_dir,
    resolve_local_dti_path,
)
from neuracle.utils.constants import (
    BUILT_IN_DIR_PATH,
    BUILT_IN_DTI_FILE_PATH,
    PROJECT_ROOT,
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
    "DATA_ROOT",
    "PROJECT_ROOT",
    "BUILT_IN_DIR_PATH",
    "BUILT_IN_DTI_FILE_PATH",
    "normalize_dir_path",
    "get_subject_dir",
    "get_task_output_dir",
    "get_model_mesh_path",
    "resolve_local_dti_path",
    "ensure_data_root",
    "reset_task_output_dir",
]
