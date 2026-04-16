"""
OSS 客户端模块

提供 OSS Bucket 实例获取功能。
"""

import logging

import oss2

from neuracle.config.env import get_aliyun_config
from neuracle.storage.oss.auth import get_assume_role

logger = logging.getLogger(__name__)


def get_bucket() -> oss2.Bucket:
    """获取 OSS Bucket 实例

    原理：
        通过 STS 临时凭证创建 OSS Bucket 实例，用于后续的文件操作。
        每次调用都会获取新的 STS Token，避免 Token 过期。

    Returns
    -------
    oss2.Bucket
        OSS Bucket 实例
    """
    assume_role = get_assume_role()
    aliyun_config = get_aliyun_config()
    auth = oss2.StsAuth(
        assume_role.body.credentials.access_key_id,
        assume_role.body.credentials.access_key_secret,
        assume_role.body.credentials.security_token,
    )
    bucket = oss2.Bucket(
        auth,
        aliyun_config["oss_endpoint"],
        aliyun_config["bucket_name"],
    )
    return bucket
