"""
阿里云 STS 认证模块

提供阿里云 STS 临时凭证获取功能，用于 OSS 访问授权。
"""

import logging
from typing import Final

from alibabacloud_sts20150401 import models as sts_models
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_tea_openapi import models as open_api_models

from neuracle.config.env import get_aliyun_config
from neuracle.utils import (
    DEFAULT_STS_ROLE_SESSION_NAME,
    DEFAULT_STS_TOKEN_DURATION_SECONDES,
)

logger = logging.getLogger(__name__)


def get_assume_role(
    duration_seconds: int = DEFAULT_STS_TOKEN_DURATION_SECONDES,
    role_session_name: str = DEFAULT_STS_ROLE_SESSION_NAME,
    show_progress: bool = False,
) -> sts_models.AssumeRoleResponse:
    """获取阿里云STS临时授权

    原理：
        通过阿里云 STS 服务获取临时访问凭证（AccessKeyId、AccessKeySecret、SecurityToken）。
        凭证有效期默认20分钟，角色会话名称用于阿里云日志统计。

    Parameters
    ----------
    duration_seconds : int, optional
        授权有效时间, by default 3600
    role_session_name : str, optional
        授权名称，用于阿里云日志统计, by default "simnibs_session"
    show_progress : bool, optional
        是否打印凭证信息, by default False

    Returns
    -------
    sts_models.AssumeRoleResponse
        授权角色信息，包含 access_key_id, access_key_secret, security_token
    """
    aliyun_config = get_aliyun_config()
    config = open_api_models.Config(
        access_key_id=aliyun_config["access_key_id"],
        access_key_secret=aliyun_config["access_key_secret"],
        endpoint=aliyun_config["sts_endpoint"],
    )
    client = StsClient(config)
    assume_role_request = sts_models.AssumeRoleRequest(
        duration_seconds=duration_seconds,
        role_arn=aliyun_config["sts_role_arn"],
        role_session_name=role_session_name,
    )
    assume_role = client.assume_role(assume_role_request)
    if show_progress:
        logger.info("access_key_id: %s", assume_role.body.credentials.access_key_id)
        logger.info(
            "access_key_secret: %s", assume_role.body.credentials.access_key_secret
        )
        logger.info("security_token: %s", assume_role.body.credentials.security_token)
    return assume_role
