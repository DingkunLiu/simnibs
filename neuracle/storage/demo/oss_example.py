"""
OSS 上传下载示例

演示如何将内存中的二进制内容上传到 OSS，以及从 OSS 下载到内存。
"""

import logging

from neuracle.config.env import load_env
from neuracle.logger import setup_logging
from neuracle.storage.oss import (
    download_bytes_from_oss,
    get_bucket,
    upload_bytes_to_oss,
)
from neuracle.utils.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


def demo_upload_download():
    """演示内存中的二进制数据上传和下载"""
    setup_logging(str(PROJECT_ROOT / "log" / "oss_example"))
    load_env()
    bucket = get_bucket()
    oss_key = "demo/test_content.bin"
    # 构造一段二进制内容（模拟任意数据）
    original_data = b"Hello from SimNIBS OSS Demo!\nThis is binary content in memory."
    # 上传
    upload_bytes_to_oss(bucket, original_data, oss_key)
    # 下载
    downloaded_data = download_bytes_from_oss(bucket, oss_key)
    # 验证
    assert downloaded_data == original_data, "Data mismatch!"
    logger.info("验证成功：上传和下载的数据一致")
    # 打印内容
    logger.info("文件内容: %s", downloaded_data.decode("utf-8"))


if __name__ == "__main__":
    demo_upload_download()
