"""
Atlas 注册表模块

负责构建和读写 atlas registry，包括：
1. 约定 atlas 原始目录、标准化目录和 manifest 目录
2. 生成 atlas registry JSON 文件
3. 加载 atlas registry
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from neuracle.utils.constants import NEURACLE_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)


ATLAS_ROOT = NEURACLE_DIR / "atlas"
ATLAS_SOURCE_DIR = ATLAS_ROOT / "atlas"
ATLAS_STANDARDIZED_DIR = ATLAS_ROOT / "standardized"
ATLAS_MANIFESTS_DIR = ATLAS_ROOT / "manifests"
ATLAS_VALIDATION_DIR = ATLAS_ROOT / "validation"
ATLAS_SCRIPTS_DIR = ATLAS_ROOT / "scripts"
ATLAS_COLOR_TABLE_DIR = ATLAS_SOURCE_DIR / "颜色查找表"
ATLAS_REGISTRY_PATH = ATLAS_MANIFESTS_DIR / "atlas_registry.json"
SIMNIBS_MNI_TEMPLATE = (
    PROJECT_ROOT / "simnibs" / "resources" / "templates" / "MNI152_T1_1mm.nii.gz"
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    读取 CSV 文件并返回行列表。

    Parameters
    ----------
    path : Path
        CSV 文件路径

    Returns
    -------
    list[dict[str, str]]
        CSV 行列表，每行是一个字典
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _slugify(value: str) -> str:
    """
    将字符串转换为安全的文件名 slug。

    Parameters
    ----------
    value : str
        输入字符串

    Returns
    -------
    str
        安全的文件名 slug

    Notes
    -----
    移除非字母数字字符，多个连续下划线合并，首尾下划线去除。
    如果结果为空，则返回 "roi"。
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "roi"


def _to_repo_relative(path: Path) -> str:
    """
    将路径转换为相对于仓库根目录的路径。

    Parameters
    ----------
    path : Path
        输入路径

    Returns
    -------
    str
        相对于仓库根目录的路径字符串

    Notes
    -----
    registry 内统一保存仓库相对路径，避免把本机绝对路径写死进去。
    """
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def _build_area_entries(label_table: Path, roi_dir: Path) -> list[dict[str, Any]]:
    """
    从标签表构建脑区条目列表。

    Parameters
    ----------
    label_table : Path
        标签表 CSV 文件路径
    roi_dir : Path
        ROI 文件目录路径

    Returns
    -------
    list[dict[str, Any]]
        脑区条目列表，包含 index、label_en、label_zh、roi_filename、roi_path

    Notes
    -----
    使用标签索引做前缀，避免不同脑区在 slugify 后发生文件名冲突。
    """
    rows = _read_csv_rows(label_table)
    areas: list[dict[str, Any]] = []
    for row in rows:
        index = int(row["index"])
        if index == 0:
            continue
        label_en = row["label_en"].strip()
        label_zh = row["label_zh"].strip()
        filename = f"{index:04d}_{_slugify(label_en)}.nii.gz"
        areas.append(
            {
                "index": index,
                "label_en": label_en,
                "label_zh": label_zh,
                "roi_filename": filename,
                "roi_path": _to_repo_relative(roi_dir / filename),
            }
        )
    return areas


def build_atlas_registry() -> dict[str, Any]:
    """
    根据现有 atlas 资源构建 registry。

    Returns
    -------
    dict[str, Any]
        包含 schema_version、simnibs_mni_template、atlas_root、
        standardized_root、atlases 的 registry 字典

    Notes
    -----
    扫描 ATLAS_SOURCE_DIR 下的所有 atlas 目录，生成标准化的
    registry 信息，包括 BN、JulichBrainAtlas、DiFuMo 等 atlas。
    """
    registry: dict[str, Any] = {
        "schema_version": 1,
        "simnibs_mni_template": _to_repo_relative(SIMNIBS_MNI_TEMPLATE),
        "atlas_root": _to_repo_relative(ATLAS_ROOT),
        "standardized_root": _to_repo_relative(ATLAS_STANDARDIZED_DIR),
        "atlases": {},
    }

    atlas_specs = [
        {
            "name": "BN_Atlas_246_1mm",
            "raw_atlas": ATLAS_SOURCE_DIR
            / "BN_Atlas_246_1mm"
            / "BN_Atlas_246_1mm.nii.gz",
            "raw_label_table": ATLAS_SOURCE_DIR
            / "BN_Atlas_246_1mm"
            / "BN_Atlas_246_labels_zh.csv",
            "color_table": ATLAS_COLOR_TABLE_DIR / "BN_Atlas_246_labels_zh.csv",
            "space_template": ATLAS_SOURCE_DIR
            / "BN_Atlas_246_1mm"
            / "MNI152_T1_1mm.nii.gz",
            "space_name": "bn_mni152_1mm",
            "is_discrete": True,
            "interpolation": "nearest",
        },
        {
            "name": "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152",
            "raw_atlas": (
                ATLAS_SOURCE_DIR
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152.nii.gz"
            ),
            "raw_label_table": (
                ATLAS_SOURCE_DIR
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv"
            ),
            "color_table": (
                ATLAS_COLOR_TABLE_DIR
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_labels_zh.csv"
            ),
            "space_template": (
                ATLAS_SOURCE_DIR
                / "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
                / "mni_icbm152_t1_tal_nlin_asym_09c.nii"
            ),
            "space_name": "icbm152_09c",
            "is_discrete": True,
            "interpolation": "nearest",
        },
    ]

    for resolution in (64, 128, 256, 512, 1024):
        atlas_name = f"DiFuMo{resolution}"
        difumo_dir = ATLAS_SOURCE_DIR / "DiFuMo" / atlas_name
        atlas_specs.append(
            {
                "name": atlas_name,
                "raw_atlas": difumo_dir / "2mm" / "maps.nii.gz",
                "raw_label_table": difumo_dir
                / f"labels_{resolution}_dictionary_zh.csv",
                "color_table": ATLAS_COLOR_TABLE_DIR
                / f"labels_{resolution}_dictionary_zh.csv",
                "space_template": ATLAS_SOURCE_DIR
                / "DiFuMo"
                / "tpl-MNI152NLin6Asym_res-02_T1w_difumo-grid.nii.gz",
                "space_name": "mni152nlin6asym_difumo_grid",
                "is_discrete": False,
                "interpolation": "linear",
            }
        )

    for spec in atlas_specs:
        standardized_dir = ATLAS_STANDARDIZED_DIR / spec["name"]
        roi_dir = standardized_dir / "rois"
        label_table_target = standardized_dir / "labels_zh.csv"
        atlas_target = standardized_dir / "atlas_simnibs_mni.nii.gz"
        areas = _build_area_entries(Path(spec["raw_label_table"]), roi_dir)

        registry["atlases"][spec["name"]] = {
            "name": spec["name"],
            "raw_atlas": _to_repo_relative(Path(spec["raw_atlas"])),
            "raw_label_table": _to_repo_relative(Path(spec["raw_label_table"])),
            "color_table": _to_repo_relative(Path(spec["color_table"])),
            "space_template": _to_repo_relative(Path(spec["space_template"])),
            "space_name": spec["space_name"],
            "is_discrete": spec["is_discrete"],
            "interpolation": spec["interpolation"],
            "standardized_dir": _to_repo_relative(standardized_dir),
            "standardized_atlas": _to_repo_relative(atlas_target),
            "standardized_label_table": _to_repo_relative(label_table_target),
            "roi_dir": _to_repo_relative(roi_dir),
            "areas": areas,
        }

    return registry


def write_atlas_registry(path: Path = ATLAS_REGISTRY_PATH) -> Path:
    """
    写出 atlas registry JSON 文件。

    Parameters
    ----------
    path : Path, optional
        输出路径，默认为 ATLAS_REGISTRY_PATH

    Returns
    -------
    Path
        输出的 JSON 文件路径
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = build_atlas_registry()
    path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_atlas_registry(path: Path = ATLAS_REGISTRY_PATH) -> dict[str, Any]:
    """
    加载 atlas registry JSON 文件。

    Parameters
    ----------
    path : Path, optional
        registry 文件路径，默认为 ATLAS_REGISTRY_PATH

    Returns
    -------
    dict[str, Any]
        registry 字典

    Notes
    -----
    如果文件不存在，会先调用 write_atlas_registry 生成。
    """
    if not path.exists():
        write_atlas_registry(path)
    return json.loads(path.read_text(encoding="utf-8"))
