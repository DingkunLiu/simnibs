"""
SimNIBS mesh validation V2 入口脚本。
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from neuracle.mesh_validation.mesh_validation_report import aggregate_report
from neuracle.mesh_validation.mesh_validation_schema import (
    SCHEMA_VERSION,
    ensure_v2_schema,
    initialize_work_root,
    read_json,
    write_json,
)
from neuracle.mesh_validation.mesh_validation_stages import (
    build_mesh_variants,
    run_forward_validation,
    run_inverse_validation,
    run_replay_validation,
)
from neuracle.mesh_validation.mesh_validation_workspace import (
    PRESET_CONFIGS,
    PRESET_ORDER,
    SIMNIBS_TISSUE_NAME_TO_TAG,
    SubjectConfig,
)
from simnibs import SIMNIBSDIR

LOGGER = logging.getLogger("mesh_validation")
REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_logging(work_root: Path) -> None:
    """
    配置日志输出。

    Parameters
    ----------
    work_root : Path
        工作根目录。

    Returns
    -------
    None
    """
    work_root.mkdir(parents=True, exist_ok=True)
    log_file = work_root / "logs" / "mesh_validation.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns
    -------
    argparse.Namespace
        命令行参数。
    """
    parser = argparse.ArgumentParser(description="SimNIBS mesh validation pipeline V2")
    parser.add_argument("--manifest", required=True, help="manifest JSON 路径")
    parser.add_argument("--work-root", help="覆盖 manifest 中的 work_root")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["mesh", "forward", "inverse", "replay", "report"],
        choices=["mesh", "forward", "inverse", "replay", "report"],
        help="执行阶段列表",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(PRESET_ORDER),
        choices=list(PRESET_ORDER),
        help="执行的 preset 列表",
    )
    parser.add_argument("--subject-ids", nargs="*", help="筛选 subject id")
    parser.add_argument("--forward-cases", nargs="*", help="筛选 forward case")
    parser.add_argument("--inverse-cases", nargs="*", help="筛选 inverse case")
    parser.add_argument("--preset-workers", type=int, default=1, help="preset 并行进程数上限")
    parser.add_argument("--debug-mesh", action="store_true", help="保留 mesh 调试输出")
    parser.add_argument("--check-only", action="store_true", help="仅检查并生成 V2 元数据与 preset ini，不执行后续阶段")
    return parser.parse_args()


def validate_parallelism(preset_workers: int) -> int:
    """
    校验 preset 并发配置。

    Parameters
    ----------
    preset_workers : int
        preset 并行进程数。

    Returns
    -------
    int
        校验后的并行进程数。
    """
    if preset_workers < 1:
        raise ValueError("--preset-workers 必须大于等于 1")
    return preset_workers


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """
    读取 manifest。

    Parameters
    ----------
    manifest_path : Path
        manifest 路径。

    Returns
    -------
    dict[str, Any]
        manifest 内容。
    """
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.setdefault("subjects", [])
    manifest.setdefault("forward_cases", [])
    manifest.setdefault("inverse_cases", [])
    return manifest


def select_subjects(manifest: dict[str, Any], selected_ids: list[str] | None) -> list[SubjectConfig]:
    """
    选择执行的 subject。

    Parameters
    ----------
    manifest : dict[str, Any]
        manifest 内容。
    selected_ids : list[str] or None
        过滤后的 subject id 列表。

    Returns
    -------
    list[SubjectConfig]
        subject 配置列表。
    """
    allowed = set(selected_ids or [])
    rows = []
    for item in manifest["subjects"]:
        subject = SubjectConfig(item["id"], item["m2m_dir"], item["reference_t1"])
        if allowed and subject.id not in allowed:
            continue
        rows.append(subject)
    return rows


def select_cases(cases: list[dict[str, Any]], selected_names: list[str] | None) -> list[dict[str, Any]]:
    """
    选择执行的 case。

    Parameters
    ----------
    cases : list[dict[str, Any]]
        候选 case 列表。
    selected_names : list[str] or None
        过滤后的 case 名称列表。

    Returns
    -------
    list[dict[str, Any]]
        过滤后的 case 列表。
    """
    allowed = set(selected_names or [])
    if not allowed:
        return cases
    return [case for case in cases if case["name"] in allowed]


def prepare_mesh_presets(work_root: Path) -> dict[str, Path]:
    """
    生成 preset ini 文件。

    Parameters
    ----------
    work_root : Path
        工作根目录。

    Returns
    -------
    dict[str, Path]
        各 preset 对应的 ini 路径。
    """
    preset_dir = work_root / "_settings"
    preset_dir.mkdir(parents=True, exist_ok=True)
    base_ini = Path(SIMNIBSDIR) / "charm.ini"
    if not base_ini.exists():
        base_ini = REPO_ROOT / "simnibs" / "charm.ini"
    generated: dict[str, Path] = {}
    for preset_name, preset_values in PRESET_CONFIGS.items():
        config = configparser.ConfigParser()
        config.read(base_ini, encoding="utf-8")
        normalized_elem_sizes = {}
        for tissue_name, values in preset_values["elem_sizes"].items():
            normalized_key = SIMNIBS_TISSUE_NAME_TO_TAG.get(tissue_name, tissue_name)
            normalized_elem_sizes[normalized_key] = values
        config["mesh"]["elem_sizes"] = json.dumps(normalized_elem_sizes, ensure_ascii=False)
        config["mesh"]["skin_facet_size"] = json.dumps(preset_values["skin_facet_size"])
        if "facet_distances" in preset_values:
            config["mesh"]["facet_distances"] = json.dumps(preset_values["facet_distances"], ensure_ascii=False)
        ini_path = preset_dir / f"{preset_name}.ini"
        with ini_path.open("w", encoding="utf-8") as handle:
            config.write(handle)
        generated[preset_name] = ini_path
    return generated


def ensure_manifest_paths(subjects: list[SubjectConfig], work_root: Path) -> None:
    """
    校验 manifest 中的路径。

    Parameters
    ----------
    subjects : list[SubjectConfig]
        subject 列表。
    work_root : Path
        工作根目录。

    Returns
    -------
    None
    """
    work_root.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        if not Path(subject.m2m_dir).exists():
            raise FileNotFoundError(f"subject 路径不存在: {subject.m2m_dir}")
        if not Path(subject.reference_t1).exists():
            raise FileNotFoundError(f"reference_t1 不存在: {subject.reference_t1}")


def build_manifest_snapshot(manifest: dict[str, Any], work_root: Path) -> dict[str, Any]:
    """
    构造 V2 manifest 快照。

    Parameters
    ----------
    manifest : dict[str, Any]
        原始 manifest。
    work_root : Path
        工作根目录。

    Returns
    -------
    dict[str, Any]
        manifest 快照。
    """
    snapshot = json.loads(json.dumps(manifest))
    snapshot["work_root"] = str(work_root)
    snapshot["schema_version"] = SCHEMA_VERSION
    return snapshot


def ensure_work_root_initialized(work_root: Path, manifest_snapshot: dict[str, Any]) -> None:
    """
    初始化或校验 V2 work_root。

    Parameters
    ----------
    work_root : Path
        工作根目录。
    manifest_snapshot : dict[str, Any]
        manifest 快照。

    Returns
    -------
    None
    """
    schema_path = work_root / "schema.json"
    if not work_root.exists():
        initialize_work_root(work_root, manifest_snapshot)
        return
    if schema_path.exists():
        ensure_v2_schema(work_root)
        write_json(work_root / "manifest.snapshot.json", manifest_snapshot)
        return
    existing_items = list(work_root.iterdir())
    if existing_items:
        raise RuntimeError(f"work_root 不是 V2 schema，请先迁移旧结果: {work_root}")
    initialize_work_root(work_root, manifest_snapshot)


def build_command_string(argv: list[str]) -> str:
    """
    生成命令行字符串。

    Parameters
    ----------
    argv : list[str]
        argv 列表。

    Returns
    -------
    str
        命令行字符串。
    """
    return subprocess.list2cmdline(argv)


def main() -> None:
    """
    程序入口。

    Returns
    -------
    None
    """
    args = parse_args()
    command = build_command_string(sys.argv)
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    work_root = Path(args.work_root or manifest.get("work_root") or (manifest_path.parent / "work")).resolve()
    manifest_snapshot = build_manifest_snapshot(manifest, work_root)
    ensure_work_root_initialized(work_root, manifest_snapshot)
    configure_logging(work_root)
    LOGGER.info("使用 V2 schema_version=%s", SCHEMA_VERSION)
    preset_workers = validate_parallelism(args.preset_workers)
    subjects = select_subjects(manifest, args.subject_ids)
    forward_cases = select_cases(manifest["forward_cases"], args.forward_cases)
    inverse_cases = select_cases(manifest["inverse_cases"], args.inverse_cases)
    preset_ini_paths = prepare_mesh_presets(work_root)
    LOGGER.info("已生成 preset ini: %s", ", ".join(sorted(preset_ini_paths)))
    if args.check_only:
        LOGGER.info("check-only 模式结束，不执行后续阶段")
        return
    ensure_manifest_paths(subjects, work_root)
    if "mesh" in args.phases:
        build_mesh_variants(subjects, args.presets, preset_ini_paths, work_root, args.debug_mesh, preset_workers, command)
    if "forward" in args.phases:
        run_forward_validation(subjects, args.presets, forward_cases, work_root, preset_ini_paths, args.debug_mesh, preset_workers, command)
    if "inverse" in args.phases:
        run_inverse_validation(subjects, args.presets, inverse_cases, work_root, preset_ini_paths, args.debug_mesh, preset_workers, command)
    if "replay" in args.phases:
        run_replay_validation(subjects, args.presets, inverse_cases, work_root, preset_ini_paths, args.debug_mesh, preset_workers, command)
    if "report" in args.phases:
        summary = aggregate_report(work_root=work_root, selected_presets=args.presets)
        LOGGER.info("结果汇总: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
