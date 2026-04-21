"""
SimNIBS mesh validation 入口脚本。
"""

from __future__ import annotations

import argparse
import configparser
import json
import logging
import sys
from pathlib import Path
from typing import Any


def _bootstrap_imports() -> Path:
    """
    项目根目录导入path

    Returns
    -------
    Path
        仓库根目录。
    """
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


REPO_ROOT = _bootstrap_imports()

from neuracle.mesh_validation.mesh_validation_report import aggregate_report
from neuracle.mesh_validation.mesh_validation_stages import (
    build_mesh_variants,
    run_forward_validation,
    run_inverse_validation,
)
from neuracle.mesh_validation.mesh_validation_workspace import (
    PRESET_CONFIGS,
    PRESET_ORDER,
    SIMNIBS_TISSUE_NAME_TO_TAG,
    SubjectConfig,
)
from simnibs import SIMNIBSDIR

LOGGER = logging.getLogger("mesh_validation")


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
    parser = argparse.ArgumentParser(description="SimNIBS mesh validation pipeline")
    parser.add_argument("--manifest", required=True, help="manifest JSON 路径")
    parser.add_argument("--work-root", help="覆盖 manifest 中的 work_root")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["mesh", "forward", "inverse", "report"],
        choices=["mesh", "forward", "inverse", "report"],
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
    parser.add_argument("--check-only", action="store_true", help="仅检查并生成 preset ini，不执行后续阶段")
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


def main() -> None:
    """
    程序入口。

    Returns
    -------
    None
    """
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    work_root = Path(args.work_root or manifest.get("work_root") or (manifest_path.parent / "work")).resolve()
    configure_logging(work_root)
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
    mesh_rows = build_mesh_variants(subjects, args.presets, preset_ini_paths, work_root, args.debug_mesh, preset_workers) if "mesh" in args.phases else []
    forward_rows = run_forward_validation(subjects, args.presets, forward_cases, work_root, preset_ini_paths, args.debug_mesh, preset_workers) if "forward" in args.phases else []
    inverse_rows = run_inverse_validation(subjects, args.presets, inverse_cases, work_root, preset_ini_paths, args.debug_mesh, preset_workers) if "inverse" in args.phases else []
    if "report" in args.phases:
        summary = aggregate_report(
            mesh_rows=mesh_rows,
            forward_rows=forward_rows,
            inverse_rows=inverse_rows,
            work_root=work_root,
            executed_phases=args.phases,
            subjects=subjects,
            inverse_cases=inverse_cases,
            preset_ini_paths=preset_ini_paths,
            debug_mesh=args.debug_mesh,
        )
        LOGGER.info("结果汇总: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
