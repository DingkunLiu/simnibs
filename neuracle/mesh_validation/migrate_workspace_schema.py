"""
Mesh validation V1 -> V2 workspace 迁移脚本。
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Callable
from collections import defaultdict
from pathlib import Path
from typing import Any

from neuracle.mesh_validation.mesh_validation_schema import (
    RUN_STATE_COMPLETED,
    build_run_payload,
    initialize_work_root,
    mesh_result_path,
    mesh_run_path,
    preset_paths_for,
    read_json,
    replay_result_path,
    replay_run_path,
    write_json,
)
from neuracle.mesh_validation.mesh_validation_schema import (
    forward_result_path,
    forward_run_path,
    inverse_result_path,
    inverse_run_path,
    replay_artifacts_dir,
)

LOGGER = logging.getLogger("mesh_validation_migrate")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns
    -------
    argparse.Namespace
        命令行参数。
    """
    parser = argparse.ArgumentParser(description="Migrate mesh validation workspace from V1 to V2")
    parser.add_argument("--old-work-root", required=True, help="旧 V1 work_root")
    parser.add_argument("--new-work-root", required=True, help="新 V2 work_root")
    return parser.parse_args()


def configure_logging() -> None:
    """
    配置日志。

    Returns
    -------
    None
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", force=True)


def ensure_target_root(new_root: Path) -> None:
    """
    校验目标目录可用于迁移。

    Parameters
    ----------
    new_root : Path
        新 work_root。

    Returns
    -------
    None
    """
    if not new_root.exists():
        return
    if not any(new_root.iterdir()):
        return
    if (new_root / "schema.json").exists():
        return
    if any(new_root.iterdir()):
        raise RuntimeError(f"目标 work_root 非空，拒绝迁移: {new_root}")


def copy_tree_incremental(
    source: Path,
    target: Path,
    should_skip: Callable[[Path], bool] | None = None,
) -> None:
    """
    以增量方式复制目录，已存在文件直接跳过。

    Parameters
    ----------
    source : Path
        源目录。
    target : Path
        目标目录。

    Returns
    -------
    None
    """
    if not source.exists():
        return
    def _copy_dir(current_source: Path, current_target: Path) -> None:
        current_target.mkdir(parents=True, exist_ok=True)
        for item in current_source.iterdir():
            relative = item.relative_to(source)
            if should_skip is not None and should_skip(relative):
                continue
            target_item = current_target / item.name
            if item.is_dir():
                _copy_dir(item, target_item)
                continue
            if target_item.exists():
                continue
            shutil.copy2(item, target_item)
    _copy_dir(source, target)


def is_old_replay_dir(relative: Path) -> bool:
    """
    鍒ゆ柇鐩稿璺緞鏄惁涓烘棫鐗?replay_on_m0 鐩綍鎴栧叾鍐呭銆?

    Parameters
    ----------
    relative : Path
        鐩稿浜?preset 鏍圭洰褰曠殑璺緞銆?

    Returns
    -------
    bool
        鏄惁搴旇璺宠繃銆?
    """
    parts = relative.parts
    return len(parts) >= 4 and parts[0] == "inverse" and parts[2].startswith("seed_") and parts[3] == "replay_on_m0"


def parse_old_seed(seed_name: str) -> int:
    """
    从旧 seed 目录名提取整数种子。

    Parameters
    ----------
    seed_name : str
        旧目录名，如 `seed_7`。

    Returns
    -------
    int
        种子值。
    """
    if not seed_name.startswith("seed_"):
        raise ValueError(f"不支持的旧 seed 目录名: {seed_name}")
    return int(seed_name.split("_", maxsplit=1)[1])


def migrated_run_payload(stage: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """
    构造迁移后的 completed run.json。

    Parameters
    ----------
    stage : str
        阶段名称。
    metadata : dict[str, Any]
        元数据字段。

    Returns
    -------
    dict[str, Any]
        run.json 内容。
    """
    return build_run_payload(stage=stage, command="migrated-from-v1", metadata=metadata, state=RUN_STATE_COMPLETED)


def parse_old_replay_artifact_relative_path(relative: Path) -> tuple[str, str, str, int, Path] | None:
    """
    解析旧版 replay artifact 相对路径。

    Parameters
    ----------
    relative : Path
        相对于 V1 work_root 的路径。

    Returns
    -------
    tuple[str, str, str, int, Path] or None
        `(subject_id, preset, case_name, seed, artifact_tail)`；
        若不是旧版 replay artifact 路径则返回 None。
    """
    parts = relative.parts
    if len(parts) < 7:
        return None
    subject_id, preset, stage_name, case_name, seed_name, replay_dir_name = parts[:6]
    if stage_name != "inverse" or replay_dir_name != "replay_on_m0":
        return None
    return subject_id, preset, case_name, parse_old_seed(seed_name), Path(*parts[6:])


def extract_v1_relative_path(old_root: Path, value: str) -> Path | None:
    """
    从旧版绝对路径中提取相对 work_root 的布局路径。

    Parameters
    ----------
    old_root : Path
        V1 work_root。
    value : str
        待解析的旧版路径字符串。

    Returns
    -------
    Path or None
        形如 `<subject>/<preset>/<stage>/...` 的相对路径；
        若无法识别则返回 None。
    """
    path_value = Path(value)
    try:
        return path_value.relative_to(old_root)
    except ValueError:
        pass
    parts = path_value.parts
    for index in range(len(parts) - 2):
        if parts[index + 2] in {"workspace", "mesh", "forward", "inverse"}:
            return Path(*parts[index:])
    return None


def expected_replay_artifact_path(
    new_root: Path,
    subject_id: str,
    preset: str,
    case_name: str,
    seed: int,
    artifact_tail: Path,
) -> Path:
    """
    生成 V2 replay artifact 目标路径。

    Parameters
    ----------
    new_root : Path
        新版 V2 work_root。
    subject_id : str
        subject 标识。
    preset : str
        preset 名称。
    case_name : str
        case 名称。
    seed : int
        随机种子。
    artifact_tail : Path
        replay_on_m0 下的相对 artifact 路径。

    Returns
    -------
    Path
        目标 artifact 路径。
    """
    return replay_artifacts_dir(preset_paths_for(new_root, subject_id, preset), case_name, seed) / artifact_tail


def ensure_no_collapsed_replay_artifacts(new_root: Path) -> None:
    """
    校验不存在错误聚合到共享目录的 replay artifact。

    Parameters
    ----------
    new_root : Path
        V2 work_root。

    Returns
    -------
    None
    """
    collapsed_root = new_root / "subjects" / "presets" / "presets"
    if not collapsed_root.exists():
        return
    collapsed_entries = [str(path) for path in collapsed_root.rglob("*")]
    if collapsed_entries:
        raise RuntimeError(
            "检测到错误聚合的 replay artifact 目录: "
            f"{collapsed_root}; 示例: {collapsed_entries[:5]}"
        )


def rewrite_migrated_path_value(old_root: Path, new_root: Path, value: Any) -> Any:
    """
    将 V1 结果中的绝对路径改写为 V2 路径。
    Parameters
    ----------
    old_root : Path
        旧 V1 work_root。
    new_root : Path
        新 V2 work_root。
    value : Any
        待改写的值。
    Returns
    -------
    Any
        改写后的值。
    """
    if isinstance(value, dict):
        return {key: rewrite_migrated_path_value(old_root, new_root, item) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite_migrated_path_value(old_root, new_root, item) for item in value]
    if not isinstance(value, str):
        return value
    relative = extract_v1_relative_path(old_root, value)
    if relative is None:
        return value
    relative_parts = relative.parts
    if len(relative_parts) < 3:
        return value
    replay_relative = parse_old_replay_artifact_relative_path(relative)
    if replay_relative is not None:
        subject_id, preset, case_name, seed, artifact_tail = replay_relative
        replay_artifact_path = expected_replay_artifact_path(new_root, subject_id, preset, case_name, seed, artifact_tail)
        if not replay_artifact_path.exists():
            raise FileNotFoundError(
                "缺少迁移后的 replay artifact: "
                f"subject={subject_id}, preset={preset}, case={case_name}, seed={seed}, "
                f"path={replay_artifact_path}"
            )
        return str(replay_artifact_path)
    subject_id, preset = relative_parts[0], relative_parts[1]
    tail = relative_parts[2:]
    new_preset_root = preset_paths_for(new_root, subject_id, preset).preset_root
    if tail[0] in {"workspace", "mesh", "forward"}:
        return str(new_preset_root / Path(*tail))
    if tail[0] != "inverse":
        return value
    return str(new_preset_root / Path(*tail))


def copy_static_roots(old_root: Path, new_root: Path) -> None:
    """
    复制 `_settings` 与 `logs` 等公共目录。

    Parameters
    ----------
    old_root : Path
        旧 work_root。
    new_root : Path
        新 work_root。

    Returns
    -------
    None
    """
    for name in ("_settings", "logs"):
        source = old_root / name
        target = new_root / name
        if source.exists():
            copy_tree_incremental(source, target)


def copy_subject_preset_dirs(old_root: Path, new_root: Path) -> None:
    """
    复制旧的 subject/preset 目录。

    Parameters
    ----------
    old_root : Path
        旧 work_root。
    new_root : Path
        新 work_root。

    Returns
    -------
    None
    """
    reports_dir = old_root / "reports"
    skipped_names = {"_settings", "logs", "reports"}
    for subject_dir in sorted(path for path in old_root.iterdir() if path.is_dir() and path.name not in skipped_names and path != reports_dir):
        for preset_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            target = preset_paths_for(new_root, subject_dir.name, preset_dir.name).preset_root
            copy_tree_incremental(preset_dir, target, should_skip=is_old_replay_dir)


def relocate_replay_dirs(old_root: Path, new_root: Path) -> None:
    """
    将旧 `inverse/.../replay_on_m0` 直接迁移到 V2 replay 目录。

    Parameters
    ----------
    old_root : Path
        旧 V1 work_root。
    new_root : Path
        新 work_root。

    Returns
    -------
    None
    """
    for replay_dir in sorted(old_root.glob("*/*/inverse/*/seed_*/replay_on_m0")):
        relative = replay_dir.relative_to(old_root)
        parts = relative.parts
        if len(parts) != 6 or parts[2] != "inverse" or parts[5] != "replay_on_m0":
            raise ValueError(f"无法解析 replay_on_m0 目录结构: {replay_dir}")
        subject_id = parts[0]
        preset = parts[1]
        case_name = parts[3]
        seed = parse_old_seed(parts[4])
        target_stage_dir = replay_result_path(preset_paths_for(new_root, subject_id, preset), case_name, seed).parent
        target_artifacts_dir = target_stage_dir / "artifacts"
        target_stage_dir.mkdir(parents=True, exist_ok=True)
        copy_tree_incremental(replay_dir, target_artifacts_dir)


def validate_relocated_replay_artifacts(old_root: Path, new_root: Path, inverse_rows: list[dict[str, Any]]) -> None:
    """
    校验 replay artifact 已迁移到正确的 V2 目录。

    Parameters
    ----------
    old_root : Path
        V1 work_root。
    new_root : Path
        V2 work_root。
    inverse_rows : list[dict[str, Any]]
        V1 inverse 报告行。

    Returns
    -------
    None
    """
    ensure_no_collapsed_replay_artifacts(new_root)
    for row in inverse_rows:
        replay_value = row.get("replay_ti_volume_path")
        if not isinstance(replay_value, str):
            continue
        rewritten = rewrite_migrated_path_value(old_root, new_root, replay_value)
        if not Path(rewritten).exists():
            raise FileNotFoundError(f"缺少 replay artifact: {rewritten}")


def validate_replay_result_paths(new_root: Path, inverse_rows: list[dict[str, Any]]) -> None:
    """
    校验 replay result.json 中记录的路径存在且未落入错误共享目录。

    Parameters
    ----------
    new_root : Path
        V2 work_root。
    inverse_rows : list[dict[str, Any]]
        V1 inverse 报告行。

    Returns
    -------
    None
    """
    ensure_no_collapsed_replay_artifacts(new_root)
    for row in inverse_rows:
        subject_id = str(row.get("subject_id"))
        preset = str(row.get("preset"))
        case_name = str(row.get("case_name"))
        seed = int(row.get("seed"))
        result_path = replay_result_path(preset_paths_for(new_root, subject_id, preset), case_name, seed)
        replay_payload = read_json(result_path)
        if replay_payload is None:
            raise FileNotFoundError(f"缺少 replay result.json: {result_path}")
        replay_path_value = replay_payload.get("replay_ti_volume_path")
        if not isinstance(replay_path_value, str):
            raise ValueError(f"replay_ti_volume_path 缺失或类型错误: {result_path}")
        replay_path = Path(replay_path_value)
        if not replay_path.exists():
            raise FileNotFoundError(
                "replay_ti_volume_path 指向不存在的文件: "
                f"subject={subject_id}, preset={preset}, case={case_name}, seed={seed}, path={replay_path}"
            )


def load_required_v1_reports(old_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    读取旧版 reports/*.json。

    Parameters
    ----------
    old_root : Path
        旧 work_root。

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
        mesh、forward、inverse 三类结果。
    """
    reports_dir = old_root / "reports"
    mesh_rows = read_json(reports_dir / "mesh_stats.json", [])
    forward_rows = read_json(reports_dir / "forward_metrics.json", [])
    inverse_rows = read_json(reports_dir / "inverse_metrics.json", [])
    if not mesh_rows and not forward_rows and not inverse_rows:
        raise FileNotFoundError(f"旧版 reports/*.json 不存在或为空: {reports_dir}")
    return mesh_rows, forward_rows, inverse_rows


def write_mesh_results(old_root: Path, new_root: Path, mesh_rows: list[dict[str, Any]]) -> int:
    """
    写入迁移后的 mesh 结果。

    Parameters
    ----------
    new_root : Path
        新 work_root。
    mesh_rows : list[dict[str, Any]]
        旧 mesh 行结果。

    Returns
    -------
    int
        写入数量。
    """
    count = 0
    for row in mesh_rows:
        subject_id = str(row["subject_id"])
        preset = str(row["preset"])
        paths = preset_paths_for(new_root, subject_id, preset)
        migrated = {
            key: rewrite_migrated_path_value(old_root, new_root, value)
            for key, value in row.items()
            if key not in {"scalp_mean_distance_mm", "scalp_hausdorff95_mm"}
        }
        write_json(mesh_result_path(paths), migrated)
        write_json(mesh_run_path(paths), migrated_run_payload("mesh", {"subject_id": subject_id, "preset": preset}))
        count += 1
    return count


def write_forward_results(old_root: Path, new_root: Path, forward_rows: list[dict[str, Any]]) -> int:
    """
    写入迁移后的 forward 结果。

    Parameters
    ----------
    new_root : Path
        新 work_root。
    forward_rows : list[dict[str, Any]]
        旧 forward 行结果。

    Returns
    -------
    int
        写入数量。
    """
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in forward_rows:
        grouped[(str(row.get("subject_id")), str(row.get("preset")), str(row.get("case_name")))].append(row)
    count = 0
    for (subject_id, preset, case_name), rows in grouped.items():
        paths = preset_paths_for(new_root, subject_id, preset)
        first = rows[0]
        if first.get("status") != "ok":
            payload = {
                "subject_id": subject_id,
                "preset": preset,
                "case_name": case_name,
                "status": first.get("status"),
                "failure_stage": first.get("failure_stage"),
                "error": first.get("error"),
                "traceback": first.get("traceback"),
            }
        else:
            roi_metrics = []
            for row in rows:
                roi_metrics.append(
                    {
                        "roi_name": row.get("roi_name"),
                        "roi_mean": row.get("roi_mean"),
                        "roi_median": row.get("roi_median"),
                        "roi_p95": row.get("roi_p95"),
                        "roi_p99": row.get("roi_p99"),
                        "roi_max": row.get("roi_max"),
                    }
                )
            payload = {
                "subject_id": subject_id,
                "preset": preset,
                "case_name": case_name,
                "status": "ok",
                "elapsed_seconds": rewrite_migrated_path_value(old_root, new_root, first.get("elapsed_seconds")),
                "ti_mesh_path": rewrite_migrated_path_value(old_root, new_root, first.get("ti_mesh_path")),
                "ti_volume_path": rewrite_migrated_path_value(old_root, new_root, first.get("ti_volume_path")),
                "final_labels_path": rewrite_migrated_path_value(old_root, new_root, first.get("final_labels_path")),
                "hotspot_value": first.get("hotspot_value"),
                "roi_metrics": roi_metrics,
            }
        write_json(forward_result_path(paths, case_name), payload)
        write_json(forward_run_path(paths, case_name), migrated_run_payload("forward", {"subject_id": subject_id, "preset": preset, "case_name": case_name}))
        count += 1
    return count


def write_inverse_and_replay_results(old_root: Path, new_root: Path, inverse_rows: list[dict[str, Any]]) -> tuple[int, int]:
    """
    写入迁移后的 inverse 与 replay 结果。

    Parameters
    ----------
    new_root : Path
        新 work_root。
    inverse_rows : list[dict[str, Any]]
        旧 inverse 行结果。

    Returns
    -------
    tuple[int, int]
        inverse 与 replay 写入数量。
    """
    inverse_count = 0
    replay_count = 0
    for row in inverse_rows:
        subject_id = str(row.get("subject_id"))
        preset = str(row.get("preset"))
        case_name = str(row.get("case_name"))
        seed = int(row.get("seed"))
        paths = preset_paths_for(new_root, subject_id, preset)
        inverse_payload = {
            "subject_id": subject_id,
            "preset": preset,
            "case_name": case_name,
            "seed": seed,
            "status": row.get("status"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "optimizer": row.get("optimizer"),
            "optimizer_fopt": row.get("optimizer_fopt"),
            "optimizer_n_test": row.get("optimizer_n_test"),
            "optimizer_n_sim": row.get("optimizer_n_sim"),
            "mapped_mesh_path": rewrite_migrated_path_value(old_root, new_root, row.get("mapped_mesh_path")),
            "mapped_ti_volume_path": rewrite_migrated_path_value(old_root, new_root, row.get("mapped_ti_volume_path")),
            "electrode_mapping_path": rewrite_migrated_path_value(old_root, new_root, row.get("electrode_mapping_path")),
            "mapped_labels": row.get("mapped_labels"),
            "mapping_distance_mean_mm": row.get("mapping_distance_mean_mm"),
            "failure_stage": row.get("failure_stage"),
            "error": row.get("error"),
            "traceback": row.get("traceback"),
        }
        replay_payload = {
            "subject_id": subject_id,
            "preset": preset,
            "case_name": case_name,
            "seed": seed,
            "status": row.get("status"),
            "elapsed_seconds": row.get("elapsed_seconds"),
            "replay_ti_volume_path": rewrite_migrated_path_value(old_root, new_root, row.get("replay_ti_volume_path")),
            "replay_roi_mean": row.get("replay_roi_mean"),
            "replay_roi_p999": row.get("replay_roi_p999"),
            "replay_non_roi_mean": row.get("replay_non_roi_mean"),
            "replay_roc": row.get("replay_roc"),
            "replay_goal": row.get("replay_goal"),
            "comparison_status": row.get("comparison_status"),
            "comparison_reason": row.get("comparison_reason"),
            "goal_gap_on_m0": row.get("goal_gap_on_m0"),
            "label_consistent": row.get("label_consistent"),
            "optimized_center_drift_mean_mm": row.get("optimized_center_drift_mean_mm"),
            "optimized_center_drift_max_mm": row.get("optimized_center_drift_max_mm"),
            "mapped_center_drift_mean_mm": row.get("mapped_center_drift_mean_mm"),
            "mapped_center_drift_max_mm": row.get("mapped_center_drift_max_mm"),
            "inverse_pass": row.get("inverse_pass"),
            "failure_stage": row.get("failure_stage"),
            "error": row.get("error"),
            "traceback": row.get("traceback"),
        }
        write_json(inverse_result_path(paths, case_name, seed), inverse_payload)
        write_json(inverse_run_path(paths, case_name, seed), migrated_run_payload("inverse", {"subject_id": subject_id, "preset": preset, "case_name": case_name, "seed": seed}))
        inverse_count += 1
        write_json(replay_result_path(paths, case_name, seed), replay_payload)
        write_json(replay_run_path(paths, case_name, seed), migrated_run_payload("replay", {"subject_id": subject_id, "preset": preset, "case_name": case_name, "seed": seed}))
        replay_count += 1
    return inverse_count, replay_count


def main() -> None:
    """
    程序入口。

    Returns
    -------
    None
    """
    args = parse_args()
    configure_logging()
    old_root = Path(args.old_work_root).resolve()
    new_root = Path(args.new_work_root).resolve()
    if not old_root.exists():
        raise FileNotFoundError(f"旧 work_root 不存在: {old_root}")
    ensure_target_root(new_root)
    initialize_work_root(
        new_root,
        {
            "schema_version": 2,
            "work_root": str(new_root),
            "migrated_from": str(old_root),
        },
    )
    copy_static_roots(old_root, new_root)
    copy_subject_preset_dirs(old_root, new_root)
    relocate_replay_dirs(old_root, new_root)
    mesh_rows, forward_rows, inverse_rows = load_required_v1_reports(old_root)
    validate_relocated_replay_artifacts(old_root, new_root, inverse_rows)
    mesh_count = write_mesh_results(old_root, new_root, mesh_rows)
    forward_count = write_forward_results(old_root, new_root, forward_rows)
    inverse_count, replay_count = write_inverse_and_replay_results(old_root, new_root, inverse_rows)
    validate_replay_result_paths(new_root, inverse_rows)
    summary = {
        "old_work_root": str(old_root),
        "new_work_root": str(new_root),
        "mesh_results": mesh_count,
        "forward_results": forward_count,
        "inverse_results": inverse_count,
        "replay_results": replay_count,
    }
    write_json(new_root / "migration.summary.json", summary)
    LOGGER.info("迁移完成: %s", summary)


if __name__ == "__main__":
    main()
