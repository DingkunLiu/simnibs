"""
Mesh validation V2 schema 与路径工具。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2
RUN_STATE_RUNNING = "running"
RUN_STATE_COMPLETED = "completed"
RUN_STATE_FAILED = "failed"
RUN_STATE_POLL_SECONDS = 10
RUN_STATE_HEARTBEAT_SECONDS = 30
RUN_STATE_STALE_SECONDS = 300


@dataclass(frozen=True)
class SubjectPaths:
    """
    subject 级别路径集合。
    """

    subject_id: str
    subject_root: Path
    presets_root: Path


@dataclass(frozen=True)
class PresetPaths:
    """
    preset 级别路径集合。
    """

    subject_id: str
    preset: str
    subject_root: Path
    preset_root: Path
    workspace_root: Path
    workspace_dir: Path
    mesh_dir: Path
    forward_root: Path
    inverse_root: Path
    replay_root: Path


def now_timestamp() -> str:
    """
    返回 UTC ISO 时间戳。

    Returns
    -------
    str
        UTC ISO 时间戳。
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_timestamp(value: str | None) -> datetime | None:
    """
    解析 ISO 时间戳。

    Parameters
    ----------
    value : str or None
        时间戳字符串。

    Returns
    -------
    datetime or None
        解析后的时间，解析失败时返回 None。
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_json(path: Path, default: Any = None) -> Any:
    """
    读取 JSON 文件。

    Parameters
    ----------
    path : Path
        JSON 路径。
    default : Any, optional
        文件不存在时返回的默认值。

    Returns
    -------
    Any
        JSON 内容。
    """
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    """
    写出 JSON 文件。

    Parameters
    ----------
    path : Path
        JSON 路径。
    payload : Any
        待写出的对象。

    Returns
    -------
    None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def subject_paths_for(work_root: Path, subject_id: str) -> SubjectPaths:
    """
    返回 subject 路径集合。

    Parameters
    ----------
    work_root : Path
        工作根目录。
    subject_id : str
        subject 标识。

    Returns
    -------
    SubjectPaths
        subject 路径集合。
    """
    subject_root = work_root / "subjects" / subject_id
    return SubjectPaths(subject_id=subject_id, subject_root=subject_root, presets_root=subject_root / "presets")


def preset_paths_for(work_root: Path, subject_id: str, preset: str) -> PresetPaths:
    """
    返回 preset 路径集合。

    Parameters
    ----------
    work_root : Path
        工作根目录。
    subject_id : str
        subject 标识。
    preset : str
        preset 名称。

    Returns
    -------
    PresetPaths
        preset 路径集合。
    """
    subject_paths = subject_paths_for(work_root, subject_id)
    preset_root = subject_paths.presets_root / preset
    workspace_root = preset_root / "workspace"
    return PresetPaths(
        subject_id=subject_id,
        preset=preset,
        subject_root=subject_paths.subject_root,
        preset_root=preset_root,
        workspace_root=workspace_root,
        workspace_dir=workspace_root / f"m2m_{subject_id}",
        mesh_dir=preset_root / "mesh",
        forward_root=preset_root / "forward",
        inverse_root=preset_root / "inverse",
        replay_root=preset_root / "replay",
    )


def mesh_result_path(paths: PresetPaths) -> Path:
    """
    返回 mesh result.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。

    Returns
    -------
    Path
        result.json 路径。
    """
    return paths.mesh_dir / "result.json"


def mesh_run_path(paths: PresetPaths) -> Path:
    """
    返回 mesh run.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。

    Returns
    -------
    Path
        run.json 路径。
    """
    return paths.mesh_dir / "run.json"


def forward_case_dir(paths: PresetPaths, case_name: str) -> Path:
    """
    返回 forward case 目录。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。

    Returns
    -------
    Path
        case 目录。
    """
    return paths.forward_root / case_name


def forward_result_path(paths: PresetPaths, case_name: str) -> Path:
    """
    返回 forward result.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。

    Returns
    -------
    Path
        result.json 路径。
    """
    return forward_case_dir(paths, case_name) / "result.json"


def forward_run_path(paths: PresetPaths, case_name: str) -> Path:
    """
    返回 forward run.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。

    Returns
    -------
    Path
        run.json 路径。
    """
    return forward_case_dir(paths, case_name) / "run.json"


def forward_artifacts_dir(paths: PresetPaths, case_name: str) -> Path:
    """
    返回 forward 工件目录。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。

    Returns
    -------
    Path
        工件目录。
    """
    return forward_case_dir(paths, case_name) / "artifacts"


def inverse_seed_dir(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 inverse seed 目录。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        seed 目录。
    """
    return paths.inverse_root / case_name / "seeds" / str(seed)


def inverse_result_path(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 inverse result.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        result.json 路径。
    """
    return inverse_seed_dir(paths, case_name, seed) / "result.json"


def inverse_run_path(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 inverse run.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        run.json 路径。
    """
    return inverse_seed_dir(paths, case_name, seed) / "run.json"


def inverse_artifacts_dir(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 inverse 工件目录。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        工件目录。
    """
    return inverse_seed_dir(paths, case_name, seed) / "artifacts"


def replay_seed_dir(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 replay seed 目录。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        seed 目录。
    """
    return paths.replay_root / case_name / "seeds" / str(seed)


def replay_result_path(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 replay result.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        result.json 路径。
    """
    return replay_seed_dir(paths, case_name, seed) / "result.json"


def replay_run_path(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 replay run.json 路径。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        run.json 路径。
    """
    return replay_seed_dir(paths, case_name, seed) / "run.json"


def replay_artifacts_dir(paths: PresetPaths, case_name: str, seed: int) -> Path:
    """
    返回 replay 工件目录。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。
    case_name : str
        case 名称。
    seed : int
        随机种子。

    Returns
    -------
    Path
        工件目录。
    """
    return replay_seed_dir(paths, case_name, seed) / "artifacts"


def initialize_work_root(work_root: Path, manifest_snapshot: dict[str, Any]) -> None:
    """
    初始化 V2 工作根目录元数据。

    Parameters
    ----------
    work_root : Path
        工作根目录。
    manifest_snapshot : dict[str, Any]
        需要写入的 manifest 快照。

    Returns
    -------
    None
    """
    work_root.mkdir(parents=True, exist_ok=True)
    write_json(
        work_root / "schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": now_timestamp(),
        },
    )
    write_json(work_root / "manifest.snapshot.json", manifest_snapshot)


def ensure_v2_schema(work_root: Path) -> None:
    """
    校验工作目录为 V2 schema。

    Parameters
    ----------
    work_root : Path
        工作根目录。

    Returns
    -------
    None
    """
    schema = read_json(work_root / "schema.json")
    if not schema:
        raise FileNotFoundError(f"缺少 schema.json: {work_root}")
    if int(schema.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"不支持的 schema_version: {schema.get('schema_version')}")


def build_run_payload(
    stage: str,
    command: str,
    metadata: dict[str, Any],
    state: str,
    started_at: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """
    构造 run.json 负载。

    Parameters
    ----------
    stage : str
        阶段名称。
    command : str
        命令行。
    metadata : dict[str, Any]
        元数据字段。
    state : str
        运行状态。
    started_at : str or None, optional
        启动时间戳。
    error : str or None, optional
        错误信息。

    Returns
    -------
    dict[str, Any]
        run.json 负载。
    """
    timestamp = now_timestamp()
    payload = {
        "state": state,
        "stage": stage,
        "started_at": started_at or timestamp,
        "updated_at": timestamp,
        "finished_at": timestamp if state in {RUN_STATE_COMPLETED, RUN_STATE_FAILED} else None,
        "command": command,
        "error": error,
    }
    payload.update(metadata)
    return payload


class RunStateTracker:
    """
    管理单个 run.json 的生命周期与心跳。
    """

    def __init__(self, run_path: Path, stage: str, command: str, metadata: dict[str, Any]) -> None:
        """
        初始化运行状态跟踪器。

        Parameters
        ----------
        run_path : Path
            run.json 路径。
        stage : str
            阶段名称。
        command : str
            命令行。
        metadata : dict[str, Any]
            元数据字段。

        Returns
        -------
        None
        """
        self.run_path = run_path
        self.stage = stage
        self.command = command
        self.metadata = metadata
        self.started_at = now_timestamp()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """
        启动运行状态和心跳线程。

        Returns
        -------
        None
        """
        payload = build_run_payload(
            stage=self.stage,
            command=self.command,
            metadata=self.metadata,
            state=RUN_STATE_RUNNING,
            started_at=self.started_at,
        )
        write_json(self.run_path, payload)
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        """
        定期刷新 updated_at。

        Returns
        -------
        None
        """
        while not self._stop_event.wait(RUN_STATE_HEARTBEAT_SECONDS):
            with self._lock:
                payload = read_json(self.run_path, {})
                if payload.get("state") != RUN_STATE_RUNNING:
                    return
                payload["updated_at"] = now_timestamp()
                write_json(self.run_path, payload)

    def finish(self, state: str, error: str | None = None) -> None:
        """
        结束运行状态。

        Parameters
        ----------
        state : str
            结束状态。
        error : str or None, optional
            错误信息。

        Returns
        -------
        None
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        with self._lock:
            payload = build_run_payload(
                stage=self.stage,
                command=self.command,
                metadata=self.metadata,
                state=state,
                started_at=self.started_at,
                error=error,
            )
            write_json(self.run_path, payload)


def wait_for_run_completion(
    run_path: Path,
    description: str,
    poll_seconds: int = RUN_STATE_POLL_SECONDS,
    stale_seconds: int = RUN_STATE_STALE_SECONDS,
) -> dict[str, Any]:
    """
    等待运行中的任务完成。

    Parameters
    ----------
    run_path : Path
        run.json 路径。
    description : str
        日志描述。
    poll_seconds : int, optional
        轮询间隔。
    stale_seconds : int, optional
        僵死判定秒数。

    Returns
    -------
    dict[str, Any]
        最终 run.json 内容。
    """
    while True:
        payload = read_json(run_path)
        if payload is None:
            raise FileNotFoundError(f"缺少运行状态文件: {description} -> {run_path}")
        state = payload.get("state")
        if state in {RUN_STATE_COMPLETED, RUN_STATE_FAILED}:
            return payload
        updated_at = parse_timestamp(str(payload.get("updated_at")))
        if updated_at is None:
            raise RuntimeError(f"运行状态缺少有效 updated_at: {description} -> {run_path}")
        delta = datetime.now(timezone.utc) - updated_at
        if delta.total_seconds() > stale_seconds:
            raise TimeoutError(f"检测到僵死运行状态: {description} -> {run_path}")
        time.sleep(poll_seconds)


def collect_result_paths(work_root: Path, stage: str) -> list[Path]:
    """
    收集某个阶段的全部 result.json 路径。

    Parameters
    ----------
    work_root : Path
        工作根目录。
    stage : str
        阶段名称。

    Returns
    -------
    list[Path]
        result.json 路径列表。
    """
    patterns = {
        "mesh": "subjects/*/presets/*/mesh/result.json",
        "forward": "subjects/*/presets/*/forward/*/result.json",
        "inverse": "subjects/*/presets/*/inverse/*/seeds/*/result.json",
        "replay": "subjects/*/presets/*/replay/*/seeds/*/result.json",
    }
    if stage not in patterns:
        raise ValueError(f"不支持的阶段: {stage}")
    return sorted(work_root.glob(patterns[stage]))


def load_stage_results(
    work_root: Path,
    stage: str,
    allowed_presets: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """
    加载某个阶段的结果列表。

    Parameters
    ----------
    work_root : Path
        工作根目录。
    stage : str
        阶段名称。
    allowed_presets : Iterable[str] or None, optional
        允许加载的 preset 集合。

    Returns
    -------
    list[dict[str, Any]]
        结果对象列表。
    """
    preset_filter = set(allowed_presets or [])
    rows: list[dict[str, Any]] = []
    for path in collect_result_paths(work_root, stage):
        payload = read_json(path)
        if payload is None:
            continue
        preset = str(payload.get("preset", ""))
        if preset_filter and preset not in preset_filter:
            continue
        rows.append(payload)
    return rows
