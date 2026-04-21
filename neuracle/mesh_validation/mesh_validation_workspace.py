"""
Mesh validation 工作区管理。
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from neuracle.charm.mesh import create_mesh_step

LOGGER = logging.getLogger(__name__)

PRESET_ORDER = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8")
PRESET_CONFIGS = {
    "M0": {
        "elem_sizes": {
            "standard": {"range": [1.0, 5.0], "slope": 1.0},
            "WM": {"range": [1.0, 7.0], "slope": 1.0},
            "GM": {"range": [1.0, 2.0], "slope": 1.0},
            "Scalp": {"range": [1.0, 10.0], "slope": 0.6},
        },
        "skin_facet_size": 2.0,
    },
    "M1": {
        "elem_sizes": {
            "standard": {"range": [1.25, 6.0], "slope": 1.1},
            "WM": {"range": [1.25, 8.0], "slope": 1.1},
            "GM": {"range": [1.0, 2.5], "slope": 1.0},
            "Scalp": {"range": [1.5, 11.0], "slope": 0.7},
        },
        "skin_facet_size": 2.5,
    },
    "M2": {
        "elem_sizes": {
            "standard": {"range": [1.5, 7.0], "slope": 1.2},
            "WM": {"range": [1.5, 8.0], "slope": 1.2},
            "GM": {"range": [1.2, 3.0], "slope": 1.0},
            "Scalp": {"range": [2.0, 12.0], "slope": 0.8},
        },
        "skin_facet_size": 3.0,
    },
    "M3": {
        "elem_sizes": {
            "standard": {"range": [2.0, 8.0], "slope": 1.3},
            "WM": {"range": [2.0, 10.0], "slope": 1.3},
            "GM": {"range": [1.5, 3.5], "slope": 1.1},
            "Scalp": {"range": [2.5, 14.0], "slope": 0.9},
        },
        "skin_facet_size": 4.0,
    },
    "M4": {
        "elem_sizes": {
            "standard": {"range": [1.5, 7.0], "slope": 1.2},
            "WM": {"range": [1.5, 8.0], "slope": 1.2},
            "GM": {"range": [1.2, 3.0], "slope": 1.0},
            "Scalp": {"range": [2.0, 12.0], "slope": 0.8},
        },
        "skin_facet_size": 3.0,
        "facet_distances": {"standard": {"range": [0.2, 4.0], "slope": 0.7}},
    },
    "M5": {
        "elem_sizes": {
            "standard": {"range": [2.25, 9.0], "slope": 1.4},
            "WM": {"range": [2.25, 11.0], "slope": 1.4},
            "GM": {"range": [1.5, 3.75], "slope": 1.2},
            "Scalp": {"range": [3.0, 15.0], "slope": 1.0},
        },
        "skin_facet_size": 4.5,
    },
    "M6": {
        "elem_sizes": {
            "standard": {"range": [2.5, 10.0], "slope": 1.5},
            "WM": {"range": [2.5, 12.0], "slope": 1.5},
            "GM": {"range": [1.5, 4.0], "slope": 1.2},
            "Scalp": {"range": [3.5, 16.0], "slope": 1.1},
        },
        "skin_facet_size": 5.0,
    },
    "M7": {
        "elem_sizes": {
            "standard": {"range": [3.0, 11.0], "slope": 1.6},
            "WM": {"range": [3.0, 13.0], "slope": 1.6},
            "GM": {"range": [1.5, 4.5], "slope": 1.3},
            "Scalp": {"range": [4.0, 18.0], "slope": 1.2},
        },
        "skin_facet_size": 6.0,
    },
    "M8": {
        "elem_sizes": {
            "standard": {"range": [3.0, 11.0], "slope": 1.6},
            "WM": {"range": [3.0, 13.0], "slope": 1.6},
            "GM": {"range": [1.5, 4.5], "slope": 1.3},
            "Scalp": {"range": [4.0, 18.0], "slope": 1.2},
        },
        "skin_facet_size": 6.0,
        "facet_distances": {"standard": {"range": [0.3, 5.0], "slope": 0.9}},
    },
}
SIMNIBS_TISSUE_NAME_TO_TAG = {"WM": "1", "GM": "2", "Scalp": "5"}
TO_MNI_TRANSFORM_FILES = ("Conform2MNI_nonl.nii.gz", "MNI2Conform_nonl.nii.gz")


class WorkspacePreparationError(RuntimeError):
    """
    工作区准备失败。
    """


class MeshGenerationError(WorkspacePreparationError):
    """
    mesh 生成失败。
    """


@dataclass(frozen=True)
class SubjectConfig:
    """
    Subject 配置。
    """

    id: str
    m2m_dir: str
    reference_t1: str


@dataclass(frozen=True)
class SubjectPaths:
    """
    Subject 路径集合。
    """

    subject_root: Path


@dataclass(frozen=True)
class PresetPaths:
    """
    preset 路径集合。
    """

    subject_id: str
    preset: str
    subject_root: Path
    preset_root: Path
    workspace_root: Path
    workspace_dir: Path
    forward_dir: Path
    inverse_dir: Path


@dataclass(frozen=True)
class WorkspaceBuildResult:
    """
    工作区构建结果。
    """

    subject_id: str
    preset: str
    paths: PresetPaths
    workspace_dir: Path
    mesh_file: Path
    eeg_positions_dir: Path
    to_mni_dir: Path
    final_tissues_path: Path
    final_tissues_lut_path: Path
    final_tissues_mni_path: Path


def subject_paths_for(work_root: Path, subject_id: str) -> SubjectPaths:
    """
    返回 subject 路径。
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
    return SubjectPaths(subject_root=work_root / subject_id)


def preset_paths_for(work_root: Path, subject_id: str, preset: str) -> PresetPaths:
    """
    返回 preset 路径。
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
    preset_root = subject_paths.subject_root / preset
    workspace_root = preset_root / "workspace"
    workspace_dir = workspace_root / f"m2m_{subject_id}"
    return PresetPaths(
        subject_id=subject_id,
        preset=preset,
        subject_root=subject_paths.subject_root,
        preset_root=preset_root,
        workspace_root=workspace_root,
        workspace_dir=workspace_dir,
        forward_dir=preset_root / "forward",
        inverse_dir=preset_root / "inverse",
    )


def build_workspace_result(paths: PresetPaths) -> WorkspaceBuildResult:
    """
    根据 preset 路径构造 workspace 结果对象。

    Parameters
    ----------
    paths : PresetPaths
        preset 路径集合。

    Returns
    -------
    WorkspaceBuildResult
        workspace 结果对象。
    """
    return WorkspaceBuildResult(
        subject_id=paths.subject_id,
        preset=paths.preset,
        paths=paths,
        workspace_dir=paths.workspace_dir,
        mesh_file=paths.workspace_dir / f"{paths.subject_id}.msh",
        eeg_positions_dir=paths.workspace_dir / "eeg_positions",
        to_mni_dir=paths.workspace_dir / "toMNI",
        final_tissues_path=paths.workspace_dir / "final_tissues.nii.gz",
        final_tissues_lut_path=paths.workspace_dir / "final_tissues_LUT.txt",
        final_tissues_mni_path=paths.workspace_dir / "toMNI" / "final_tissues_MNI.nii.gz",
    )


def load_existing_workspace(subject: SubjectConfig, preset: str, work_root: Path) -> WorkspaceBuildResult:
    """
    从现有产物加载 workspace 结果对象。

    Parameters
    ----------
    subject : SubjectConfig
        subject 配置。
    preset : str
        preset 名称。
    work_root : Path
        工作根目录。

    Returns
    -------
    WorkspaceBuildResult
        已存在的 workspace 结果对象。
    """
    workspace = build_workspace_result(preset_paths_for(work_root, subject.id, preset))
    validate_workspace(workspace)
    return workspace


def resolve_workspace_eeg_cap(workspace: WorkspaceBuildResult, cap_name: str) -> str:
    """
    解析工作区内的 EEG cap 路径。
    Parameters
    ----------
    workspace : WorkspaceBuildResult
        工作区结果。
    cap_name : str
        cap 文件名或绝对路径。
    Returns
    -------
    str
        EEG cap 路径。
    """
    if os.path.isabs(cap_name):
        return cap_name
    normalized = cap_name if cap_name.endswith(".csv") else f"{cap_name}.csv"
    return str(workspace.eeg_positions_dir / normalized)


def prepare_workspace(
    subject: SubjectConfig,
    preset: str,
    work_root: Path,
    settings_path: Path,
    debug_mesh: bool,
) -> WorkspaceBuildResult:
    """
    构建单一可运行工作区。
    Parameters
    ----------
    subject : SubjectConfig
        subject 配置。
    preset : str
        preset 名称。
    work_root : Path
        工作根目录。
    settings_path : Path
        preset ini 路径。
    debug_mesh : bool
        是否保留 mesh 调试输出。
    Returns
    -------
    WorkspaceBuildResult
        工作区结果。
    """
    paths = preset_paths_for(work_root, subject.id, preset)
    _reset_directory(paths.workspace_root)
    paths.workspace_dir.mkdir(parents=True, exist_ok=True)
    source_m2m_dir = Path(subject.m2m_dir)
    _materialize_source_assets(source_m2m_dir, paths.workspace_dir, subject.id)
    try:
        create_mesh_step(
            subject_dir=subject.m2m_dir,
            debug=debug_mesh,
            settings_path=str(settings_path),
            output_dir=str(paths.workspace_dir),
        )
    except Exception as exc:  # noqa: BLE001
        raise MeshGenerationError(f"生成 mesh 失败: {exc}") from exc
    _materialize_to_mni_transforms(source_m2m_dir, paths.workspace_dir)
    result = build_workspace_result(paths)
    validate_workspace(result)
    return result


def validate_workspace(workspace: WorkspaceBuildResult) -> None:
    """
    校验工作区产物完整性。
    Parameters
    ----------
    workspace : WorkspaceBuildResult
        工作区结果。
    Returns
    -------
    None
    """
    required_paths = (
        workspace.workspace_dir,
        workspace.mesh_file,
        workspace.eeg_positions_dir,
        workspace.final_tissues_path,
        workspace.final_tissues_lut_path,
        workspace.to_mni_dir,
        workspace.final_tissues_mni_path,
        *(workspace.to_mni_dir / name for name in TO_MNI_TRANSFORM_FILES),
    )
    for path in required_paths:
        if not path.exists() and not path.is_symlink():
            raise WorkspacePreparationError(f"工作区缺少必需产物: {path}")
    if workspace.to_mni_dir.is_symlink():
        raise WorkspacePreparationError(f"toMNI 目录不允许为符号链接: {workspace.to_mni_dir}")
    for item in workspace.to_mni_dir.iterdir():
        if item.is_symlink():
            try:
                resolved = item.resolve(strict=False)
            except (OSError, RuntimeError) as exc:
                raise WorkspacePreparationError(f"检测到自引用链接: {item}") from exc
            if resolved == item:
                raise WorkspacePreparationError(f"检测到自引用链接: {item}")


def _materialize_source_assets(source_m2m_dir: Path, workspace_dir: Path, subject_id: str) -> None:
    """
    物化只读输入资产。
    Parameters
    ----------
    source_m2m_dir : Path
        原始 m2m 目录。
    workspace_dir : Path
        工作区目录。
    subject_id : str
        subject 标识。
    Returns
    -------
    None
    """
    excluded_names = {
        f"{subject_id}.msh",
        "final_tissues.nii.gz",
        "final_tissues_LUT.txt",
        "eeg_positions",
        "toMNI",
    }
    for source_item in source_m2m_dir.iterdir():
        if source_item.name in excluded_names:
            continue
        _materialize_path(source_item, workspace_dir / source_item.name)


def _materialize_to_mni_transforms(source_m2m_dir: Path, workspace_dir: Path) -> None:
    """
    物化 toMNI 形变场。
    Parameters
    ----------
    source_m2m_dir : Path
        原始 m2m 目录。
    workspace_dir : Path
        工作区目录。
    Returns
    -------
    None
    """
    source_to_mni_dir = source_m2m_dir / "toMNI"
    target_to_mni_dir = workspace_dir / "toMNI"
    target_to_mni_dir.mkdir(parents=True, exist_ok=True)
    for name in TO_MNI_TRANSFORM_FILES:
        source_path = source_to_mni_dir / name
        if not source_path.exists():
            raise WorkspacePreparationError(f"缺少 toMNI 形变场: {source_path}")
        _materialize_path(source_path, target_to_mni_dir / name)


def _materialize_path(source: Path, target: Path) -> None:
    """
    以链接优先、复制兜底的方式物化路径。
    Parameters
    ----------
    source : Path
        源路径。
    target : Path
        目标路径。
    Returns
    -------
    None
    """
    source_resolved = source.resolve(strict=False)
    target_resolved = target.resolve(strict=False)
    if source_resolved == target_resolved:
        raise WorkspacePreparationError(f"禁止把路径物化到自身: {source} -> {target}")
    if target.exists() or target.is_symlink():
        _remove_path(target)
    try:
        os.symlink(source, target, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def _reset_directory(path: Path) -> None:
    """
    重建目录。
    Parameters
    ----------
    path : Path
        目标目录。
    Returns
    -------
    None
    """
    if path.exists() or path.is_symlink():
        _remove_path(path)
    path.mkdir(parents=True, exist_ok=True)


def _remove_path(path: Path) -> None:
    """
    删除路径。
    Parameters
    ----------
    path : Path
        目标路径。
    Returns
    -------
    None
    """
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
