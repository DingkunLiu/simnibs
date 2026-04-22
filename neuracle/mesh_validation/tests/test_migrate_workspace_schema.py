from pathlib import Path

import pytest

from neuracle.mesh_validation.mesh_validation_schema import preset_paths_for, replay_result_path
from neuracle.mesh_validation.migrate_workspace_schema import (
    copy_tree_incremental,
    ensure_target_root,
    is_old_replay_dir,
    parse_old_seed,
    relocate_replay_dirs,
    rewrite_migrated_path_value,
)


def test_relocate_replay_dirs_moves_artifacts_to_preset_replay_dir(tmp_path: Path) -> None:
    old_root = tmp_path / "work_v1"
    new_root = tmp_path / "work_v2"
    replay_a = old_root / "ernie" / "M1" / "inverse" / "focality_inv_demo" / "seed_7" / "replay_on_m0"
    replay_b = old_root / "ernie" / "M2" / "inverse" / "focality_inv_demo" / "seed_7" / "replay_on_m0"
    replay_a.mkdir(parents=True)
    replay_b.mkdir(parents=True)
    (replay_a / "a.txt").write_text("m1", encoding="utf-8")
    (replay_b / "b.txt").write_text("m2", encoding="utf-8")

    relocate_replay_dirs(old_root, new_root)

    m1_target = replay_result_path(preset_paths_for(new_root, "ernie", "M1"), "focality_inv_demo", 7).parent / "artifacts"
    m2_target = replay_result_path(preset_paths_for(new_root, "ernie", "M2"), "focality_inv_demo", 7).parent / "artifacts"
    assert (m1_target / "a.txt").read_text(encoding="utf-8") == "m1"
    assert (m2_target / "b.txt").read_text(encoding="utf-8") == "m2"
    assert replay_a.exists()
    assert replay_b.exists()


def test_rewrite_migrated_path_value_rewrites_replay_paths_to_v2_replay_dir(tmp_path: Path) -> None:
    old_root = tmp_path / "work_v1"
    new_root = tmp_path / "work_v2"
    old_path = old_root / "ernie" / "M1" / "inverse" / "focality_inv_demo" / "seed_7" / "replay_on_m0" / "artifact.nii.gz"
    expected = replay_result_path(preset_paths_for(new_root, "ernie", "M1"), "focality_inv_demo", 7).parent / "artifacts" / "artifact.nii.gz"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("ok", encoding="utf-8")

    rewritten = rewrite_migrated_path_value(old_root, new_root, str(old_path))

    assert rewritten == str(expected)
    assert "subjects\\presets\\presets" not in rewritten


def test_rewrite_migrated_path_value_accepts_legacy_absolute_prefix(tmp_path: Path) -> None:
    old_root = tmp_path / "work_v1"
    new_root = tmp_path / "work_v2"
    legacy_path = Path(
        "/home/dell/simnibs/neuracle/mesh_validation/work_remote_full/ernie/M1/inverse/focality_inv_demo/seed_7/replay_on_m0/artifact.nii.gz"
    )
    expected = replay_result_path(preset_paths_for(new_root, "ernie", "M1"), "focality_inv_demo", 7).parent / "artifacts" / "artifact.nii.gz"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("ok", encoding="utf-8")

    rewritten = rewrite_migrated_path_value(old_root, new_root, str(legacy_path))

    assert rewritten == str(expected)


def test_rewrite_migrated_path_value_fails_when_replay_artifact_is_missing(tmp_path: Path) -> None:
    old_root = tmp_path / "work_v1"
    new_root = tmp_path / "work_v2"
    old_path = old_root / "ernie" / "M1" / "inverse" / "focality_inv_demo" / "seed_7" / "replay_on_m0" / "artifact.nii.gz"

    with pytest.raises(FileNotFoundError, match="artifact.nii.gz"):
        rewrite_migrated_path_value(old_root, new_root, str(old_path))


def test_parse_old_seed_rejects_unexpected_seed_name() -> None:
    with pytest.raises(ValueError):
        parse_old_seed("7")


def test_copy_tree_incremental_skips_existing_files_and_copies_missing_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "nested").mkdir(parents=True)
    (source / "inverse" / "demo" / "seed_7" / "replay_on_m0").mkdir(parents=True)
    target.mkdir()
    (source / "keep.txt").write_text("source", encoding="utf-8")
    (source / "nested" / "new.txt").write_text("new", encoding="utf-8")
    (source / "inverse" / "demo" / "seed_7" / "replay_on_m0" / "artifact.txt").write_text("artifact", encoding="utf-8")
    (target / "keep.txt").write_text("target", encoding="utf-8")

    copy_tree_incremental(source, target, should_skip=is_old_replay_dir)

    assert (target / "keep.txt").read_text(encoding="utf-8") == "target"
    assert (target / "nested" / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (target / "inverse" / "demo" / "seed_7" / "replay_on_m0").exists()


def test_ensure_target_root_allows_incremental_resume_for_initialized_v2_root(tmp_path: Path) -> None:
    new_root = tmp_path / "work_v2"
    new_root.mkdir()
    (new_root / "schema.json").write_text("{}", encoding="utf-8")
    (new_root / "subjects").mkdir()

    ensure_target_root(new_root)
