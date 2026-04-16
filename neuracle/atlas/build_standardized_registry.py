from __future__ import annotations

from neuracle.atlas.registry import write_atlas_registry


def main() -> None:
    # registry 固定写相对路径，便于跨机器和跨目录迁移。
    path = write_atlas_registry()
    print(f"已生成 atlas registry: {path}")


if __name__ == "__main__":
    main()
