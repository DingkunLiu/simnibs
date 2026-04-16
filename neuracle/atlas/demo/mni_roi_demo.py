"""
MNI 坐标 ROI demo。
"""

from neuracle.atlas.demo.roi_demo_common import (
    build_mni_sphere_roi,
    setup_demo_environment,
    summarize_roi,
    write_roi_demo,
)


def main() -> None:
    setup_demo_environment("mni_roi")
    # 这里直接保留 MNI 空间球形 ROI，由 SimNIBS 在内部完成到 subject 的映射。
    center = [-38.6, -18.7, 64.8]
    radius = 20.0
    roi = build_mni_sphere_roi(center=center, radius=radius)

    # 节点数反映球形 ROI 在 subject central surface 上的覆盖范围。
    node_count = summarize_roi(roi)
    output_dir = write_roi_demo(roi, "mni_sphere_roi")

    print(f"mni center: {center}")
    print(f"radius: {radius}")
    print(f"roi nodes: {node_count}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
