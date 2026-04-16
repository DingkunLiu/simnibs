"""
JulichBrainAtlas ROI demo。
"""

from neuracle.atlas.demo.roi_demo_common import (
    build_atlas_roi,
    setup_demo_environment,
    summarize_roi,
    write_roi_demo,
)


def main() -> None:
    setup_demo_environment("atlas_julich_roi")
    # Julich 选择一个明确分侧的躯体感觉皮层脑区，便于检查 bilateral atlas 的落位效果。
    atlas_name = "JulichBrainAtlas_3.1_207areas_MPM_bilateral_MNI152"
    area_name = "Area 3b (PostCG)_lh"
    roi, roi_path = build_atlas_roi(atlas_name, area_name)

    # 这里的节点数是 ROI 投影到 subject central surface 之后的节点数。
    node_count = summarize_roi(roi)
    output_dir = write_roi_demo(roi, "atlas_julich_area_3b_postcg_lh")

    print(f"atlas: {atlas_name}")
    print(f"area: {area_name}")
    print(f"roi mask: {roi_path}")
    print(f"roi nodes: {node_count}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
