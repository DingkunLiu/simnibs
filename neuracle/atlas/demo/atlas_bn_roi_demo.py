"""
BN atlas ROI demo。

前提：
1. 已生成 atlas registry
2. 已完成 BN atlas 标准化
3. 已生成标准化后的单脑区 ROI 文件
"""

from neuracle.atlas.demo.roi_demo_common import (
    build_atlas_roi,
    setup_demo_environment,
    summarize_roi,
    write_roi_demo,
)


def main() -> None:
    setup_demo_environment("atlas_bn_roi")
    # BN 选择一个稳定且容易核对的左侧脑区，便于验证 atlas -> subject 映射链路。
    atlas_name = "BN_Atlas_246_1mm"
    area_name = "A8m_L"

    # 这里返回的是 SimNIBS ROI 对象，以及对应的标准化 MNI mask 文件路径。
    roi, roi_path = build_atlas_roi(atlas_name, area_name)

    # 节点数表示该 ROI 最终落到 subject central surface 后命中的表面节点数量。
    node_count = summarize_roi(roi)

    # 输出 Gmsh 可视化文件，便于人工检查 ROI 是否落在预期位置。
    output_dir = write_roi_demo(roi, "atlas_bn_A8m_L")

    print(f"atlas: {atlas_name}")
    print(f"area: {area_name}")
    print(f"roi mask: {roi_path}")
    print(f"roi nodes: {node_count}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
