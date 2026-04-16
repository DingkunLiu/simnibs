"""
DiFuMo ROI demo。

说明：
1. 扩展为覆盖 64/128/256/512/1024 五个分辨率
2. 优先选择左侧海马相关脑区
3. 只有 DiFuMo_1024 标签里明确提供了 LH 海马成分
4. 其余分辨率只能选择最接近的海马相关 component，不能强行标注为左侧
"""

from neuracle.atlas.demo.roi_demo_common import (
    build_atlas_roi,
    setup_demo_environment,
    summarize_roi,
    write_roi_demo,
)

DIFUMO_DEMOS = [
    {
        "atlas_name": "DiFuMo64",
        "area_name": "Superior frontal sulcus",
        "output_name": "atlas_difumo64_superior_frontal_sulcus",
        # 64 维标签中没有海马相关条目，因此这里显式选一个替代脑区保证 demo 可运行。
        "note": "该分辨率标签中没有海马相关条目，改用替代脑区 Superior frontal sulcus。",
    },
    {
        "atlas_name": "DiFuMo128",
        "area_name": "Parahippocampal gyrus",
        "output_name": "atlas_difumo128_parahippocampal_gyrus",
        "note": "标签未区分左右，使用最接近左侧海马的海马旁回 component。",
    },
    {
        "atlas_name": "DiFuMo256",
        "area_name": "Hippocampus anterior",
        "output_name": "atlas_difumo256_hippocampus_anterior",
        "note": "标签未区分左右，使用最接近左侧海马的前海马 component。",
    },
    {
        "atlas_name": "DiFuMo512",
        "area_name": "Hippocampus",
        "output_name": "atlas_difumo512_hippocampus",
        "note": "标签未区分左右，使用最接近左侧海马的海马 component。",
    },
    {
        "atlas_name": "DiFuMo1024",
        "area_name": "Hippocampus anterior LH",
        "output_name": "atlas_difumo1024_hippocampus_anterior_lh",
        "note": "该分辨率标签中明确提供左侧海马成分。",
    },
]


def main() -> None:
    setup_demo_environment("atlas_difumo_roi")
    for item in DIFUMO_DEMOS:
        # 每个分辨率独立跑一遍，方便比较不同 component 数量下 ROI 落位差异。
        atlas_name = item["atlas_name"]
        area_name = item["area_name"]
        print(f"atlas: {atlas_name}")
        print(f"note: {item['note']}")

        # 若同名脑区映射到多个 component，这里会自动得到合并后的缓存 mask。
        roi, roi_path = build_atlas_roi(atlas_name, area_name)
        node_count = summarize_roi(roi)
        output_dir = write_roi_demo(roi, item["output_name"])

        print(f"area: {area_name}")
        print(f"roi mask: {roi_path}")
        print(f"roi nodes: {node_count}")
        print(f"output: {output_dir}")
        print()


if __name__ == "__main__":
    main()
