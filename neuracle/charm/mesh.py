"""
CHARM 步骤7: 四面体网格生成

从组织标签图像生成 tetrahedral 头模型网格。

原理：
    1. 加载上采样的组织标签图像
    2. 裁剪图像至感兴趣区域
    3. 使用 CGAL 进行四面体网格生成
    4. 重新标记内部空气边界
    5. 变换 EEG 电极位置到受试者空间
    6. 输出最终的 .msh 文件

输入：
    - segmentation/tissue_labeling_upsampled.nii.gz

输出：
    - {subid}.msh (头模型网格)
    - eeg_positions/*.csv, *.geo (EEG 电极位置)
    - mni_transf/final_labels.nii.gz, final_labels_MNI.nii.gz

用法：
    python -m neuracle.charm.mesh <subid> [--debug]
"""

import glob
import logging
import os
import shutil

import nibabel as nib
import numpy as np

from neuracle.utils.charm_utils import read_settings
from neuracle.utils.constants import N_WORKERS
from simnibs.mesh_tools.mesh_io import ElementData, write_msh
from simnibs.mesh_tools.meshing import create_mesh
from simnibs.utils import cond_utils, file_finder, transformations
from simnibs.utils.transformations import crop_vol

logger = logging.getLogger(__name__)


def create_mesh_step(
    subject_dir: str,
    debug: bool = False,
) -> None:
    """
    创建四面体网格

    从组织标签图像生成 tetrahedral 头模型网格。
    包括：加载组织标签图像、裁剪感兴趣区域、CGAL 四面体网格生成、
    重新标记内部空气边界、EEG 电极位置变换、输出最终 msh 文件。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})
    debug : bool, optional
        是否保存调试文件 (default: False)

    Returns
    -------
    None

    See Also
    --------
    create_mesh : CGAL 网格生成核心函数
    relabel_internal_air : 重新标记内部空气边界
    warp_coordinates : 坐标变换函数
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    output_msh_path = os.path.join(subject_dir, f"{sub_files.subid}.msh")
    settings = read_settings()
    mesh_settings = settings["mesh"]
    logger.info("开始生成网格")
    label_image = nib.load(sub_files.tissue_labeling_upsampled)
    label_buffer = np.round(label_image.get_fdata()).astype(np.uint16)
    label_affine = label_image.affine
    label_buffer, label_affine, _ = crop_vol(
        label_buffer, label_affine, label_buffer > 0, thickness_boundary=5
    )
    elem_sizes = mesh_settings["elem_sizes"]
    smooth_size_field = mesh_settings["smooth_size_field"]
    skin_facet_size = mesh_settings["skin_facet_size"]
    if not skin_facet_size:
        logger.info("skin_facet_size 未设置或为 0，禁用皮肤面大小限制")
        skin_facet_size = None
    facet_distances = mesh_settings["facet_distances"]
    optimize = mesh_settings["optimize"]
    apply_cream = mesh_settings["apply_cream"]
    remove_spikes = mesh_settings["remove_spikes"]
    skin_tag = mesh_settings["skin_tag"]
    if not skin_tag:
        logger.info("skin_tag 未设置或为 0，不输出皮肤表面")
        skin_tag = None
    hierarchy = mesh_settings["hierarchy"]
    if not hierarchy:
        logger.info("hierarchy 未设置，使用默认层级")
        hierarchy = None
    smooth_steps = mesh_settings["smooth_steps"]
    skin_care = mesh_settings["skin_care"]
    mmg_noinsert = mesh_settings["mmg_noinsert"]
    logger.info("使用的皮肤标签: %s", skin_tag)
    debug_path = None
    if debug:
        debug_path = sub_files.subpath
        logger.info("启用调试模式，调试文件将保存至: %s", debug_path)
    num_threads = settings["general"]["threads"]
    if num_threads <= 0:
        logger.info("线程数配置无效 (%d)，使用 N_WORKERS (%d)", num_threads, N_WORKERS)
        num_threads = N_WORKERS
    final_mesh = create_mesh(
        label_buffer,
        label_affine,
        elem_sizes=elem_sizes,
        smooth_size_field=smooth_size_field,
        skin_facet_size=skin_facet_size,
        facet_distances=facet_distances,
        optimize=optimize,
        remove_spikes=remove_spikes,
        skin_tag=skin_tag,
        hierarchy=hierarchy,
        apply_cream=apply_cream,
        smooth_steps=smooth_steps,
        skin_care=skin_care,
        num_threads=num_threads,
        mmg_noinsert=mmg_noinsert,
        debug_path=debug_path,
        debug=debug,
    )
    logger.info("正在重新标记内部空气边界")
    final_mesh = final_mesh.relabel_internal_air()
    logger.info("正在写入网格文件")
    write_msh(final_mesh, output_msh_path)
    v = final_mesh.view(cond_list=cond_utils.standard_cond(), add_logo=True)
    v.write_opt(output_msh_path)
    logger.info("正在变换 EEG 电极位置")
    idx = (final_mesh.elm.elm_type == 2) & (final_mesh.elm.tag1 == skin_tag)
    mesh = final_mesh.crop_mesh(elements=final_mesh.elm.elm_number[idx])
    if not os.path.exists(sub_files.eeg_cap_folder):
        os.mkdir(sub_files.eeg_cap_folder)
        logger.info("创建 EEG cap 文件夹: %s", sub_files.eeg_cap_folder)
    cap_files = glob.glob(os.path.join(file_finder.ElectrodeCaps_MNI, "*.csv"))
    for fn in cap_files:
        fn_out = os.path.splitext(os.path.basename(fn))[0]
        fn_out = os.path.join(sub_files.eeg_cap_folder, fn_out)
        transformations.warp_coordinates(
            fn,
            sub_files.subpath,
            transformation_direction="mni2subject",
            out_name=fn_out + ".csv",
            out_geo=fn_out + ".geo",
            mesh_in=mesh,
            skin_tag=skin_tag,
        )
    logger.info("正在从网格写入标签图像")
    MNI_template = file_finder.Templates().mni_volume
    mesh = final_mesh.crop_mesh(elm_type=4)
    field = mesh.elm.tag1.astype(np.uint16)
    ed = ElementData(field)
    ed.mesh = mesh
    ed.to_deformed_grid(
        sub_files.mni2conf_nonl,
        MNI_template,
        out=sub_files.final_labels_MNI,
        out_original=sub_files.final_labels,
        method="assign",
        order=0,
        reference_original=sub_files.reference_volume,
    )
    fn_lut = sub_files.final_labels.rsplit(".", 2)[0] + "_LUT.txt"
    shutil.copyfile(file_finder.templates.final_tissues_LUT, fn_lut)
    logger.info("网格生成完成")
