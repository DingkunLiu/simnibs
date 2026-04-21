"""
CHARM 步骤 7: 四面体网格生成。

该模块从已有的组织标签图像生成头模四面体网格，并导出配套的 EEG
电极位置与最终组织标签体积。默认行为保持与原始 CHARM 流程一致；
当传入 `settings_path` 和 `output_dir` 时，可在不覆盖原始 `m2m`
目录的前提下，生成独立的网格变体输出。
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


def _resolve_mesh_outputs(subject_dir: str, output_dir: str | None) -> dict[str, str]:
    """
    解析 mesh 步骤的输出路径。

    Parameters
    ----------
    subject_dir : str
        原始 subject 目录路径。
    output_dir : str or None
        独立输出目录。为 None 时，沿用原始 `m2m` 目录输出。

    Returns
    -------
    dict[str, str]
        mesh 步骤涉及的输出路径集合。
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    root_dir = output_dir or sub_files.subpath
    eeg_cap_folder = os.path.join(root_dir, "eeg_positions")
    mni_dir = os.path.join(root_dir, "toMNI")

    return {
        "root_dir": root_dir,
        "output_msh_path": os.path.join(root_dir, f"{sub_files.subid}.msh"),
        "eeg_cap_folder": eeg_cap_folder,
        "final_labels": os.path.join(root_dir, "final_tissues.nii.gz"),
        "final_labels_lut": os.path.join(root_dir, "final_tissues_LUT.txt"),
        "final_labels_mni": os.path.join(mni_dir, "final_tissues_MNI.nii.gz"),
    }


def create_mesh_step(
    subject_dir: str,
    debug: bool = False,
    settings_path: str | None = None,
    output_dir: str | None = None,
) -> None:
    """
    创建四面体网格。

    Parameters
    ----------
    subject_dir : str
        原始 subject 目录路径 (`m2m_{subid}`)。
    debug : bool, optional
        是否输出调试中间文件 (default: False)。
    settings_path : str or None, optional
        CHARM 配置文件路径。为 None 时，使用 SimNIBS 默认 `charm.ini`
        (default: None)。
    output_dir : str or None, optional
        独立输出目录。为 None 时写回原始 `m2m` 目录；传入后会把 mesh、
        `eeg_positions`、`final_tissues*` 写入该目录，避免覆盖原始数据
        (default: None)。

    Returns
    -------
    None
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    outputs = _resolve_mesh_outputs(subject_dir=subject_dir, output_dir=output_dir)
    settings = read_settings(settings_path=settings_path)
    mesh_settings = settings["mesh"]

    os.makedirs(outputs["root_dir"], exist_ok=True)
    os.makedirs(outputs["eeg_cap_folder"], exist_ok=True)
    os.makedirs(os.path.dirname(outputs["final_labels_mni"]), exist_ok=True)

    logger.info("开始生成网格: subject=%s, output_dir=%s", subject_dir, outputs["root_dir"])
    label_image = nib.load(sub_files.tissue_labeling_upsampled)
    label_buffer = np.round(label_image.get_fdata()).astype(np.uint16)
    label_affine = label_image.affine
    label_buffer, label_affine, _ = crop_vol(
        label_buffer,
        label_affine,
        label_buffer > 0,
        thickness_boundary=5,
    )

    elem_sizes = mesh_settings["elem_sizes"]
    smooth_size_field = mesh_settings["smooth_size_field"]
    skin_facet_size = mesh_settings["skin_facet_size"]
    if not skin_facet_size:
        logger.info("skin_facet_size 未设置或为 0，禁用头皮表面尺寸限制")
        skin_facet_size = None

    facet_distances = mesh_settings["facet_distances"]
    optimize = mesh_settings["optimize"]
    apply_cream = mesh_settings["apply_cream"]
    remove_spikes = mesh_settings["remove_spikes"]
    skin_tag = mesh_settings["skin_tag"]
    if not skin_tag:
        logger.info("skin_tag 未设置或为 0，不输出头皮表面")
        skin_tag = None

    hierarchy = mesh_settings["hierarchy"]
    if not hierarchy:
        logger.info("hierarchy 未设置，使用默认层级")
        hierarchy = None

    smooth_steps = mesh_settings["smooth_steps"]
    skin_care = mesh_settings["skin_care"]
    mmg_noinsert = mesh_settings["mmg_noinsert"]
    debug_path = outputs["root_dir"] if debug else None

    num_threads = settings["general"]["threads"]
    if num_threads <= 0:
        logger.info("线程数配置无效 (%d)，改用 N_WORKERS=%d", num_threads, N_WORKERS)
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

    logger.info("重新标记内部空气边界")
    final_mesh = final_mesh.relabel_internal_air()

    logger.info("写入 mesh 文件: %s", outputs["output_msh_path"])
    write_msh(final_mesh, outputs["output_msh_path"])
    view = final_mesh.view(cond_list=cond_utils.standard_cond(), add_logo=True)
    view.write_opt(outputs["output_msh_path"])

    logger.info("导出 EEG 电极位置到: %s", outputs["eeg_cap_folder"])
    idx = (final_mesh.elm.elm_type == 2) & (final_mesh.elm.tag1 == skin_tag)
    scalp_mesh = final_mesh.crop_mesh(elements=final_mesh.elm.elm_number[idx])
    cap_files = glob.glob(os.path.join(file_finder.ElectrodeCaps_MNI, "*.csv"))
    for cap_path in cap_files:
        cap_name = os.path.splitext(os.path.basename(cap_path))[0]
        output_prefix = os.path.join(outputs["eeg_cap_folder"], cap_name)
        transformations.warp_coordinates(
            cap_path,
            sub_files.subpath,
            transformation_direction="mni2subject",
            out_name=output_prefix + ".csv",
            out_geo=output_prefix + ".geo",
            mesh_in=scalp_mesh,
            skin_tag=skin_tag,
        )

    logger.info("从 mesh 反写最终组织标签")
    mni_template = file_finder.Templates().mni_volume
    tetra_mesh = final_mesh.crop_mesh(elm_type=4)
    field = tetra_mesh.elm.tag1.astype(np.uint16)
    element_data = ElementData(field)
    element_data.mesh = tetra_mesh
    element_data.to_deformed_grid(
        sub_files.mni2conf_nonl,
        mni_template,
        out=outputs["final_labels_mni"],
        out_original=outputs["final_labels"],
        method="assign",
        order=0,
        reference_original=sub_files.reference_volume,
    )
    shutil.copyfile(file_finder.templates.final_tissues_LUT, outputs["final_labels_lut"])
    logger.info("网格生成完成: %s", outputs["output_msh_path"])
