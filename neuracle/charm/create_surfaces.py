"""
CHARM 步骤6: 皮层表面重建

从分割结果重建皮层表面（white、pial、central surface）。

原理：
    1. 如果提供 FreeSurfer 目录，从 FreeSurfer 表面加载
    2. 否则使用 TopoFit 方法进行皮层表面估计（默认）
    3. 可选地根据表面更新分割结果

输入：
    - segmentation/tissue_labeling_upsampled.nii.gz
    - segmentation/norm_image.nii.gz

输出：
    - surfaces/lh.white
    - surfaces/rh.white
    - surfaces/lh.pial
    - surfaces/rh.pial
    - surfaces/lh.central
    - surfaces/rh.central
    - surfaces/lh.sphere
    - surfaces/rh.sphere
    - surfaces/lh.sphere.reg
    - surfaces/rh.sphere.reg

用法：
    python -m neuracle.charm.create_surfaces <subid> [--fs-subjects-dir <path>]
"""

import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import time

import nibabel as nib
import numpy as np

from neuracle.utils.charm_utils import read_settings
from simnibs import SIMNIBSDIR
from simnibs.mesh_tools.mesh_io import (
    Msh,
    load_freesurfer_surfaces,
    read_gifti_surface,
    read_off,
    write_gifti_surface,
    write_off,
)
from simnibs.segmentation import brain_surface, charm_utils
from simnibs.segmentation.charm_main import (
    _fs_lut_labels_to_fs_lut_values,
    _write_cortex_as_gifti,
    _write_gifti,
    _write_vertex_data_as_curv,
)
from simnibs.utils import file_finder, settings_reader
from simnibs.utils.spawn_process import spawn_process
from simnibs.utils.threading import run_in_multiprocessing_pool

logger = logging.getLogger(__name__)


def create_surfaces(
    subject_dir: str,
    fs_dir: str | None = None,
) -> None:
    """
    创建皮层表面

    从分割结果重建皮层表面（white、pial、central surface）。
    支持 TopoFit 方法（默认）和 FreeSurfer 方法两种重建方式。

    Parameters
    ----------
    subject_dir : str
        受试者目录路径 (m2m_{subid})
    fs_dir : str or None, optional
        FreeSurfer subject directory (default: None)

    Returns
    -------
    None

    See Also
    --------
    _create_surfaces_from_topofit : TopoFit 表面重建
    _create_surfaces_from_freesurfer : FreeSurfer 表面重建
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    settings = read_settings()
    surface_settings = settings["surfaces"]
    samseg_settings = settings["samseg"]
    atlas_settings = _get_atlas_settings(samseg_settings)
    os.makedirs(sub_files.surface_folder, exist_ok=True)
    if fs_dir:
        logger.info("使用 FreeSurfer 方法创建表面")
        _create_surfaces_from_freesurfer(
            sub_files, fs_dir, surface_settings, atlas_settings
        )
    else:
        logger.info("使用 TopoFit 方法创建表面")
        _create_surfaces_from_topofit(sub_files, surface_settings, atlas_settings)
    logger.info("表面重建完成")


def _create_surfaces_from_topofit(
    sub_files: file_finder.SubjectFiles,
    surface_settings: dict,
    atlas_settings: dict,
) -> None:
    """
    使用 TopoFit 方法创建表面

    Parameters
    ----------
    sub_files : SubjectFiles
        Subject files 对象
    surface_settings : dict
        表面设置
    atlas_settings : dict
        Atlas 设置

    Returns
    -------
    None
    """
    t_start = time.perf_counter()
    tissue_map_simnibs = atlas_settings["conductivity_mapping"]["simnibs_tissues"]

    logger.info("正在估计皮层表面")
    cortex, curv = brain_surface.cortical_surface_estimation(
        [sub_files],
        surface_settings["topofit_contrast"],
        surface_settings["topofit_resolution"],
        surface_settings["topofit_device"],
    )
    # only one subject
    cortex = cortex[0]
    curv = curv[0]
    cortex.lh.registration = None
    cortex.rh.registration = None

    logger.info("正在估计中央灰质表面")
    central = cortex.estimate_layers(
        surface_settings["central_surface_method"],
        surface_settings["central_surface_fraction"],
        curv_kwargs=dict(smooth_iter=10),
        return_surface=True,
    )

    logger.info("正在保存皮层表面")
    _write_cortex_as_gifti(cortex, sub_files)
    _write_gifti(central.lh, sub_files, "central", "lh")
    _write_gifti(central.rh, sub_files, "central", "rh")
    _write_vertex_data_as_curv(curv, sub_files)

    if not cortex.lh.has_registration():
        logger.info("正在生成球面配准")

        _ = run_in_multiprocessing_pool(
            surface_settings["spherical_registration_process_pool"],
            brain_surface.spherical_registration_cat,
            itertools.product([sub_files], file_finder.HEMISPHERES),
            start_method="spawn",
        )

    if surface_settings["update_segmentation_from_surfaces"]:
        logger.info("正在使用皮层表面更新分割结果")
        charm_utils.update_labeling_from_cortical_surfaces(
            sub_files,
            _fs_lut_labels_to_fs_lut_values(
                surface_settings["update_segmentation_protect"]
            ),
            tissue_map_simnibs,
        )

    t_end = time.perf_counter()
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(t_end - t_start))
    logger.info("表面创建耗时: %s", time_elapsed)


def _create_surfaces_from_freesurfer(
    sub_files: file_finder.SubjectFiles,
    fs_dir: str,
    surface_settings: dict,
    atlas_settings: dict,
) -> None:
    """
    从 FreeSurfer 表面创建

    Parameters
    ----------
    sub_files : SubjectFiles
        Subject files 对象
    fs_dir : str
        FreeSurfer subject directory
    surface_settings : dict
        表面设置
    atlas_settings : dict
        Atlas 设置

    Returns
    -------
    None
    """
    t_start = time.perf_counter()
    logger.info("开始创建表面")
    fs_sub = file_finder.FreeSurferSubject(fs_dir)
    logger.info("正在使用 FreeSurfer 的表面")
    logger.info("FreeSurfer subject 目录为 %s", fs_sub.root.resolve())
    surfaces = {
        "white": load_freesurfer_surfaces(fs_sub, "white", coord="ras"),
        "sphere": load_freesurfer_surfaces(fs_sub, "sphere"),
        "sphere.reg": load_freesurfer_surfaces(fs_sub, "sphere.reg"),
    }
    surfaces["pial"] = _load_freesurfer_pial_surface(fs_sub)
    logger.info("正在估计中央灰质表面")
    tissue_map_simnibs = atlas_settings["conductivity_mapping"]["simnibs_tissues"]
    surfaces["central"] = {
        h: brain_surface.estimate_central_surface(
            surfaces["white"][h], surfaces["pial"][h]
        )
        for h in sub_files.hemispheres
    }
    for surface, v in surfaces.items():
        for hemi, mesh in v.items():
            write_gifti_surface(mesh, sub_files.get_surface(hemi, surface))
    if surface_settings["update_segmentation_from_surfaces"]:
        logger.info("正在使用皮层表面更新分割结果")
        charm_utils.update_labeling_from_cortical_surfaces(
            sub_files,
            surface_settings["protect"],
            tissue_map_simnibs,
        )
    t_end = time.perf_counter()
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(t_end - t_start))
    logger.info("表面创建耗时: %s", time_elapsed)


def _create_surfaces_from_cat12(
    sub_files: file_finder.SubjectFiles,
    surface_settings: dict,
    atlas_settings: dict,
) -> None:
    """
    使用 CAT12 方法创建表面

    Parameters
    ----------
    sub_files : SubjectFiles
        Subject files 对象
    surface_settings : dict
        表面设置
    atlas_settings : dict
        Atlas 设置

    Returns
    -------
    None
    """
    logger.info("开始创建表面 (CAT12 方法)")
    starttime = time.time()
    fs_avg_dir = file_finder.Templates().freesurfer_templates
    nprocesses = surface_settings["processes"]
    surf = surface_settings["surf"]
    pial = surface_settings["pial"]
    vdist = surface_settings["vdist"]
    voxsize_pbt = surface_settings["voxsize_pbt"]
    voxsize_refine_cs = surface_settings["voxsize_refinecs"]
    th_initial = surface_settings["th_initial"]
    no_selfintersections = surface_settings["no_selfintersections"]
    fillin_gm_from_surf = surface_settings["fillin_gm_from_surf"]
    open_sulci_from_surf = surface_settings["open_sulci_from_surf"]
    exclusion_tissues_fillin_gm = surface_settings["exclusion_tissues_fillin_gm"]
    exclusion_tissues_open_csf = surface_settings["exclusion_tissues_open_csf"]
    multithreading_script = [
        os.path.join(SIMNIBSDIR, "segmentation", "run_cat_multiprocessing.py")
    ]
    args_list = (
        [
            "--Ymf",
            sub_files.norm_image,
            "--Yleft_path",
            sub_files.hemi_mask,
            "--Ymaskhemis_path",
            sub_files.cereb_mask,
            "--surface_folder",
            sub_files.surface_folder,
            "--fsavgdir",
            fs_avg_dir,
            "--surf",
        ]
        + surf
        + ["--pial"]
        + pial
        + [
            "--vdist",
            str(vdist[0]),
            str(vdist[1]),
            "--voxsize_pbt",
            str(voxsize_pbt[0]),
            str(voxsize_pbt[1]),
            "--voxsizeCS",
            str(voxsize_refine_cs[0]),
            str(voxsize_refine_cs[1]),
            "--th_initial",
            str(th_initial),
            "--no_intersect",
            str(no_selfintersections),
            "--nprocesses",
            str(nprocesses),
        ]
    )
    proc = subprocess.run(
        [sys.executable] + multithreading_script + args_list, stderr=subprocess.PIPE
    )
    # 过滤 CAT12 进度条噪声
    stderr_text = proc.stderr.decode("ASCII", errors="ignore")
    stderr_text = re.sub(r"/-\|/-", "", stderr_text)
    stderr_text = re.sub(
        r"Selecting intersections ... \d{1,3} %Selecting intersections ... \d{1,3} %",
        "",
        stderr_text,
    )
    stderr_text = stderr_text.replace("\r", "")
    if stderr_text.strip():
        logger.debug(stderr_text)
    if proc.returncode != 0:
        logger.error("CAT12 子进程执行失败，返回码: %d", proc.returncode)
        logger.error("错误信息: %s", proc.stderr.decode("ASCII", errors="ignore"))
        raise RuntimeError(
            f"CAT12 subprocess failed with return code {proc.returncode}"
        )
    proc.check_returncode()
    elapsed = time.time() - starttime
    logger.info("表面创建总耗时 (HH:MM:SS):")
    logger.info(time.strftime("%H:%M:%S", time.gmtime(elapsed)))
    sub_files = file_finder.SubjectFiles(subpath=sub_files.subpath)
    if fillin_gm_from_surf or open_sulci_from_surf:
        logger.info(
            "根据表面改善 GM 分割: fillin_gm=%s, open_sulci=%s",
            fillin_gm_from_surf,
            open_sulci_from_surf,
        )
        _improve_gm_from_surfaces(
            sub_files,
            fillin_gm_from_surf,
            open_sulci_from_surf,
            exclusion_tissues_fillin_gm,
            exclusion_tissues_open_csf,
        )


def _improve_gm_from_surfaces(
    sub_files: file_finder.SubjectFiles,
    fillin_gm_from_surf: bool,
    open_sulci_from_surf: bool,
    exclusion_tissues_fillin_gm: list,
    exclusion_tissues_open_csf: list,
) -> None:
    """
    根据表面改善 GM 分割

    Parameters
    ----------
    sub_files : SubjectFiles
        Subject files 对象
    fillin_gm_from_surf : bool
        是否填充 GM 层
    open_sulci_from_surf : bool
        是否从表面展开脑沟
    exclusion_tissues_fillin_gm : list
        填充 GM 排除的组织
    exclusion_tissues_open_csf : list
        开放 CSF 排除的组织

    Returns
    -------
    None
    """
    logger.info("正在根据表面改善 GM 分割")
    starttime = time.time()
    label_nii = nib.load(sub_files.tissue_labeling_upsampled)
    label_img = np.asanyarray(label_nii.dataobj)
    label_affine = label_nii.affine
    label_nii = nib.load(sub_files.labeling)
    label_org_img = np.asanyarray(label_nii.dataobj)
    label_org_affine = label_nii.affine
    if fillin_gm_from_surf:
        logger.info("开始 GM 层填充")
        m = Msh()
        if "lh" in sub_files.hemispheres:
            m = m.join_mesh(read_gifti_surface(sub_files.get_surface("lh", "central")))
        if "rh" in sub_files.hemispheres:
            m = m.join_mesh(read_gifti_surface(sub_files.get_surface("rh", "central")))
        if m.nodes.nr > 0:
            label_img = charm_utils._fillin_gm_layer(
                label_img,
                label_affine,
                label_org_img,
                label_org_affine,
                m,
                exclusion_tissues=exclusion_tissues_fillin_gm,
            )
            label_nii = nib.Nifti1Image(label_img, label_affine)
            nib.save(label_nii, sub_files.tissue_labeling_upsampled)
            logger.info("GM 层填充完成")
        else:
            logger.warning("左右半球均未重建。跳过从 GM 表面填充。")
    if open_sulci_from_surf:
        logger.info("开始从表面展开脑沟")
        m = Msh()
        if "lh" in sub_files.hemispheres:
            mesh2 = read_gifti_surface(sub_files.get_surface("lh", "pial"))
            with tempfile.NamedTemporaryFile(suffix=".off", delete=False) as f:
                mesh_fn = f.name
            write_off(mesh2, mesh_fn)
            cmd = [file_finder.path2bin("meshfix"), mesh_fn, "-o", mesh_fn]
            spawn_process(cmd, lvl=logging.DEBUG)
            m = m.join_mesh(read_off(mesh_fn))
            os.remove(mesh_fn)
            if os.path.isfile("meshfix_log.txt"):
                os.remove("meshfix_log.txt")
        if "rh" in sub_files.hemispheres:
            mesh2 = read_gifti_surface(sub_files.get_surface("rh", "pial"))
            with tempfile.NamedTemporaryFile(suffix=".off", delete=False) as f:
                mesh_fn = f.name
            write_off(mesh2, mesh_fn)
            cmd = [file_finder.path2bin("meshfix"), mesh_fn, "-o", mesh_fn]
            spawn_process(cmd, lvl=logging.DEBUG)
            m = m.join_mesh(read_off(mesh_fn))
            os.remove(mesh_fn)
            if os.path.isfile("meshfix_log.txt"):
                os.remove("meshfix_log.txt")
        if m.nodes.nr > 0:
            charm_utils._open_sulci(
                label_img,
                label_affine,
                label_org_img,
                label_org_affine,
                m,
                exclusion_tissues=exclusion_tissues_open_csf,
            )
            label_nii = nib.Nifti1Image(label_img, label_affine)
            nib.save(label_nii, sub_files.tissue_labeling_upsampled)
            logger.info("从表面展开脑沟完成")
        else:
            logger.warning("左右半球 pial 表面均未重建。跳过从表面展开脑沟。")
    elapsed = time.time() - starttime
    logger.info("GM 改善总耗时 (HH:MM:SS):")
    logger.info(time.strftime("%H:%M:%S", time.gmtime(elapsed)))


def _load_freesurfer_pial_surface(fs_sub: file_finder.FreeSurferSubject) -> dict:
    """
    加载 FreeSurfer pial 表面（处理 symlink 问题）

    Parameters
    ----------
    fs_sub : FreeSurferSubject
        FreeSurfer subject 对象

    Returns
    -------
    dict
        表面字典
    """
    try:
        m = load_freesurfer_surfaces(fs_sub, "pial", coord="ras")
    except OSError:
        try:
            m = load_freesurfer_surfaces(fs_sub, "pial.T2", coord="ras")
        except FileNotFoundError:
            m = load_freesurfer_surfaces(fs_sub, "pial.T1", coord="ras")
    return m


def _get_atlas_settings(samseg_settings: dict) -> dict:
    """获取 Atlas 设置

    Parameters
    ----------
    samseg_settings : dict
        Samseg 设置

    Returns
    -------
    dict
        Atlas 设置
    """
    atlas_name = samseg_settings["atlas_name"]
    atlas_path = os.path.join(file_finder.templates.charm_atlas_path, atlas_name)
    return settings_reader.read_ini(os.path.join(atlas_path, atlas_name + ".ini"))
