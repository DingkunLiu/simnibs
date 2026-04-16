"""
CHARM 步骤5: 体积与表面分割

执行分割、偏置场校正，并生成上采样的组织标记图像。

原理：
    1. 使用 samseg 进行参数估计和分割
    2. 输出偏置场校正后的图像
    3. 进行后处理（形态学操作）
    4. 生成上采样（1mm）的组织标签图像

输入：
    - segmentation/template_coregistered.nii.gz
    - T1fs.nii.gz / T2_reg.nii.gz (偏置校正输出)

输出：
    - segmentation/tissue_labeling_upsampled.nii.gz
    - segmentation/tissue_labeling_upsampled_LUT.txt

用法：
    python -m neuracle.charm.segment <subid> [--debug]
"""

import logging
import os
import shutil

import nibabel as nib

from neuracle.utils.charm_utils import read_settings, setup_atlas
from neuracle.utils.constants import N_WORKERS
from simnibs.segmentation import charm_utils
from simnibs.segmentation import samseg_whole_head as simnibs_samseg
from simnibs.segmentation import simnibs_segmentation_utils
from simnibs.utils import file_finder

logger = logging.getLogger(__name__)


def run_segmentation(
    subject_dir: str,
    debug: bool = False,
) -> None:
    """
    执行分割流程

    使用 samseg 进行参数估计和分割，执行偏置场校正，生成上采样的组织标记图像。
    包括：参数估计、分割输出、偏置场校正、MNI 变形场生成、后处理等步骤。

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
    setup_atlas : Atlas 路径和参数设置
    _estimate_parameters : 参数估计
    _post_process_segmentation : 分割后处理
    """
    sub_files = file_finder.SubjectFiles(subpath=subject_dir)
    settings = read_settings()
    samseg_settings = settings["samseg"]
    segment_settings = settings["segment"]
    num_threads = settings["general"]["threads"]
    if isinstance(num_threads, int) and num_threads > 0:
        simnibs_samseg.gems.setGlobalDefaultNumberOfThreads(num_threads)
        logger.info("使用 %d 个线程进行分割", num_threads)
    else:
        simnibs_samseg.gems.setGlobalDefaultNumberOfThreads(N_WORKERS)
        logger.info("线程数无效 (%s)，使用 N_WORKERS (%d)", num_threads, N_WORKERS)
    show_figs = False
    show_movies = False
    visualizer = simnibs_samseg.initVisualizer(show_figs, show_movies)
    (
        template_name,
        atlas_settings,
        atlas_path,
        atlas_level1,
        atlas_level2,
        atlas_affine_name,
        gmm_parameters,
    ) = setup_atlas(samseg_settings, sub_files.T2_reg, None)
    os.makedirs(sub_files.segmentation_folder, exist_ok=True)
    os.makedirs(sub_files.surface_folder, exist_ok=True)
    input_images = []
    if os.path.exists(sub_files.T1_denoised):
        input_images.append(sub_files.T1_denoised)
        logger.info("使用去噪后的 T1 图像: %s", sub_files.T1_denoised)
    else:
        input_images.append(sub_files.reference_volume)
        logger.info("使用参考 T1 图像: %s", sub_files.reference_volume)
    if os.path.exists(sub_files.T2_reg):
        if os.path.exists(sub_files.T2_reg_denoised):
            input_images.append(sub_files.T2_reg_denoised)
            logger.info("使用去噪后的 T2 图像: %s", sub_files.T2_reg_denoised)
        else:
            input_images.append(sub_files.T2_reg)
            logger.info("使用原始 T2 图像: %s", sub_files.T2_reg)
    logger.info("正在估计参数。")
    segment_parameters_and_inputs = charm_utils._estimate_parameters(
        sub_files.segmentation_folder,
        sub_files.template_coregistered,
        atlas_path,
        input_images,
        segment_settings,
        gmm_parameters,
        visualizer,
        parameter_filename=os.path.join(sub_files.segmentation_folder, "parameters.p")
        if debug
        else None,
    )
    bias_corrected_image_names = [sub_files.T1_bias_corrected]
    if len(input_images) > 1:
        bias_corrected_image_names.append(sub_files.T2_bias_corrected)
        logger.info("将使用 T2 图像进行偏置场校正")
    logger.info("正在输出归一化图像和标签。")
    tissue_settings = atlas_settings["conductivity_mapping"]
    csf_factor = segment_settings["csf_factor"]
    simnibs_segmentation_utils.writeBiasCorrectedImagesAndSegmentation(
        bias_corrected_image_names,
        sub_files.labeling,
        segment_parameters_and_inputs,
        tissue_settings,
        csf_factor,
        sub_files.template_coregistered,
    )
    fn_lut = sub_files.labeling.rsplit(".", 2)[0] + "_LUT.txt"
    shutil.copyfile(file_finder.templates.labeling_LUT, fn_lut)
    logger.info("正在输出 MNI 变形场。")
    os.makedirs(sub_files.mni_transf_folder, exist_ok=True)
    simnibs_segmentation_utils.saveWarpField(
        template_name,
        sub_files.mni2conf_nonl,
        sub_files.conf2mni_nonl,
        segment_parameters_and_inputs,
    )
    logger.info("正在对分割结果进行后处理")
    os.makedirs(sub_files.label_prep_folder, exist_ok=True)
    upsampled_image_names = [sub_files.T1_upsampled]
    if len(bias_corrected_image_names) > 1:
        upsampled_image_names.append(sub_files.T2_upsampled)
        logger.info("将上采样 T2 图像")
    cleaned_upsampled_tissues = charm_utils._post_process_segmentation(
        bias_corrected_image_names,
        upsampled_image_names,
        tissue_settings,
        csf_factor,
        segment_parameters_and_inputs,
        sub_files.template_coregistered,
        atlas_affine_name,
        sub_files.tissue_labeling_before_morpho,
        sub_files.upper_mask,
        debug=debug,
    )
    img_tmp = nib.load(sub_files.reference_volume)
    scode = img_tmp.get_sform(coded=True)[1]
    qcode = img_tmp.get_qform(coded=True)[1]
    upsampled_image = nib.load(sub_files.T1_upsampled)
    affine_upsampled = upsampled_image.affine
    upsampled_tissues = nib.Nifti1Image(cleaned_upsampled_tissues, affine_upsampled)
    upsampled_tissues.set_qform(affine_upsampled, qcode)
    upsampled_tissues.set_sform(affine_upsampled, scode)
    nib.save(upsampled_tissues, sub_files.tissue_labeling_upsampled)
    upsampled_image.set_qform(affine_upsampled, qcode)
    upsampled_image.set_sform(affine_upsampled, scode)
    nib.save(upsampled_tissues, sub_files.T1_upsampled)
    if len(bias_corrected_image_names) > 1:
        upsampled_image = nib.load(sub_files.T2_upsampled)
        upsampled_image.set_qform(affine_upsampled, qcode)
        upsampled_image.set_sform(affine_upsampled, scode)
        nib.save(upsampled_tissues, sub_files.T2_upsampled)
        logger.info("T2 上采样图像已保存")
    fn_lut = sub_files.tissue_labeling_upsampled.rsplit(".", 2)[0] + "_LUT.txt"
    shutil.copyfile(file_finder.templates.final_tissues_LUT, fn_lut)
    logger.info("分割完成")
