import os
import shutil
import nibabel as nib
import nibabel.processing
import numpy as np
import numpy.typing as npt
from functools import partial
from scipy import ndimage
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
)
from scipy.ndimage import label
from scipy.io import loadmat

import samseg
from . import simnibs_segmentation_utils
from .samseg_whole_head import SamsegWholeHead
from .affine_whole_head import AffineWholeHead, initializationOptionsWholeHead
from ._thickness import _calc_thickness
from ._sanlm import sanlm
from .brain_surface import mask_from_surface
from simnibs.utils.simnibs_logger import logger
from simnibs.utils.transformations import resample_vol, volumetric_affine
from simnibs.utils.file_finder import SubjectFiles

from simnibs.mesh_tools.mesh_io import load_subject_surfaces


def _register_atlas_to_input_affine(
    T1,
    template_file_name,
    affine_mesh_collection_name,
    mesh_level1,
    mesh_level2,
    save_path,
    template_coregistered_name,
    init_atlas_settings,
    neck_tissues,
    visualizer,
    noneck,
    init_transform=None,
    world_to_world_transform_matrix=None,
    scaling_center=[0.0, -100.0, 0.0],
    k_values=[20.0, 10.0, 5.0],
    debug=False,
):
    # Import the affine registration function
    scales = init_atlas_settings["affine_scales"]
    thetas = init_atlas_settings["affine_rotations"]
    horizontal_shifts = init_atlas_settings["affine_horizontal_shifts"]
    vertical_shifts = init_atlas_settings["affine_vertical_shifts"]
    thetas_rad = [theta * np.pi / 180 for theta in thetas]
    neck_search_bounds = init_atlas_settings["neck_search_bounds"]
    ds_factor = init_atlas_settings["downsampling_factor_affine"]

    affine = AffineWholeHead(T1, affine_mesh_collection_name, template_file_name)

    init_options = initializationOptionsWholeHead(
        pitchAngles=thetas_rad,
        scales=scales,
        scalingCenter=scaling_center,
        horizontalTableShifts=horizontal_shifts,
        verticalTableShifts=vertical_shifts,
    )

    (
        image_to_image_transform,
        world_to_world_transform,
        optimization_summary,
    ) = affine.registerAtlasWholeHead(
        worldToWorldTransformMatrix=world_to_world_transform_matrix,
        initTransform=init_transform,
        initializationOptions=init_options,
        targetDownsampledVoxelSpacing=ds_factor,
        visualizer=visualizer,
        noneck=noneck,
        Ks=k_values,
    )

    affine.saveResultsWholeHead(
        T1,
        template_file_name,
        save_path,
        template_coregistered_name,
        image_to_image_transform,
        world_to_world_transform,
    )
    if world_to_world_transform_matrix is None:
        logger.info("Template registration summary.")
        logger.info(
            "Number of Iterations: %d, Cost: %f\n"
            % (optimization_summary["numberOfIterations"], optimization_summary["cost"])
        )

    if not noneck:
        logger.info("Adjusting neck.")
        exitcode = affine.adjust_neck(
            T1,
            template_coregistered_name,
            mesh_level1,
            mesh_level2,
            neck_search_bounds,
            neck_tissues,
            visualizer,
            debug,
            downsampling_target=2.0,
        )
        if exitcode == -1:
            file_path = os.path.split(template_coregistered_name)
            shutil.copy(mesh_level1, os.path.join(file_path[0], "atlas_level1.txt.gz"))
            shutil.copy(mesh_level2, os.path.join(file_path[0], "atlas_level2.txt.gz"))
        else:
            logger.info("Neck adjustment done.")
    else:
        logger.info("No neck, copying meshes over.")
        file_path = os.path.split(template_coregistered_name)
        shutil.copy(mesh_level1, os.path.join(file_path[0], "atlas_level1.txt.gz"))
        shutil.copy(mesh_level2, os.path.join(file_path[0], "atlas_level2.txt.gz"))


def _denoise_input_and_save(input_name, output_name):
    input_raw = nib.load(input_name)
    img = input_raw.get_fdata()
    # Sometimes the images have an extra unit dimension,
    # squeeze that out if it's there.
    img = img.squeeze()
    img_smoothed = sanlm(img, 3, 1)
    output_smoothed = nib.Nifti1Image(img_smoothed, input_raw.affine)
    nib.save(output_smoothed, output_name)


def _init_atlas_affine(t1_scan, mni_template, affine_settings):
    registerer = samseg.gems.KvlAffineRegistration(
        affine_settings["translation_scale"],
        affine_settings["max_iter"],
        affine_settings["num_histogram_bins"],
        affine_settings["shrink_factors"],
        affine_settings["bg_value"],
        affine_settings["smoothing_factors"],
        affine_settings["center_of_mass"],
        affine_settings["samp_factor"],
        "b",
    )
    registerer.read_images(t1_scan, mni_template)
    registerer.initialize_transform()
    registerer.register()
    trans_mat = registerer.get_transformation_matrix()
    # registerer.write_out_result(os.path.join(path_to_segment_folder, 'mni_transformed.nii.gz'))
    logger.info(trans_mat)
    # ITK returns the matrix mapping the fixed image to the
    # moving image so let's invert it.
    return np.linalg.inv(trans_mat)


def _estimate_parameters(
    path_to_segment_folder,
    template_coregistered_name,
    path_to_atlas_folder,
    input_images,
    segment_settings,
    gmm_parameters,
    visualizer,
    user_optimization_options=None,
    user_model_specifications=None,
    parameter_filename=None,
):
    ds_targets = segment_settings["downsampling_targets"]
    kernel_size = segment_settings["bias_kernel_width"]
    bg_mask_sigma = segment_settings["background_mask_sigma"]
    bg_mask_th = segment_settings["background_mask_threshold"]
    stiffness = segment_settings["mesh_stiffness"]
    covariances = segment_settings["diagonal_covariances"]
    shared_gmm_parameters = gmm_parameters
    # shared_gmm_parameters = samseg.io.kvlReadSharedGMMParameters(gmm_parameters)

    if user_optimization_options is None:
        user_optimization_options = {
            "multiResolutionSpecification": [
                {
                    "atlasFileName": os.path.join(
                        path_to_segment_folder, "atlas_level1.txt.gz"
                    ),
                    "targetDownsampledVoxelSpacing": ds_targets[0],
                    "maximumNumberOfIterations": 100,
                    "estimateBiasField": True,
                },
                {
                    "atlasFileName": os.path.join(
                        path_to_segment_folder, "atlas_level2.txt.gz"
                    ),
                    "targetDownsampledVoxelSpacing": ds_targets[1],
                    "maximumNumberOfIterations": 100,
                    "estimateBiasField": True,
                },
            ]
        }

    if user_model_specifications is None:
        user_model_specifications = {
            "atlasFileName": os.path.join(
                path_to_segment_folder, "atlas_level2.txt.gz"
            ),
            "biasFieldSmoothingKernelSize": kernel_size,
            "brainMaskingSmoothingSigma": bg_mask_sigma,
            "brainMaskingThreshold": bg_mask_th,
            "K": stiffness,
            "useDiagonalCovarianceMatrices": covariances,
            # "sharedGMMParameters": shared_gmm_parameters,
            "sharedGMMParameters": samseg.io.kvlReadSharedGMMParameters(gmm_parameters),
        }

    samseg_kwargs = dict(
        imageFileNames=input_images,
        atlasDir=path_to_atlas_folder,
        savePath=path_to_segment_folder,
        transformedTemplateFileName=template_coregistered_name,
        userModelSpecifications=user_model_specifications,
        userOptimizationOptions=user_optimization_options,
        imageToImageTransformMatrix=None,
        visualizer=visualizer,
        saveHistory=False,
        saveMesh=False,
        savePosteriors=False,
        saveWarp=True,
    )

    logger.info("Starting segmentation.")
    samsegment = SamsegWholeHead(**samseg_kwargs)
    samsegment.preProcessWholeHead()
    samsegment.fitModel()

    # Print optimization summary
    optimizationSummary = samsegment.getOptimizationSummary()
    for multiResolutionLevel, item in enumerate(optimizationSummary):
        logger.info(
            "atlasRegistrationLevel%d %d %f\n"
            % (multiResolutionLevel, item["numberOfIterations"], item["perVoxelCost"])
        )

    return samsegment.saveParametersAndInput(parameter_filename)


def _post_process_segmentation(
    bias_corrected_image_names,
    upsampled_image_names,
    tissue_settings,
    csf_factor,
    parameters_and_inputs,
    transformed_template_name,
    affine_atlas,
    before_morpho_name,
    upper_mask,
    debug=False,
):
    logger.info("Upsampling bias corrected images.")
    for input_number, bias_corrected in enumerate(bias_corrected_image_names):
        corrected_input = nib.load(bias_corrected)
        resampled_input, new_affine, orig_res = resample_vol(
            corrected_input.get_fdata(), corrected_input.affine, 0.5, order=1
        )
        upsampled = nib.Nifti1Image(resampled_input, new_affine)
        nib.save(upsampled, upsampled_image_names[input_number])

    # Next we need to reconstruct the segmentation with the upsampled data
    # and map it into the simnibs tissues
    upsampled_tissues, upper_part = simnibs_segmentation_utils.segmentUpsampled(
        upsampled_image_names,
        tissue_settings,
        parameters_and_inputs,
        transformed_template_name,
        affine_atlas,
        csf_factor,
    )

    del parameters_and_inputs
    # Cast the upsampled image to int16  to save space
    for upsampled_image in upsampled_image_names:
        upsampled = nib.load(upsampled_image)
        upsampled.set_data_dtype(np.int16)
        nib.save(upsampled, upsampled_image)

    affine_upsampled = upsampled.affine
    if debug:
        upsampled_tissues_im = nib.Nifti1Image(upsampled_tissues, affine_upsampled)
        nib.save(upsampled_tissues_im, before_morpho_name)

    upper_part_im = nib.Nifti1Image(upper_part.astype(np.int16), affine_upsampled)
    upper_part_im.set_data_dtype(np.int16)
    nib.save(upper_part_im, upper_mask)
    del upper_part_im
    # Do morphological operations
    simnibs_tissues = tissue_settings["simnibs_tissues"]
    upsampled_tissues = _morphological_operations(
        upsampled_tissues, upper_part, simnibs_tissues
    )

    return upsampled_tissues


def _clean_brain(label_img, tissues, unass, se, se_n, vol_limit=10):
    # Do some clean-ups, mainly CSF and skull
    # First combine the WM and GM
    brain = (label_img == tissues["WM"]) | (label_img == tissues["GM"])
    dil = binary_dilation(
        _get_largest_components(binary_erosion(brain, se_n, 1), se, vol_limit),
        se,
        1,
    )

    unass |= brain ^ dil

    # Add the CSF and open again
    csf = label_img == tissues["CSF"]
    brain_csf = brain | csf
    # Vol limit in voxels
    dil = binary_dilation(
        _get_largest_components(binary_erosion(brain_csf, se, 1), se_n, vol_limit=80),
        se,
        1,
    )
    unass |= csf & ~dil
    del brain, csf, dil
    return brain_csf


def _get_skull(label_img, brain_csf, tissues, se_n, se, num_iter=2, vol_limit=50):
    # Clean the outer border of the skull
    bone = (label_img == tissues["Compact_bone"]) | (
        label_img == tissues["Spongy_bone"]
    )
    veins = label_img == tissues["Blood"]
    air_pockets = label_img == tissues["Air_pockets"]
    scalp = label_img == tissues["Scalp"]
    muscle = label_img == tissues["Muscle"]
    eyes = label_img == tissues["Eyes"]
    # Use scalp to clean out noisy skull bits within the scalp
    skull_outer = brain_csf | bone | veins | air_pockets
    skull_outer = binary_fill_holes(skull_outer, se_n)
    skull_outer = binary_dilation(
        _get_largest_components(
            binary_erosion(skull_outer, se, num_iter), se_n, vol_limit
        ),
        se,
        num_iter,
    )
    skull_inner = bone | scalp | air_pockets | muscle | eyes
    skull_inner = binary_fill_holes(skull_inner, se_n)
    skull_inner = binary_dilation(
        _get_largest_components(
            binary_erosion(skull_inner, se, num_iter), se_n, vol_limit
        ),
        se,
        num_iter,
    )

    dil = bone & skull_outer & skull_inner
    return bone, skull_outer, dil


def _clean_veins(label_img, unass, tissues, se, se_n, num_iter=1, vol_limit=10):
    # Open the veins
    veins = label_img == tissues["Blood"]
    dil = binary_dilation(
        _get_largest_components(binary_erosion(veins, se, num_iter), se_n, vol_limit),
        se,
        num_iter,
    )
    unass |= dil ^ veins


def _clean_eyes(label_img, unass, tissues, se, se_n, num_iter=1, vol_limit=10):
    # Clean the eyes
    eyes = label_img == tissues["Eyes"]
    dil = binary_dilation(
        _get_largest_components(binary_erosion(eyes, se, num_iter), se_n, vol_limit),
        se,
        num_iter,
    )
    # dil = binary_opening(eyes, se, 1)
    unass |= dil ^ eyes


def _clean_muscles(label_img, unass, tissues, se, num_iter=1):
    # Clean muscles
    muscle = label_img == tissues["Muscle"]
    dil = binary_opening(muscle, se, num_iter)
    unass |= dil ^ muscle


def _clean_scalp(
    label_img, unass, skull_outer, tissues, se, se_n, num_iter=2, num_limit=1
):
    # And finally the scalp
    scalp = label_img == tissues["Scalp"]
    eyes = label_img == tissues["Eyes"]
    muscle = label_img == tissues["Muscle"]
    head = scalp | skull_outer | eyes | muscle
    dil = binary_dilation(
        _get_largest_components(binary_erosion(head, se, num_iter), se_n, num_limit),
        se,
        num_iter,
    )
    unass |= scalp & ~dil


def _ensure_csf(label_img, tissues, upper_part, se, num_iter1=1, num_iter2=6):
    # Ensuring a CSF layer between GM and Skull and GM and Blood
    # Relabel regions in the expanded GM which are in skull or blood to CSF
    brain_gm = label_img == tissues["GM"]
    C_BONE = label_img == tissues["Compact_bone"]
    S_BONE = label_img == tissues["Spongy_bone"]
    U_SKIN = (label_img == tissues["Scalp"]) & upper_part

    brain_dilated = binary_dilation(brain_gm, se, num_iter1)
    overlap = brain_dilated & (C_BONE | S_BONE | U_SKIN)
    label_img[overlap] = tissues["CSF"]

    S_BONE = label_img == tissues["Spongy_bone"]

    CSF_brain = brain_gm | (label_img == tissues["CSF"])
    CSF_brain_dilated = binary_dilation(CSF_brain, se, num_iter1)
    spongy_csf = CSF_brain_dilated & S_BONE
    upper_part = binary_erosion(upper_part, se, num_iter2)
    skin_csf = CSF_brain_dilated & (label_img == tissues["Scalp"]) & upper_part
    label_img[spongy_csf] = tissues["Compact_bone"]
    label_img[skin_csf] = tissues["Compact_bone"]


def _ensure_skull(label_img, tissues, se, num_iter=1):
    # Ensure the outer skull label is compact bone
    C_BONE = label_img == tissues["Compact_bone"]
    S_BONE = label_img == tissues["Spongy_bone"]
    SKULL_outer = (C_BONE | S_BONE) & ~binary_erosion((C_BONE | S_BONE), se, num_iter)
    label_img[SKULL_outer] = tissues["Compact_bone"]
    # Relabel air pockets to air
    label_img[label_img == tissues["Air_pockets"]] = 0


def _morphological_operations(label_img, upper_part, simnibs_tissues):
    """Does morphological operations to
    1. Smooth out the labeling and remove noise
    2. A CSF layer between GM and Skull and between GM and CSF
    3. Outer bone layers are compact bone
    """
    se = ndimage.generate_binary_structure(3, 3)
    se_n = ndimage.generate_binary_structure(3, 1)
    unass = np.zeros_like(label_img) > 0
    brain_csf = _clean_brain(label_img, simnibs_tissues, unass, se, se_n)
    bone, skull_outer, dil = _get_skull(label_img, brain_csf, simnibs_tissues, se_n, se)
    # Protect thin areas that would be removed by erosion
    bone_thickness = _calc_thickness(dil)

    thin_parts = bone & (bone_thickness < 3.5) & (bone_thickness > 0)
    dil |= thin_parts
    del thin_parts
    del bone_thickness

    unass = unass | (dil ^ bone)
    del bone, dil

    logger.info("Cleaning tissues")
    _clean_veins(label_img, unass, simnibs_tissues, se, se_n)
    _clean_eyes(label_img, unass, simnibs_tissues, se, se_n)
    _clean_muscles(label_img, unass, simnibs_tissues, se)
    _clean_scalp(label_img, unass, skull_outer, simnibs_tissues, se, se_n)

    logger.info("Labeling unassigned voxels")
    # Filling missing parts
    # The labeling is uint16, so code the unassigned voxels with
    #   (2**16 - 1) - 1 = 65534
    label_unassign = 65534
    label_img[unass] = label_unassign
    # Add background to tissues
    simnibs_tissues["BG"] = 0
    label_img = label_unassigned_elements(
        label_img, label_unassign, list(simnibs_tissues.values()) + [label_unassign]
    )

    # old way of labeling unassigned voxels. `label_unassigned_elements` is
    # faster and tend to preserve smaller structures
    # _smoothfill(label_img, unass, simnibs_tissues)

    logger.info("Ensuring CSF between gray matter and skull/blood")
    _ensure_csf(label_img, simnibs_tissues, upper_part, se)

    logger.info("Ensuring that the outer skull is compact bone")
    _ensure_skull(label_img, simnibs_tissues, se)

    return label_img


def generate_gaussian_kernel(i, ndim, sigma=1, zero_center=False):
    """Generate a gaussian kernel of size `i` in `ndim` dimensions."""
    assert i % 2 == 1, "Window size must be odd"
    center = ndim * (np.floor(i / 2).astype(int),)
    kernel = np.zeros(ndim * (i,), dtype=np.float32)
    kernel[center] = 1
    kernel = gaussian_filter(kernel, sigma)
    kernel /= kernel[center]
    if zero_center:
        kernel[center] = 0
    return kernel


def label_unassigned_elements(
    label_arr, label_unassign, labels=None, window_size=3, ignore_labels=None
) -> np.ndarray:
    """Label unassigned elements in `label_arr`. For each unassigned element,
    find its neighbors within a certain `window_size`, weigh these according
    to their euclidean distance (Gaussian kernel) and assign the label with the
    highest weight.

    PARAMETERS
    ----------
    label_arr : ndarray
        The array
    label_unassign : int
        The label of the unassigned elements.
    labels : array-like | None
        The labels in `label_arr`. If None (default), will be inferred from the
        array using `np.unique`.
    window_size : int
        The size of the neighborhood around each element to consider when
        relabeling.
    ignore_labels : array-like | None
        Do not use these labels when labeling unassigned elements (default =
        None).

    RETURNS
    -------
    labeling : ndarray
        The relabeled array.
    """

    # Finding the indices of the unassigned voxels is slow with fortran ordered
    # arrays
    labeling = label_arr.T if label_arr.flags["F_CONTIGUOUS"] else label_arr

    # Weighting kernel and related
    pad_width = np.floor(window_size / 2).astype(int)
    kernel = generate_gaussian_kernel(window_size, labeling.ndim, zero_center=True)
    window = kernel.shape
    kernel = kernel.ravel()

    # Find the list of label (if necessary) and ensure it is unique
    labels = np.unique(labeling) if labels is None else np.unique(labels)
    labels = labels.astype(labeling.dtype)
    n_labels = labels.size
    assert label_unassign in labels, "`label_unassign` is not in the provided `labels`!"

    # Ignore desired labels and the unassigned label
    if ignore_labels is None:
        ignore_labels = [label_unassign]
    else:
        assert all(i in labels for i in ignore_labels)
        ignore_labels = ignore_labels.copy()
        ignore_labels = ignore_labels + [label_unassign]  # avoid inplace

    # Ensure continuous labels (e.g.,, [0, 1, 2, 3], not [0, 1, 10, 15])
    is_continuous_labels = labels[-1] - labels[0] + 1 == n_labels
    if is_continuous_labels:
        continuous_labels = labels
        mapped_ignore_labels = np.array(ignore_labels)
        labeling = labeling.copy()  # ensure that we do not modify the input
    else:
        continuous_labels = np.arange(n_labels)
        mapper = np.zeros(labels[-1] + 1, dtype=labeling.dtype)
        mapper[labels] = continuous_labels
        labeling = mapper[labeling]
        mapped_ignore_labels = mapper[ignore_labels]

    is_unassign = np.nonzero(labeling == mapped_ignore_labels[-1])
    while (n_unassign := is_unassign[0].size) > 0:
        # print("Number of unassigned voxels:", n_unassign)
        labeling_view = np.lib.stride_tricks.sliding_window_view(
            np.pad(labeling, pad_width, "symmetric"), window
        )
        unassign_window = labeling_view[is_unassign].reshape(-1, kernel.size)

        # Compute weights in a loop to save memory
        weights = np.array(
            [
                (
                    np.zeros(n_unassign)
                    if i in mapped_ignore_labels
                    else (unassign_window == i) @ kernel
                )
                for i in continuous_labels
            ]
        )
        # The slightly faster but less memory-friendly solution
        # n_total = unassign_window.size
        # oh_enc = np.zeros((n_labels, n_total), dtype=np.uint8)
        # oh_enc[unassign_window.ravel(), np.arange(n_total)] = 1
        # oh_enc = oh_enc.reshape(n_labels, *unassign_window.shape)
        # weights = oh_enc @ kernel
        # if mapped_ignore_labels.size > 0:
        #     weights[mapped_ignore_labels] = 0

        # In the case of ties, the lowest index is returned. This is arbitrary
        # but the default behavior of argmax
        new_label = weights.argmax(0)
        valid = weights.sum(0) > 0
        invalid = ~valid

        labeling[tuple(i[valid] for i in is_unassign)] = new_label[valid]
        is_unassign = tuple(i[invalid] for i in is_unassign)

        if is_unassign[0].size == n_unassign:
            logger.warning(
                "Some elements could not be labeled (probably because they are surrounded by labels in `ignore_labels`)"
            )
            # perhaps we may want to issue the warning using the warnings
            # module instead
            # warnings.warn(
            #     "Some elements could not be labeled, probably because they are surrounded by labels in `ignore_labels`",
            #     RuntimeWarning,
            # )
            break

    # Revert array
    if not is_continuous_labels:
        labeling = labels[labeling]
    labeling = labeling.T if label_arr.flags["F_CONTIGUOUS"] else labeling

    return labeling


def _smooth(label_img, simnibs_tissues, tissues_to_smooth):
    """Smooth some of the tissues for "nicer" tissue labelings."""
    labs = label_img.copy()
    max_val = np.zeros_like(label_img, dtype=np.float32)
    for i, t in enumerate(simnibs_tissues):
        vol = (label_img == simnibs_tissues[t]).astype(np.float32)
        if t in tissues_to_smooth:
            cs = gaussian_filter(vol, 1)
        else:
            cs = vol

        # Check the max values and update
        max_mask = cs > max_val
        labs[max_mask] = tissues_to_smooth[t]
        max_val[max_mask] = cs[max_mask]

    label_img[:] = labs[:]
    del labs


def _smoothfill(label_img, unassign, simnibs_tissues):
    """Hackish way to fill unassigned voxels,
    works by smoothing the masks and binarizing
    the smoothed masks.
    """
    sum_of_unassigned = np.inf
    # for as long as the number of unassigned is changing
    # let's find the binarized version using a running
    # max index as we need to loop anyway. This way we
    # don't need to call np.argmax
    while unassign.sum() < sum_of_unassigned:
        sum_of_unassigned = unassign.sum()
        if sum_of_unassigned == 0:
            break
        labs = 65535 * np.ones_like(label_img)
        max_val = np.zeros_like(label_img, dtype=np.float32)
        for i, t in enumerate(simnibs_tissues):
            # Don't smooth WM
            vol = (label_img == simnibs_tissues[t]).astype(np.float32)
            if t == "WM":
                cs = vol
            else:
                cs = gaussian_filter(vol, 1)

            # Check the max values and update
            max_mask = cs > max_val
            labs[max_mask] = simnibs_tissues[t]
            max_val[max_mask] = cs[max_mask]

        label_img[:] = labs[:]
        unassign = labs == 65535
        del labs


def _fill_missing(label_img, unassign):
    """Hackish way to fill unassigned voxels,
    works by smoothing the masks and binarizing
    the smoothed masks. Works in-place
    """

    def _get_most_frequent_label(x, y, z, label_im, pad):
        nhood = label_im[
            max(0, x - pad) : min(label_im.shape[0], x + pad),
            max(0, y - pad) : min(label_im.shape[1], y + pad),
            max(0, z - pad) : min(label_im.shape[2], z + pad),
        ]

        dim = nhood.shape

        return np.bincount(nhood.reshape((np.prod(dim)))).argmax()

    label_img[unassign] = 255
    inds_tmp = unassign.nonzero()
    inds_tmp = np.array(inds_tmp)
    num_unassigned = (label_img == 255).sum()
    num_unassigned_new = num_unassigned
    while num_unassigned_new > 0:
        fill_map = map(
            partial(_get_most_frequent_label, label_im=label_img, pad=5),
            np.nditer(inds_tmp[0, :]),
            np.nditer(inds_tmp[1, :]),
            np.nditer(inds_tmp[2, :]),
        )
        fill_array = np.array(list(fill_map))
        label_img[inds_tmp[0, :], inds_tmp[1, :], inds_tmp[2, :]] = fill_array
        inds_tmp = inds_tmp[:, fill_array == 255]
        num_unassigned_new = (label_img == 255).sum()
        logger.info("Unassigned: " + str(num_unassigned_new))
        if num_unassigned_new == num_unassigned:
            logger.info("Number of unassigned voxels not going down. Breaking.")
            break

        num_unassigned = num_unassigned_new


def _get_largest_components(vol, se, vol_limit=0, num_limit=-1, return_sizes=False):
    """Get the n largest components from a volume.

    PARAMETERS
    ----------
    vol : ndarray
        Image volume. A dimX x dimY x dimZ array containing the image data.
    se : ndarray
        Structuring element to use when detecting components (i.e. setting the
        connectivity which defines a component).
    n : int
        Number of (largest) components to retain.
    return_sizes : bool, optional
        Whether or not to also return the sizes (in voxels) of each component
        that was retained (default = False).

    RETURNS
    ----------
    Components : ndarray
        Binary dimX x dimY x dimZ array where entries corresponding to retained
        components are True and the remaining entries are False.
    """
    vol_lbl = label(vol, se)[0]
    labels, region_size = np.unique(vol_lbl, return_counts=True)
    labels = labels[1:]  # disregard background (label=0)
    region_size = region_size[1:]  #
    mask = region_size > vol_limit
    region_size = region_size[mask]
    labels = labels[mask]

    if num_limit == -1:
        num_limit = len(labels)

    labels = labels[np.argsort(region_size)[::-1]]
    components = np.zeros_like(vol, dtype=bool)
    for i in labels[:num_limit]:
        components = components | (vol_lbl == i)

    if return_sizes:
        return components, region_size[labels[:num_limit]]
    else:
        return components


def _registerT1T2(fixed_image, moving_image, output_image):
    registerer = samseg.gems.KvlRigidRegistration()
    # linear interpolation
    # registerer = samseg.gems.KvlRigidRegistration(
    #     translationScale=-100.,
    #     numberOfIterations=100,
    #     numberOfHistogramBins=50,
    #     shrinkScales=[2.0, 1.0, 0.0],
    #     backgroundGrayLevel=0.,
    #     smoothingSigma=[4.0, 2.0, 0.0],
    #     useCenterOfMassInitialization=False,
    #     samplingRate=0.5,
    #     interpolator="l",
    # )
    registerer.read_images(fixed_image, moving_image)
    registerer.initialize_transform()
    registerer.register()
    registerer.write_out_result(output_image)

    # The register function uses double internally
    # Let's cast to float32 and copy the header from
    # the fixed image to avoid any weird behaviour
    if os.path.exists(output_image):
        T2_reg = nib.load(output_image)
        fixed_tmp = nib.load(fixed_image)
        T2_data = T2_reg.get_fdata().astype(np.float32)
        T2_im = nib.Nifti1Image(T2_data, fixed_tmp.affine)
        nib.save(T2_im, output_image)


def read_freesurfer_lut(filename):
    value, label, r, g, b, a = np.loadtxt(filename, dtype=str, unpack=True)
    value = value.astype(int)
    ctab = np.stack((r, g, b, a), axis=1).astype(int)
    return value, label, ctab


def update_labeling_from_cortical_surfaces(
    m2m: SubjectFiles,
    protect: dict[str, list],
    tissue_mapping: dict[str, int],
):
    """

    Generate a gray matter mask based on the estimated pial surfaces and use
    this to update the gray matter segmentation by relabeling to CSF all voxels
    labeled as gray matter but which are not inside the pial surface.

    To avoid relabeling subcortical structures and cerebellum, we add these to
    the gray matter mask as well.


    Parameters
    ----------
    m2m : SubjectFiles
        SubjectFiles object of the subject directory.
    protect : dict
        The following keys are required (as strings)
            gm_to_csf
                Tissues to protect when relabeling gray matter outside pial
                surface to CSF
            wm_to_gm
                Tissues to protect when relabeling white matter outside white
                matter surface to gray matter.
            gm_to_wm
                Tissues to protect when relabeling gray matter inside white
                matter surface to white matter.
            csf_to_gm
                Tissues to protect when relabeling CSF inside pial surface to
                gray matter.
    tissue_mapping : dict
        The mapping of SimNIBS tissue names to numbers.
    """

    # SimNIBS label image (to update)
    sm_label_img = nib.load(m2m.tissue_labeling_upsampled)
    sm_label_data = np.asanyarray(sm_label_img.dataobj)

    # full SAMSEG labeling image (used to protect)
    fs_label_img = nib.load(m2m.labeling)
    fs_label_img = nibabel.processing.resample_from_to(
        fs_label_img, sm_label_img, order=0
    )
    fs_label_data = np.asanyarray(fs_label_img.dataobj)

    # Rasterize surface masks
    white = load_subject_surfaces(m2m, "white")
    white = white["lh"].join_mesh(white["rh"])
    white_surf_mask = mask_from_surface(
        white.nodes[:], white.elm[:, :3] - 1, sm_label_img.affine, sm_label_img.shape
    )
    pial = load_subject_surfaces(m2m, "pial")
    pial = pial["lh"].join_mesh(pial["rh"])
    pial_surf_mask = mask_from_surface(
        pial.nodes[:], pial.elm[:, :3] - 1, sm_label_img.affine, sm_label_img.shape
    )

    # Update using morphological operations
    sm_label_data = update_labeling_from_cortical_surfaces_(
        sm_label_data,
        fs_label_data,
        white_surf_mask,
        pial_surf_mask,
        protect,
        tissue_mapping,
    )

    # overwrite original image
    sm_label_img_updated = nib.Nifti1Image(sm_label_data, sm_label_img.affine)
    sm_label_img_updated.to_filename(m2m.tissue_labeling_upsampled)


def update_labeling_from_cortical_surfaces_(
    label_image: npt.NDArray,
    protect_labeling: npt.NDArray,
    white_surf_mask: npt.NDArray,
    pial_surf_mask: npt.NDArray,
    protect: dict[str, list[int]],
    tissue_mapping: dict[str, int],
):
    """The behavior of this function for WM and GM in bone, skin etc. is not
    well defined.

    Parameters
    ----------
    label_image
        Label image to be corrected.
    protect_labeling
        Label image from which areas to be "protected" are derived.
    white_surf_mask:
        Mask where all voxels inside the white matter surface is True.
    pial_surf_mask:
        Mask where all voxels inside the gray matter surface (also incl. those
        inside the white_surf_mask) is True.
    protect:
        Mapping of tissues to protect (labels refer to those in
        `protect_labeling`).
    tissue_mapping:
        Mapping of tissue names to tissue labels where the labels refer to
        those in `label_image`.

    Returns
    -------
    The corrected labeling image.
    """

    ndim = label_image.ndim

    # (1) Build masks

    # cortical WM mask
    white_surf_mask = binary_opening(white_surf_mask, iterations=1)

    wm_extra_mask = np.isin(protect_labeling, protect["wm_to_gm"])
    wm_extra_mask = binary_dilation(wm_extra_mask, iterations=2)

    # cortical GM mask
    pial_surf_mask = binary_opening(pial_surf_mask, iterations=1)

    gm_extra_mask = np.isin(protect_labeling, protect["gm_to_csf"])
    gm_extra_mask = binary_dilation(gm_extra_mask, iterations=2)

    # relabel GM inside of white surfaces to WM
    # structures to protect from becoming WM (subcortical structures)
    wm_ignore_mask = np.isin(protect_labeling, protect["gm_to_wm"])
    wm_ignore_mask = binary_dilation(wm_ignore_mask, iterations=2)

    # Ensure a layer of CSF
    bone = np.isin(
        label_image, (tissue_mapping["Compact_bone"], tissue_mapping["Spongy_bone"])
    )
    se = ndimage.generate_binary_structure(ndim, ndim)
    dilated_bone = binary_dilation(bone, se)
    csf_blood_mask_shrunk = (
        np.isin(label_image, (tissue_mapping["CSF"], tissue_mapping["Blood"]))
        & ~dilated_bone
    )
    csf_ignore_mask = np.isin(protect_labeling, protect["csf_to_gm"])

    # (2) Apply
    # Mind the order in which the steps are applied!

    # Cortical white matter
    # (1) GM (inside WM surface but not subcortical) to WM
    replace = (label_image == tissue_mapping["GM"]) & white_surf_mask & ~wm_ignore_mask
    label_image[replace] = tissue_mapping["WM"]
    # (2) CSF/blood (inside WM surface but not subcortical nor ventricles) to WM
    replace = (
        csf_blood_mask_shrunk & white_surf_mask & ~csf_ignore_mask & ~wm_ignore_mask
    )
    label_image[replace] = tissue_mapping["WM"]
    # (3) WM (outside WM surface but not cerebellar WM, brainstem etc.) to GM
    replace = (label_image == tissue_mapping["WM"]) & ~(white_surf_mask | wm_extra_mask)
    label_image[replace] = tissue_mapping["GM"]

    # Cortical gray matter

    # (1) CSF/blood (inside GM surface but not inside WM surface nor venctricles) to GM
    replace = (
        csf_blood_mask_shrunk & pial_surf_mask & ~white_surf_mask & ~csf_ignore_mask
    )
    label_image[replace] = tissue_mapping["GM"]
    # (2) GM (outside GM surface but not subcortical, cerebellar GM etc.) to CSF
    replace = (label_image == tissue_mapping["GM"]) & ~(pial_surf_mask | gm_extra_mask)
    label_image[replace] = tissue_mapping["CSF"]

    return label_image


def _cut_and_combine_labels(
    fn_tissue_labeling_upsampled, fn_mni_template, fn_affine, tms_settings, n_dil=40
):
    """
    Cut away neck of tissue_labeling_upsampled.nii.gz and
    combine some of the labels. Overwrites the original file.
    Used to create meshes optimized for TMS.

    Parameters
    ----------
    fn_tissue_labeling_upsampled : string
        filename of tissue_labeling_upsampled.nii.gz
    fn_mni_template : string
        filename of the MNI template
    fn_affine : string
        filename of the affine transformation from T1 to MNI
    tms_settings : dict
        specifies old and new labels

    Returns
    -------
    None.

    """
    # cut label image using MNI mask
    logger.info("Cutting neck region using MNI mask")
    label_image = nib.load(fn_tissue_labeling_upsampled)
    label_buffer = np.round(label_image.get_fdata()).astype(
        np.uint16
    )  # Cast to uint16, otherwise meshing complains
    label_affine = label_image.affine

    mni_image = nib.load(fn_mni_template)
    mni_buffer = np.ones(mni_image.shape, dtype=bool)
    mni_affine = mni_image.affine

    trafo = loadmat(fn_affine)["worldToWorldTransformMatrix"]

    upperhead = volumetric_affine(
        (mni_buffer, mni_affine),
        np.linalg.inv(trafo),
        target_space_affine=label_affine,
        target_dimensions=label_image.shape,
        intorder=0,
    )
    upperhead = binary_dilation(upperhead, iterations=n_dil)
    label_buffer[~upperhead] = 0

    # combine labels
    logger.info("Combining labels")
    for idx_old, idx_new in zip(tms_settings["old_label"], tms_settings["new_label"]):
        logger.debug("  old label: %d, new label: %d " % (idx_old, idx_new))
        label_buffer[label_buffer == idx_old] = idx_new

    label_image = nib.Nifti1Image(label_buffer, label_affine)
    nib.save(label_image, fn_tissue_labeling_upsampled)


# def _downsample_surface(m, n_nodes):
#     """
#         downsample a surface using meshfix

#     Parameters
#     ----------
#     m : simnibs.Msh
#         surface
#     n_nodes : int
#         target number of nodes.

#     Returns
#     -------
#     mout : simnibs.Msh
#         downsampled surface

#     NOTE: this is a primitive wrapper around meshfix, tag1 and tag2 of the
#     returned mesh will be set to 1, meshes with multiple surfaces are not
#     supported
#     """
#     with tempfile.NamedTemporaryFile(suffix=".off") as f:
#         mesh_fn = f.name
#     write_off(m, mesh_fn)
#     cmd = [file_finder.path2bin("meshfix"), mesh_fn,
#            "-u", "2", "--vertices", str(n_nodes), "-o", mesh_fn]
#     spawn_process(cmd, lvl=logging.DEBUG)

#     mout = read_off(mesh_fn)
#     os.remove(mesh_fn)
#     if os.path.isfile("meshfix_log.txt"):
#         os.remove("meshfix_log.txt")
#     return mout
