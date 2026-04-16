import samseg
from samseg import gems
from samseg.ProbabilisticAtlas import ProbabilisticAtlas
from samseg.SamsegUtility import writeImage, logTransform
from samseg.io import kvlReadCompressionLookupTable, kvlReadSharedGMMParameters
import os
from samseg.GMM import GMM
import numpy as np
import nibabel as nib
import gc
from samseg.utilities import requireNumpyArray


def getModelSpecificationsWholeHead(atlasDir, userModelSpecifications={}):
    # Create default model specifications as a dictionary
    if "FreeSurferLabels" not in userModelSpecifications.keys():
        FreeSurferLabels, names, colors = kvlReadCompressionLookupTable(
            os.path.join(atlasDir, "compressionLookupTable.txt")
        )
    else:
        FreeSurferLabels = None
        names = None
        colors = None
    if "sharedGMMParameters" not in userModelSpecifications.keys():
        sharedGMMParameters = kvlReadSharedGMMParameters(
            os.path.join(atlasDir, "sharedGMMParameters.txt")
        )
    else:
        sharedGMMParameters = None

    modelSpecifications = {
        "FreeSurferLabels": FreeSurferLabels,
        "atlasFileName": os.path.join(atlasDir, "atlas_level2.txt.gz"),
        "names": names,
        "colors": colors,
        "sharedGMMParameters": sharedGMMParameters,
        "useDiagonalCovarianceMatrices": True,
        "brainMaskingSmoothingSigma": 3.0,  # sqrt of the variance of a Gaussian blurring kernel
        "brainMaskingThreshold": 0.01,
        "K": 0.1,  # stiffness of the mesh
        "biasFieldSmoothingKernelSize": 50,  # distance in mm of sinc function center to first zero crossing
        "whiteMatterAndCortexSmoothingSigma": 0,  # Samseg requires this attribute
    }

    modelSpecifications.update(userModelSpecifications)
    return modelSpecifications


def readCroppedImages(imageFileNames, transformedTemplateFileName):
    # Read the image data from disk. At the same time, construct a 3-D affine transformation (i.e.,
    # translation, rotation, scaling, and skewing) as well - this transformation will later be used
    # to initially transform the location of the atlas mesh's nodes into the coordinate system of the image.
    imageBuffers = []
    print(imageFileNames)
    print(transformedTemplateFileName)
    for imageFileName in imageFileNames:
        # Get the pointers to image and the corresponding transform
        image = gems.KvlImage(imageFileName, transformedTemplateFileName)
        transform = image.transform_matrix
        cropping = image.crop_slices
        imageBuffers.append(image.getImageBuffer())

    imageBuffers = np.transpose(imageBuffers, axes=[1, 2, 3, 0])

    # Also read in the voxel spacing -- this is needed since we'll be specifying bias field smoothing kernels,
    # downsampling steps etc in mm.
    nonCroppedImage = gems.KvlImage(imageFileNames[0])
    imageToWorldTransformMatrix = nonCroppedImage.transform_matrix.as_numpy_array
    voxelSpacing = np.sum(imageToWorldTransformMatrix[0:3, 0:3] ** 2, axis=0) ** (1 / 2)

    #
    return imageBuffers, transform, voxelSpacing, cropping


def maskOutBackground(
    imageBuffers,
    imageFileNames,
    atlasFileName,
    transform,
    brainMaskingSmoothingSigma,
    brainMaskingThreshold,
    probabilisticAtlas,
    upsampled,
    visualizer=None,
    maskOutZeroIntensities=True,
):
    # Setup a null visualizer if necessary
    if visualizer is None:
        visualizer = samseg.initVisualizer(False, False)

    # Read the affinely coregistered atlas mesh (in reference position)
    mesh = probabilisticAtlas.getMesh(atlasFileName, transform)

    # Mask away uninteresting voxels. This is done by a poor man's implementation of a dilation operation on
    # a non-background class mask; followed by a cropping to the area covered by the mesh (needed because
    # otherwise there will be voxels in the data with prior probability zero of belonging to any class)
    imageSize = imageBuffers.shape[0:3]
    ##backgroundPrior = np.zeros(imageSize)
    ##labelNumber = [0, 54, 44, 46, 43, 51, 50, 42, 45, 49, 52, 48]
    ##for l in labelNumber:
    ##    backgroundPrior = backgroundPrior + np.float32(mesh.rasterize_1a(imageSize, l))
    labelNumber = 0
    backgroundPrior = mesh.rasterize_1a(imageSize, labelNumber)
    # Threshold background prior at 0.5 - this helps for atlases built from imperfect (i.e., automatic)
    # segmentations, whereas background areas don't have zero probability for non-background structures
    backGroundThreshold = 2**8
    backGroundPeak = 2**16 - 1
    backgroundPrior = np.ma.filled(
        np.ma.masked_greater(backgroundPrior, backGroundThreshold), backGroundPeak
    ).astype(np.float32)

    visualizer.show(
        probabilities=backgroundPrior,
        images=imageBuffers,
        window_id="samsegment background",
        title="Background Priors",
    )

    # brainMaskingSmoothingSigma is the appropriate smoothing factor for a 1mm^3 scan, 
    # which should be scaled according to the actual resolution of the scan
    if not upsampled:
        # estimate appropriate factor for smoothing sigma for  voxel space of T1
        voxelSize = np.array(nib.load(imageFileNames[0]).header.get_zooms())
        # voxelSize = np.mean(np.abs(voxelDims))
    else:
        # the voxel size of the upsampled scan is .5 mm
        voxelSize = np.array([.5, .5, .5])

    
    smoothingSigmas = 1.0 / voxelSize * brainMaskingSmoothingSigma
    smoothedBackgroundPrior = gems.KvlImage.smooth_image_buffer(
        backgroundPrior, smoothingSigmas
    )
    visualizer.show(
        probabilities=smoothedBackgroundPrior,
        window_id="samsegment smoothed",
        title="Smoothed Background Priors",
    )

    # 65535 = 2^16 - 1. priors are stored as 16bit ints
    # To put the threshold in perspective: for Gaussian smoothing with a 3D isotropic kernel with variance
    # diag( sigma^2, sigma^2, sigma^2 ) a single binary "on" voxel at distance sigma results in a value of
    # 1/( sqrt(2*pi)*sigma )^3 * exp( -1/2 ).
    # More generally, a single binary "on" voxel at some Eucledian distance d results in a value of
    # 1/( sqrt(2*pi)*sigma )^3 * exp( -1/2*d^2/sigma^2 ). Turning this around, if we threshold this at some
    # value "t", a single binary "on" voxel will cause every voxel within Eucledian distance
    #
    #   d = sqrt( -2*log( t * ( sqrt(2*pi)*sigma )^3 ) * sigma^2 )
    #
    # of it to be included in the mask.
    #
    # As an example, for 1mm isotropic data, the choice of sigma=3 and t=0.01 yields ... complex value ->
    # actually a single "on" voxel will then not make any voxel survive, as the normalizing constant (achieved
    # at Mahalanobis distance zero) is already < 0.01
    brainMaskThreshold = 65535.0 * (1.0 - brainMaskingThreshold)
    brainMask = np.ma.less(smoothedBackgroundPrior, brainMaskThreshold)

    # Crop to area covered by the mesh
    alphas = mesh.alphas
    areaCoveredAlphas = [[0.0, 1.0]] * alphas.shape[0]
    mesh.alphas = areaCoveredAlphas  # temporary replacement of alphas
    areaCoveredByMesh = mesh.rasterize_1b(imageSize, 1)
    mesh.alphas = alphas  # restore alphas
    brainMask = np.logical_and(brainMask, areaCoveredByMesh)

    # If a pixel has a zero intensity in any of the contrasts, that is also masked out across all contrasts
    if maskOutZeroIntensities:
        numberOfContrasts = imageBuffers.shape[-1]
        for contrastNumber in range(numberOfContrasts):
            brainMask *= imageBuffers[:, :, :, contrastNumber] > 0

    # Mask the images
    maskedImageBuffers = imageBuffers.copy()
    maskedImageBuffers[np.logical_not(brainMask), :] = 0

    #
    return maskedImageBuffers, brainMask


def writeBiasCorrectedImagesAndSegmentation(
    output_names_bias,
    output_name_segmentation,
    parameters_and_inputs,
    tissue_settings,
    csf_factor,
    template_coregistered
):
    # We need an init of the probabilistic segmentation class
    # to call instance methods
    probabilisticAtlas = ProbabilisticAtlas()
    # Bias correct images
    biasFields = parameters_and_inputs["biasFields"]
    imageBuffers = parameters_and_inputs["imageBuffers"]
    mask = parameters_and_inputs["mask"]

    # Write these out
    exampleImageFileName = parameters_and_inputs["imageFileNames"]
    exampleImage = gems.KvlImage(exampleImageFileName[0])
    cropping = parameters_and_inputs["cropping"]
    _, expBiasFields = undoLogTransformAndBiasField(
        imageBuffers, biasFields, mask
    )

    # Next do the segmentation, first read in the mesh
    modelSpecifications = parameters_and_inputs["modelSpecifications"]
    transform_matrix = parameters_and_inputs["transform"]
    transform = gems.KvlTransform(requireNumpyArray(transform_matrix))
    deformation = parameters_and_inputs["deformation"]
    deformationAtlasFileName = parameters_and_inputs["deformationAtlas"]

    mesh = probabilisticAtlas.getMesh(
        modelSpecifications.atlasFileName,
        transform=transform,
        K=modelSpecifications.K,
        initialDeformation=deformation,
        initialDeformationMeshCollectionFileName=deformationAtlasFileName,
    )

    fractionsTable = parameters_and_inputs["fractionsTable"]
    GMMparameters = parameters_and_inputs["GMMParameters"]
    numberOfGaussiansPerClass = parameters_and_inputs["gaussiansPerClass"]
    means = GMMparameters["means"]
    variances = GMMparameters["variances"]
    mixtureWeights = GMMparameters["mixtureWeights"]
    names = parameters_and_inputs["names"]
    bg_label = names.index("Background")
    FreeSurferLabels = np.array(modelSpecifications.FreeSurferLabels, dtype=np.uint16)
    segmentation_tissues = tissue_settings["segmentation_tissues"]
    csf_tissues = segmentation_tissues["CSF"]

    segmentation = _calculateSegmentationLoop(
        imageBuffers - biasFields,
        mask,
        fractionsTable,
        mesh,
        numberOfGaussiansPerClass,
        means,
        variances,
        mixtureWeights,
        FreeSurferLabels,
        bg_label,
        csf_tissues,
        csf_factor,
    )

    writeImage(output_name_segmentation, segmentation, cropping, exampleImage)

    unmaskedBuffers, _, _, _ = (
        readCroppedImages(parameters_and_inputs["imageFileNames"], template_coregistered)
    )

    biasCorrectedBuffers = unmaskedBuffers / expBiasFields
    biasCorrectedBuffers[segmentation == FreeSurferLabels[bg_label]] = 0
    for contrastNumber, out_name in enumerate(output_names_bias):
        # Bias field correct and write
        writeImage(
            out_name, biasCorrectedBuffers[..., contrastNumber], cropping, exampleImage
        )

def undoLogTransformAndBiasField(imageBuffers, biasFields, mask):
    #
    expBiasFields = np.zeros(biasFields.shape, order='F')
    numberOfContrasts = imageBuffers.shape[-1]
    for contrastNumber in range(numberOfContrasts):
        # We're computing it also outside of the mask, but clip the intensities there to the range
        # observed inside the mask (with some margin) to avoid crazy extrapolation values
        biasField = biasFields[:, :, :, contrastNumber]
        clippingMargin = np.log(2)
        clippingMin = biasField[mask].min() - clippingMargin
        clippingMax = biasField[mask].max() + clippingMargin
        biasField[biasField < clippingMin] = clippingMin
        biasField[biasField > clippingMax] = clippingMax
        expBiasFields[:, :, :, contrastNumber] = np.exp(biasField)

    #
    expImageBuffers = np.exp(imageBuffers) / expBiasFields
    return expImageBuffers, expBiasFields

def segmentUpsampled(
    input_bias_corrected,
    tissue_settings,
    parameters_and_inputs,
    transformedTemplateFileName,
    affine_atlas,
    csf_factor,
):
    # We need an init of the probabilistic segmentation class
    # to call instance methods
    probabilisticAtlas = ProbabilisticAtlas()

    # Read the input images doing the necessary cropping
    (
        imageBuffersUpsampled,
        transformUpsampled,
        voxelSpacingUpsampled,
        croppingUpsampled,
    ) = readCroppedImages(input_bias_corrected, transformedTemplateFileName)
    # Redo background masking now with the upsampled scans, note this only rasterizes
    # a single class so should be decent memory-wise
    modelSpecifications = parameters_and_inputs["modelSpecifications"]
    imageBuffersUpsampled, maskUpsampled = maskOutBackground(
        imageBuffersUpsampled,
        input_bias_corrected,
        modelSpecifications.atlasFileName,
        transformUpsampled,
        modelSpecifications.brainMaskingSmoothingSigma,
        modelSpecifications.brainMaskingThreshold,
        probabilisticAtlas,
        upsampled=True,
    )

    # Log-transform the intensities, note the scans are already
    # bias corrected so no need to remove the bias contribution

    imageBuffersUpsampled = logTransform(imageBuffersUpsampled, maskUpsampled)

    # Calculate the posteriors.
    # NOTE: the method for calculating the posteriors
    # in the gmm class rasterizes all classes in the atlas at once.
    # This is very memory-heavy if we have many classes
    # and the input resolution is high. Here we do it instead in
    # a loop to save memory.
    deformation = parameters_and_inputs["deformation"]
    deformationAtlasFileName = parameters_and_inputs["deformationAtlas"]
    meshUpsampled = probabilisticAtlas.getMesh(
        modelSpecifications.atlasFileName,
        transformUpsampled,
        initialDeformation=deformation,
        initialDeformationMeshCollectionFileName=deformationAtlasFileName,
    )
    del deformation
    fractionsTable = parameters_and_inputs["fractionsTable"]
    GMMparameters = parameters_and_inputs["GMMParameters"]
    numberOfGaussiansPerClass = parameters_and_inputs["gaussiansPerClass"]
    means = GMMparameters["means"]
    variances = GMMparameters["variances"]
    mixtureWeights = GMMparameters["mixtureWeights"]
    names = parameters_and_inputs["names"]
    bg_label = names.index("Background")
    FreeSurferLabels = np.array(modelSpecifications.FreeSurferLabels, dtype=np.uint16)
    simnibs_tissues = tissue_settings["simnibs_tissues"]
    segmentation_tissues = tissue_settings["segmentation_tissues"]
    csf_tissues = segmentation_tissues["CSF"]
    segmentation = _calculateSegmentationLoop(
        imageBuffersUpsampled,
        maskUpsampled,
        fractionsTable,
        meshUpsampled,
        numberOfGaussiansPerClass,
        means,
        variances,
        mixtureWeights,
        FreeSurferLabels,
        bg_label,
        csf_tissues,
        csf_factor,
    )

    del meshUpsampled
    del imageBuffersUpsampled

    tissue_labeling = np.zeros_like(segmentation)
    for t, label in simnibs_tissues.items():
        tissue_labeling[
            np.isin(segmentation, FreeSurferLabels[segmentation_tissues[t]])
        ] = label

    example_image = gems.KvlImage(input_bias_corrected[0])
    uncropped_tissue_labeling = np.zeros(
        example_image.getImageBuffer().shape, dtype=np.uint16, order="F"
    )
    uncropped_tissue_labeling[croppingUpsampled] = tissue_labeling

    # Create a head mask for post-processing
    affine_upsampled = probabilisticAtlas.getMesh(affine_atlas, transformUpsampled)

    upper_part = np.zeros(example_image.getImageBuffer().shape, bool, order="F")
    upper_part_cropped = affine_upsampled.rasterize(maskUpsampled.shape, 1)
    upper_part[croppingUpsampled] = (65535 - upper_part_cropped) > 32768
    del affine_upsampled
    del maskUpsampled
    del upper_part_cropped
    gc.collect()
    return uncropped_tissue_labeling, upper_part


def saveWarpField(template_name, warp_to_mni, warp_from_mni, parameters_and_inputs):
    # Save warp field two ways: the subject space world coordinates
    # in template space, i.e., the iamge voxel coordinates in
    # physical space for every voxel in the template.
    # And the other way around, i.e., the template voxel coordinates
    # in physical space for every voxel in the image.

    # We need an init of the probabilistic segmentation class
    # to call instance methods
    probabilisticAtlas = ProbabilisticAtlas()

    # First write image -> template.
    # Get the node positions in image voxels
    modelSpecifications = parameters_and_inputs["modelSpecifications"]
    transform_matrix = parameters_and_inputs["transform"]
    transform = gems.KvlTransform(requireNumpyArray(transform_matrix))

    deformation = parameters_and_inputs["deformation"]
    deformationAtlasFileName = parameters_and_inputs["deformationAtlas"]
    nodePositions = probabilisticAtlas.getMesh(
        modelSpecifications.atlasFileName,
        transform,
        K=modelSpecifications.K,
        initialDeformation=deformation,
        initialDeformationMeshCollectionFileName=deformationAtlasFileName,
    ).points

    # The image is cropped as well so the voxel coordinates
    # do not exactly match with the original image,
    # i.e., there's a shift. Let's undo that.
    cropping = parameters_and_inputs["cropping"]
    nodePositions += [slc.start for slc in cropping]

    # Get mapping from voxels to world space of the image.
    imageFileNames = parameters_and_inputs["imageFileNames"]
    image = nib.load(imageFileNames[0])
    imageToWorldTransformMatrix = image.affine
    image_buffer = image.get_fdata()
    # Transform the node positions
    nodePositionsInWorldSpace = (
        imageToWorldTransformMatrix
        @ np.pad(nodePositions, ((0, 0), (0, 1)), "constant", constant_values=1).T
    ).T
    nodePositionsInWorldSpace = nodePositionsInWorldSpace[:, 0:3]
    template = gems.KvlImage(template_name)
    templateToWorldTransformMatrix = template.transform_matrix.as_numpy_array
    template_buffer = template.getImageBuffer()

    # Rasterize the final node coordinates (in image space)
    # using the initial template mesh
    mesh = probabilisticAtlas.getMesh(
        modelSpecifications.atlasFileName, K=modelSpecifications.K
    )

    # Get node positions in template voxel space
    nodePositionsTemplate = mesh.points

    # Rasterize the coordinate values
    coordmapTemplate = mesh.rasterize_values(
        template_buffer.shape, nodePositionsInWorldSpace
    )
    # Write the warp file
    temp_header = nib.load(template_name)
    warp_image = nib.Nifti1Image(coordmapTemplate, temp_header.affine)
    # templateToWorldTransformMatrix)
    nib.save(warp_image, warp_to_mni)

    # Now do it the other way, i.e., template->image
    nodePositionsTemplateWorldSpace = (
        temp_header.affine
        @ np.pad(
            nodePositionsTemplate, ((0, 0), (0, 1)), "constant", constant_values=1
        ).T
    ).T
    nodePositionsTemplateWorldSpace = nodePositionsTemplateWorldSpace[:, 0:3]
    # Okay get the mesh in image space
    mesh = probabilisticAtlas.getMesh(
        modelSpecifications.atlasFileName,
        transform,
        initialDeformation=deformation,
        initialDeformationMeshCollectionFileName=deformationAtlasFileName,
    )

    imageBuffers = parameters_and_inputs["imageBuffers"]
    coordmapImage = mesh.rasterize_values(
        imageBuffers.shape[0:-1], nodePositionsTemplateWorldSpace
    )
    # The image buffer is cropped so need to set
    # everything to the correct place
    uncroppedWarp = np.zeros(image_buffer.shape + (3,), dtype=np.float32, order="F")

    for c in range(coordmapImage.shape[-1]):
        uncroppedMap = np.zeros(image_buffer.shape, dtype=np.float32, order="F")
        uncroppedMap[cropping] = np.squeeze(coordmapImage[:, :, :, c])
        uncroppedWarp[:, :, :, c] = uncroppedMap

    # Write the warp
    warp_image = nib.Nifti1Image(uncroppedWarp, imageToWorldTransformMatrix)
    nib.save(warp_image, warp_from_mni)


def _calculateSegmentationLoop(
    biasCorrectedImageBuffers,
    mask,
    fractionsTable,
    mesh,
    numberOfGaussiansPerClass,
    means,
    variances,
    mixtureWeights,
    FreeSurferLabels,
    bg_label,
    csf_tissues,
    csf_factor,
):
    data = biasCorrectedImageBuffers[mask, :]
    channels = data.shape[1]
    numberOfVoxels = data.shape[0]
    numberOfStructures = fractionsTable.shape[1]

    # We need an init of the GMM class
    # to call instance methods
    gmm = GMM(
        numberOfGaussiansPerClass, data.shape[1], True, means, variances, mixtureWeights
    )

    # These will store the max values and indices
    maxValues = np.zeros(biasCorrectedImageBuffers.shape[0:3], dtype=np.float32)
    maxIndices = np.empty(biasCorrectedImageBuffers.shape[0:3], dtype=np.uint16)

    maxIndices[:] = FreeSurferLabels[bg_label]
    print("done")

    # The different structures can share their mixtures between classes
    # E.g., thalamus can be half wm and half gm. This needs to be accounted
    # for when looping. We'll first loop over the structures, i.e.,
    # everything that's in the segmentation, so that we also
    # include the correct fractions for each class (if any)
    # NOTE: this function ONLY calculates the labeling and
    # returns the normalizer (if specified). I.e., we don't
    # need to normalize to find the label with highest probability.
    # But we're not going to compute the posteriors here.
    # We're also filling zero values (outside the mask) from the prior
    print("Segmenting, can take a while...")
    for structureNumber in range(numberOfStructures):
        # Rasterize the current structure from the atlas
        # and cast to float from uint16
        nonNormalized = mesh.rasterize(mask.shape, structureNumber) / 65535.0
        prior = nonNormalized[mask]

        # Find which classes we need to look at
        # to get the fractions correct
        print(
            "Segmented "
            + str(structureNumber + 1)
            + " out of "
            + str(numberOfStructures)
            + " structures."
        )
        classesToLoop = [
            i for i, val in enumerate(fractionsTable[:, structureNumber] > 1e-10) if val
        ]
        likelihoods = np.zeros(numberOfVoxels, dtype=np.float32)
        for classNumber in classesToLoop:
            # Compute likelihood for this class
            numberOfComponents = numberOfGaussiansPerClass[classNumber]
            fraction = fractionsTable[classNumber, structureNumber]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = (
                    sum(numberOfGaussiansPerClass[:classNumber]) + componentNumber
                )
                mean = np.expand_dims(means[gaussianNumber, :], 1)
                variance = variances[gaussianNumber, :, :]
                mixtureWeight = mixtureWeights[gaussianNumber]

                gaussianLikelihood = gmm.getGaussianLikelihoods(data, mean, variance)
                likelihoods += gaussianLikelihood * mixtureWeight * fraction

        # Now compute the non-normalized posterior

        if (channels > 1) and (structureNumber in csf_tissues):
            nonNormalized[mask] = csf_factor * likelihoods * prior
        else:
            nonNormalized[mask] = likelihoods * prior

        if structureNumber == 0:
            # In the first iteration just save the non-normalized values
            # Indices are initialized to zero anyway
            maxValues = nonNormalized
        else:
            # Check whether we have higher values and save indices
            higher_values = nonNormalized > maxValues
            maxValues[higher_values] = nonNormalized[higher_values]
            maxIndices[higher_values] = FreeSurferLabels[structureNumber]

    return maxIndices
