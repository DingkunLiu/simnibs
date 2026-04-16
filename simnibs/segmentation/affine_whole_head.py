import os

import nibabel as nib
import scipy.io
import scipy.ndimage
import numpy as np
from scipy.optimize import minimize_scalar
import logging


from samseg import gems
from samseg.Affine import Affine, initializationOptions, ProbabilisticAtlas
from samseg.utilities import requireNumpyArray
from .simnibs_segmentation_utils import readCroppedImages

eps = np.finfo(float).eps

# For debugging


class initializationOptionsWholeHead:
    def __init__(
        self,
        pitchAngles=np.array([0]) / 180.0 * np.pi,  # in radians
        scales=[[1.0, 1.0, 1.0]],
        horizontalTableShifts=[
            20.0,
            10.0,
            0.0,
            -10.0,
            -20.0,
        ],  # in mm, in template space
        verticalTableShifts=[-10.0, 0.0, 10.0],  # in mm, in template space
        tryCenterOfGravity=True,
        searchForTableShiftsSeparately=False,
        pitchCenter=[0.0, 0.0, 0.0],  # in mm, in template space - anterior commissure
        scalingCenter=[0.0, -120.0, 0.0],  # in mm, in template space - back of the head
        initialPitchAngle=0,  # in radians
        initialScale=1.0,
        initialTableShift=[0.0, 0.0, 0.0],  # in mm, in template space
    ):
        self.pitchAngles = pitchAngles
        self.scales = scales
        self.horizontalTableShifts = horizontalTableShifts
        self.verticalTableShifts = verticalTableShifts
        self.tryCenterOfGravity = tryCenterOfGravity
        self.searchForTableShiftsSeparately = searchForTableShiftsSeparately
        self.pitchCenter = pitchCenter
        self.scalingCenter = scalingCenter
        self.initialPitchAngle = initialPitchAngle
        self.initialScale = initialScale
        self.initialTableShift = initialTableShift


class AffineWholeHead(Affine):
    def registerAtlasWholeHead(
        self,
        worldToWorldTransformMatrix=None,
        initTransform=None,
        Ks=[20.0, 10.0, 5.0],
        initializationOptions=initializationOptions(),
        targetDownsampledVoxelSpacing=3.0,
        maximalDeformationStopCriterion=0.005,
        visualizer=None,
        noneck=False,
    ):
        # ------ Set up ------
        self.setUp(initTransform, targetDownsampledVoxelSpacing, visualizer)

        # ------ Register mesh to image ------
        (
            imageToImageTransformMatrix,
            worldToWorldTransformMatrix,
            optimizationSummary,
        ) = self.registerMeshToImageWholeHead(
            worldToWorldTransformMatrix=worldToWorldTransformMatrix,
            Ks=Ks,
            maximalDeformationStopCriterion=maximalDeformationStopCriterion,
            initializationOptions=initializationOptions,
            noneck=noneck,
        )

        return (
            imageToImageTransformMatrix,
            worldToWorldTransformMatrix,
            optimizationSummary,
        )

    def getInitializationWholeHead(self, initializationOptions, noneck):
        # Remember some things that are fixed
        self.pitchCenter = initializationOptions.pitchCenter
        self.scalingCenter = initializationOptions.scalingCenter
        self.initialTableShift = initializationOptions.initialTableShift
        self.initialPitchAngle = initializationOptions.initialPitchAngle
        self.initialScale = initializationOptions.initialScale

        # Get the mesh. Although we're not penalizing prior deformation here (K=0), we still need to first put
        # the reference mesh in subject space to make sure no tetrahedra get negative volume (which would trigger
        # a sentinel "infinity" cost even when K=0)
        imageToImageTransformMatrix = (
            np.linalg.inv(self.imageToWorldTransformMatrix)
            @ self.templateImageToWorldTransformMatrix
        )
        imageToImageTransform = gems.KvlTransform(
            requireNumpyArray(imageToImageTransformMatrix)
        )
        mesh = ProbabilisticAtlas().getMesh(
            self.meshCollectionFileName, transform=imageToImageTransform, K=0.0
        )

        positionsInTemplateSpace = (
            ProbabilisticAtlas().mapPositionsFromSubjectToTemplateSpace(
                mesh.points, imageToImageTransform
            )
        )

        #
        bestInitialTableShift = initializationOptions.initialTableShift
        if initializationOptions.tryCenterOfGravity:
            # Get the centers of gravity of atlas and image, and use that to propose a translation
            mesh.points = positionsInTemplateSpace
            templateSize = np.round(np.max(mesh.points, axis=0) + 1).astype("int")
            priors = mesh.rasterize(templateSize, -1)

            if noneck:
                head = np.sum(priors[:, :, :, 2:], axis=3)
            else:
                head = np.sum(priors[:, :, :, 1:], axis=3)

            centerOfGravityTemplate = np.array(
                scipy.ndimage.center_of_mass(head)
            )  # in image space

            centerOfGravityImage = np.array(
                scipy.ndimage.center_of_mass(self.image.getImageBuffer())
            )  # in image space
            tmp = self.getTransformMatrix()
            tmp = tmp @ self.templateImageToWorldTransformMatrix
            centerOfGravityTemplate = (
                tmp[0:3, 0:3] @ centerOfGravityTemplate + tmp[0:3, 3]
            )  # in world space
            centerOfGravityImage = (
                self.imageToWorldTransformMatrix[0:3, 0:3] @ centerOfGravityImage
                + self.imageToWorldTransformMatrix[0:3, 3]
            )  # in world space
            centeringTableShift = centerOfGravityImage - centerOfGravityTemplate

            # Try which one is best
            initialTableShifts = [
                initializationOptions.initialTableShift,
                centeringTableShift,
            ]
            _, _, _, _, bestInitialTableShift, _ = self.gridSearch(
                mesh,
                positionsInTemplateSpace,
                initialTableShifts=initialTableShifts,
                visualizerTitle="Affine grid search center of gravity",
            )

        # Remove neck alphas and back ground to get the scaling, rotation and shifts correctly
        alphas_orig = mesh.alphas
        alphas_tmp = alphas_orig.copy()
        alphas_tmp = alphas_tmp[:, 2:]
        mesh.alphas = alphas_tmp
        if initializationOptions.searchForTableShiftsSeparately:
            # Perform grid search for intialization (done separately for translation and scaling/rotation to limit the
            # number of combinations to be tested)
            _, _, bestHorizontalTableShift, bestVerticalTableShift, _, _ = (
                self.gridSearch(
                    mesh,
                    positionsInTemplateSpace,
                    horizontalTableShifts=initializationOptions.horizontalTableShifts,
                    verticalTableShifts=initializationOptions.verticalTableShifts,
                    initialTableShifts=[bestInitialTableShift],
                    visualizerTitle="Affine grid search shifts",
                )
            )

            _, _, _, _, _, bestImageToImageTransformMatrix = self.gridSearch(
                mesh,
                positionsInTemplateSpace,
                pitchAngles=initializationOptions.pitchAngles,
                scales=initializationOptions.scales,
                horizontalTableShifts=[bestHorizontalTableShift],
                verticalTableShifts=[bestVerticalTableShift],
                initialTableShifts=[bestInitialTableShift],
                visualizerTitle="Affine grid search anges/scales",
            )
        else:
            # Basic grid search
            _, _, _, _, _, bestImageToImageTransformMatrix = self.gridSearch(
                mesh,
                positionsInTemplateSpace,
                pitchAngles=initializationOptions.pitchAngles,
                scales=initializationOptions.scales,
                horizontalTableShifts=initializationOptions.horizontalTableShifts,
                verticalTableShifts=initializationOptions.verticalTableShifts,
                initialTableShifts=[bestInitialTableShift],
                visualizerTitle="Affine grid search",
            )

        # Stick the original alphas back
        mesh.alphas = alphas_orig
        return bestImageToImageTransformMatrix

    def optimizeTransformationWholeHead(
        self,
        initialImageToImageTransformMatrix,
        K,
        maximalDeformationStopCriterion,
        noneck,
    ):
        # In our models the mesh stiffness (scaling the log-prior) is relative to the log-likelihood,
        # which scales with the number of voxels being modeled. But for MI the log-likelihood
        # is normalized, so we need to divide the mesh stiffness by the (expected) number of voxels
        # covered in order to compensate.
        Keffective = K / self.expectedNumberOfVoxelsCovered

        # Get the mesh
        initialImageToImageTransform = gems.KvlTransform(
            requireNumpyArray(initialImageToImageTransformMatrix)
        )
        mesh = ProbabilisticAtlas().getMesh(
            self.meshCollectionFileName,
            transform=initialImageToImageTransform,
            K=Keffective,
        )
        originalNodePositions = mesh.points

        if noneck:
            alphas = mesh.alphas
            alphas_tmp = np.zeros((alphas.shape[0], alphas.shape[1] - 1))
            alphas_tmp[:, 0] = alphas[:, 0] + alphas[:, 1]
            alphas_tmp[:, 1:] = alphas[:, 2:]
            mesh.alphas = alphas_tmp

        # Get a registration cost  and stick it in an optimizer
        calculator = gems.KvlCostAndGradientCalculator(
            "MutualInformation", [self.image], "Affine"
        )
        optimization_parameters = {
            "Verbose": 0,
            "MaximalDeformationStopCriterion": maximalDeformationStopCriterion,
            "LineSearchMaximalDeformationIntervalStopCriterion": maximalDeformationStopCriterion,
            "BFGS-MaximumMemoryLength": 12.0,  # Affine registration only has 12 DOF
        }
        optimizer = gems.KvlOptimizer(
            "L-BFGS", mesh, calculator, optimization_parameters
        )

        # Peform the optimization
        numberOfIterations = 0
        minLogLikelihoodTimesPriors = []
        maximalDeformations = []
        visualizerTitle = f"Affine optimization (K: {K})"
        self.visualizer.start_movie(window_id=visualizerTitle, title=visualizerTitle)
        self.visualizer.show(
            mesh=mesh,
            images=self.image.getImageBuffer(),
            window_id=visualizerTitle,
            title=visualizerTitle,
        )
        while True:
            minLogLikelihoodTimesPrior, maximalDeformation = (
                optimizer.step_optimizer_atlas()
            )
            print(
                "maximalDeformation=%.4f minLogLikelihood=%.4f"
                % (maximalDeformation, minLogLikelihoodTimesPrior)
            )
            minLogLikelihoodTimesPriors.append(minLogLikelihoodTimesPrior)
            maximalDeformations.append(maximalDeformation)

            if maximalDeformation == 0:
                break
            numberOfIterations += 1
            self.visualizer.show(
                mesh=mesh,
                images=self.image.getImageBuffer(),
                window_id=visualizerTitle,
                title=visualizerTitle,
            )

        self.visualizer.show_movie(window_id=visualizerTitle)
        nodePositions = mesh.points
        pointNumbers = [0, 110, 201, 302]
        originalY = np.vstack((originalNodePositions[pointNumbers].T, [1, 1, 1, 1]))

        Y = np.vstack((nodePositions[pointNumbers].T, [1, 1, 1, 1]))
        extraImageToImageTransformMatrix = Y @ np.linalg.inv(originalY)
        appliedScaling = np.linalg.det(extraImageToImageTransformMatrix) ** (1 / 3)
        print(f"appliedScaling: {appliedScaling:0.4f}")
        imageToImageTransformMatrix = (
            extraImageToImageTransformMatrix @ initialImageToImageTransformMatrix
        )

        optimizationSummary = {
            "numberOfIterations": len(minLogLikelihoodTimesPriors),
            "cost": minLogLikelihoodTimesPriors[-1],
        }
        return imageToImageTransformMatrix, optimizationSummary

    def registerMeshToImageWholeHead(
        self,
        worldToWorldTransformMatrix,
        Ks,
        maximalDeformationStopCriterion,
        initializationOptions,
        noneck=False,
    ):
        if worldToWorldTransformMatrix is not None:
            # The world-to-world transfrom is externally given, so let's just compute the corresponding image-to-image
            # transform (needed for subsequent computations) and be done
            print("world-to-world transform supplied - skipping registration")
            imageToImageTransformMatrix = (
                np.linalg.inv(self.originalImageToWorldTransformMatrix)
                @ worldToWorldTransformMatrix
                @ self.templateImageToWorldTransformMatrix
            )
            optimizationSummary = None
        else:
            # The solution is not externally given, so we need to compute it.
            print("performing affine atlas registration")
            print("image: %s" % self.imageFileName)
            print("template: %s" % self.templateFileName)

            # Get an initialization of the image-to-image transform (template -> subject)
            imageToImageTransformMatrix = self.getInitializationWholeHead(
                initializationOptions, noneck
            )

            # Optimze image-to-image transform (template->subject)
            for K in Ks:
                imageToImageTransformMatrix, optimizationSummary = (
                    self.optimizeTransformationWholeHead(
                        imageToImageTransformMatrix,
                        K,
                        maximalDeformationStopCriterion,
                        noneck,
                    )
                )

            # Final result: the image-to-image (from template to image) as well as the world-to-world transform that
            # we computed (the latter would be the identity matrix if we didn't move the image at all)
            imageToImageTransformMatrix = (
                self.upSamplingTranformMatrix @ imageToImageTransformMatrix
            )  # template-to-original-image
            worldToWorldTransformMatrix = (
                self.originalImageToWorldTransformMatrix
                @ imageToImageTransformMatrix
                @ np.linalg.inv(self.templateImageToWorldTransformMatrix)
            )

        # return result
        return (
            imageToImageTransformMatrix,
            worldToWorldTransformMatrix,
            optimizationSummary,
        )

    def saveResultsWholeHead(
        self,
        image_name,
        template_name,
        save_path,
        template_coregistered_name,
        image_to_image_transform,
        world_to_world_transform,
    ):
        # ------ Save Registration Results ------
        template = gems.KvlImage(template_name)
        image = gems.KvlImage(image_name)
        image_to_world_transform = image.transform_matrix.as_numpy_array
        # Save the world-to-world transformation matrix in RAS
        RAS2LPS = np.diag([-1, -1, 1, 1])
        world_to_world_transform_ras = RAS2LPS @ world_to_world_transform @ RAS2LPS
        scipy.io.savemat(
            os.path.join(save_path, "coregistrationMatrices.mat"),
            {
                "imageToImageTransformMatrix": image_to_image_transform,
                "worldToWorldTransformMatrix": world_to_world_transform_ras,
            },
        )

        # Save the coregistered template. For historical reasons,
        # we applied the estimated transformation to the template...
        # let's do that now
        template_image_to_world_transform = np.asfortranarray(
            image_to_world_transform @ image_to_image_transform
        )
        template.write(
            template_coregistered_name,
            gems.KvlTransform(template_image_to_world_transform),
        )

    def removeNeck(self, mesh):
        alphas = mesh.alphas
        # Note: the spine class is the last one in the affine atlas.
        # Might need to change this in the future.
        spineAlphas = alphas[:, 49]
        mask = spineAlphas > 0.01

        # Get z-coordinates
        zPositions = mesh.points[:, -1]

        # Find the max position for spine and cut from there
        spinePosZ = zPositions[mask]
        maxInds = np.argwhere(np.argmax(spinePosZ))
        maxZCoord = zPositions[maxInds[0]]
        zMask = zPositions < maxZCoord

        # Create a "neck" class which includes everything in the neck
        alphasWithNeck = np.zeros((alphas.shape[0], alphas.shape[1] + 1))
        alphasWithNeck[:, 1:] = alphas
        alphasWithNeck[zMask, 1:] = 0
        alphasWithNeck[zMask, 0] = 1.0

        return alphasWithNeck

    def adjust_neck(
        self,
        T1,
        transformed_template_file_name,
        mesh_level1,
        mesh_level2,
        neck_bounds,
        neck_tissues,
        visualizer,
        debug=False,
        downsampling_target=3.0,
    ):
        logger = logging.getLogger(__name__)
        image_buffer, transformation_to_image, voxel_spacing, cropping = (
            readCroppedImages([T1], transformed_template_file_name)
        )
        transformation_matrix = transformation_to_image.as_numpy_array
        # Figure out how much to downsample (depends on voxel size)
        downsampling_factors = np.round(downsampling_target / voxel_spacing)
        downsampling_factors[downsampling_factors < 1] = 1
        image_buffer = image_buffer[
            :: int(downsampling_factors[0]),
            :: int(downsampling_factors[1]),
            :: int(downsampling_factors[2]),
        ]
        # Read in the mesh collection, and set K to be very small
        # so that the intensity cost dominates
        K = 1e-20
        mesh_collection = gems.KvlMeshCollection()
        mesh_collection.read(mesh_level2)
        mesh_collection.k = K * np.prod(downsampling_factors)
        mesh_collection.transform(transformation_to_image)
        mesh = mesh_collection.reference_mesh
        # Downsample if needed
        mesh.scale(1 / downsampling_factors)

        # Let's define the neck deformation based on the distance
        alphas = mesh.alphas
        # Note: the spine class is the last one in the affine atlas.
        # Might need to change this in the future.
        spineAlphas = alphas[:, neck_tissues[2]]
        mask_spine = spineAlphas > 0.1

        # Let's figure out the voxel orienation
        T1_nib = nib.load(T1)
        ort = nib.aff2axcodes(T1_nib.affine)
        del T1_nib
        if "S" in ort:
            z_dim = ort.index("S")
            up = True
        elif "I" in ort:
            z_dim = ort.index("I")
            up = False
        else:
            logger.info("Can't figure out orientation. Skipping neck adjustment.")
            return -1

        # Get z-coordinates in image voxel space
        z_positions = mesh.points[:, z_dim]
        spine_positions_z = z_positions[mask_spine]
        mask_neck = 0
        neck_pos = 0
        # The values are stored from bottom of the head (0) to the top (-1)
        # Note, here the mesh nodes are already transformed to the voxels
        # of the image.
        if up:
            top_ind = np.argmax(spine_positions_z)
            top_pos = spine_positions_z[top_ind]
            mask_neck = z_positions < top_pos
            neck_pos = z_positions[mask_neck]
            z_dist = top_pos - neck_pos
        else:
            top_ind = np.argmin(spine_positions_z)
            top_pos = spine_positions_z[top_ind]
            mask_neck = z_positions > top_pos
            neck_pos = z_positions[mask_neck]
            z_dist = neck_pos - top_pos

        # Okay the distance from the top of the spine defines the amount
        # of deformation in the A->P direction
        # Let's figure out where A->P is
        if "A" in ort:
            x_dim = ort.index("A")
        elif "P" in ort:
            x_dim = ort.index("P")
        else:
            logger.info("Can't figure out orientation. Skipping neck adjustment.")
            return -1

        deformation_field = np.zeros_like(mesh.points)
        deformation_field[mask_neck, x_dim] = z_dist
        if debug:
            import matplotlib.pyplot as plt

            alphas_tmp = alphas.copy()
            alphas_tmp[mask_neck, :] = 0
            mesh.alphas = alphas_tmp
            probs = mesh.rasterize_atlas(image_buffer.shape)
            spine_probs = probs[:, :, 31, 46]
            fig, ax = plt.subplots(1, 1)
            ax.imshow(spine_probs, cmap="gray")
            plt.show()

        # Okay one more trick that seems to work, only consider a subset
        # of the structures to compute the cost
        # These are: air internal, spine, cortical bone and spongy bone
        # NOTE! This need to be read from the setup as otherwise this
        # will fail if the atlas is changed!
        alphas_new = alphas[:, neck_tissues]
        mesh.alphas = alphas_new

        # Let's see how it looks
        visualizer.show(
            mesh=mesh,
            images=image_buffer,
            window_id="Initial Neck",
            title="Initial Neck",
        )

        if debug:
            import matplotlib.pyplot as plt

            probs = mesh.rasterize_atlas(image_buffer.shape)
            probs_to_show = np.sum(probs, axis=3)
            probs_to_show = probs_to_show[:, :, 31]
            fig, ax = plt.subplots(1, 1)
            ax.imshow(probs_to_show, cmap="gray")
            plt.show()

        image = gems.KvlImage(requireNumpyArray(image_buffer))
        calculator = gems.KvlCostAndGradientCalculator(
            "MutualInformation", [image], "Affine"
        )

        # Check initial cost
        initCost, _ = calculator.evaluate_mesh_position_a(mesh)
        logger.info("Initial cost: " + str(initCost))

        # Optimize the linear deformation, I'll do this now naively using
        # the Nelder-Mead Simplex algorithm. We should be able to compute the
        # gradient for this simple thing, so something smarter should be used
        initial_node_positions = mesh.points

        res = minimize_scalar(
            self._neck_cost,
            None,
            neck_bounds,
            (
                initial_node_positions,
                deformation_field,
                calculator,
                mesh,
                image_buffer,
                visualizer,
            ),
            method="Bounded",
        )

        mesh.points = initial_node_positions - res.x * deformation_field

        visualizer.show(
            mesh=mesh,
            images=image_buffer,
            window_id="Corrected Neck",
            title="Corrected Neck",
        )

        if debug:
            import matplotlib.pyplot as plt

            probs = mesh.rasterize_atlas(image_buffer.shape)
            probs_to_show = np.sum(probs, axis=3)
            probs_to_show = probs_to_show[:, :, 31]
            fig, ax = plt.subplots(1, 1)
            ax.imshow(probs_to_show, cmap="gray")
            plt.show()
        # Now need to write out the mesh collections with the new positions
        mesh.scale(downsampling_factors)
        mesh.alphas = alphas
        mesh_collection.transform(
            gems.KvlTransform(requireNumpyArray(np.linalg.inv(transformation_matrix)))
        )
        file_path = os.path.split(transformed_template_file_name)
        mesh_collection.write(os.path.join(file_path[0], "atlas_level2.txt"))

        # Also the other mesh collection
        mesh_collection.read(mesh_level1)
        mesh_collection.k = K
        mesh_collection.transform(transformation_to_image)
        mesh = mesh_collection.reference_mesh

        # Get z-coordinates
        alphas = mesh.alphas
        # Note: the spine class is the last one in the affine atlas.
        # Might need to change this in the future.
        spineAlphas = alphas[:, neck_tissues[2]]
        mask_spine = spineAlphas > 0.1
        z_positions = mesh.points[:, z_dim]
        spine_positions_z = z_positions[mask_spine]

        if up:
            top_ind = np.argmax(spine_positions_z)
            top_pos = spine_positions_z[top_ind]
            mask_neck = z_positions < top_pos
            neck_pos = z_positions[mask_neck]
            z_dist = top_pos - neck_pos
        else:
            top_ind = np.argmin(spine_positions_z)
            top_pos = spine_positions_z[top_ind]
            mask_neck = z_positions > top_pos
            neck_pos = z_positions[mask_neck]
            z_dist = neck_pos - top_pos

        deformation_field = np.zeros_like(mesh.points)
        deformation_field[mask_neck, x_dim] = z_dist
        initial_node_positions = mesh.points
        mesh.points = initial_node_positions - res.x * deformation_field

        mesh_collection.transform(
            gems.KvlTransform(requireNumpyArray(np.linalg.inv(transformation_matrix)))
        )
        mesh_collection.write(os.path.join(file_path[0], "atlas_level1.txt"))

        return 0

    def _neck_cost(
        self,
        x,
        initial_node_positions,
        deformation_field,
        calculator,
        mesh,
        image_buffer,
        visualizer,
    ):
        mesh.points = initial_node_positions - x * deformation_field
        cost, _ = calculator.evaluate_mesh_position_a(mesh)
        mesh.points = initial_node_positions
        return cost
