.. _charm_docs:

charm
=========

.. note:: Charm replaces :ref:`headreco_docs` or :ref:`mri2mesh_docs` that were included in older SimNIBS versions.

Description
------------

charm reconstructs a tetrahedral head mesh from T1- and T2-weighted structural MR images. It runs also with only a T1w image, but it will achieve more reliable skull segmentations when a T2w image is supplied.

Example: Basic Usage
--------------------

1. Open a terminal and go to the :file:`ernie/` folder of the example data set.
2. Run the reconstruction:

  .. code-block:: bash

     charm ernie org/ernie_T1.nii.gz org/ernie_T2.nii.gz

  \
  The subject ID (subID) :code:`ernie` is given as first argument. Charm will create a folder named :file:`m2m_ernie/` that contains the segmentation results and the final head mesh :file:`ernie.msh`. The input images are given as final arguments (first the T1, then the T2).

\

  Alternatively, the reconstruction can be run with only the T1w image as input, but this can result in a less accurate skull region:

  .. code-block:: bash

     charm ernie org/ernie_T1.nii.gz

  \

3. Check the segmentation. Click on the final segmentation viewer in the results.html (to be found in the m2m-folder of the subject). The viewer shows the outlines of the reconstructed tissue compartments, enabling a visual check whether the outlines are accurate.


Configuring CHARM
-----------------
Whereas options relevant to the controlling how ``charm`` behaves (e.g., which steps to execute) are available from the command line, detailed configuration is done in the ``charm.ini`` file. By default, ``charm`` will read a standard configuration file from the SimNIBS directory. It is also possible to pass a custom file by using the ``--usesettings`` options. (If no file is specified the default settings file is copied to the directory of the current subject.) The settings file contain different sections each of which allow you to tune the behavior of the different steps of the pipeline. Here we describe a selected subset of the available options in different sections.

surfaces
""""""""
As of SimNIBS 4.6, we use an implementation of the TopoFit network to reconstruct cortical surfaces. Here it is possible to specify the specific model parameters to use by changing ``topofit_contrast`` and ``topofit_resolution``.

- By default, ``topofit_contrast = "T1w"`` which expects *the first input image* to ``charm`` to be a T1w image. If this is not the case, you can set it to "random" which will use a contrast agnostic model.
- By default, ``topofit_resolution = "1mm"`` thus expecting *the first input image* to ``charm`` to be approximately 1 mm isotropic resolution. If this is not the case, you can set it to "random" which use a resolution agnistic model.

It is also possible to control how the central gray matter surface (infra-supragranular border) is estimated by setting ``central_surface_fraction`` (default is 0.5) and ``central_surface_method`` (options are equidistance, equivolume [default]).

It is also possible to use cortical surfaces created by FreeSurfer's *recon-all* pipeline with CHARM. To do this, pass the subject directory using the option ``--fs-dir``.


Further notes
--------------

* If you encounter spurious segmentation results this *could* be due to a suboptimal affine registration to MNI space. To fix this, you have several options:

  1. Use another initialization method (e.g., if using "TREGA", try for example "atlas"). This can be changed in the ``charm.ini`` file.
  2. When using the "atlas" registration method, it is possible to modify the search bounds on rotation, scaling, and translation. Please see the tutorial :ref:`fix_affine_registration_tutorial` for how to achieve this.
  3. When using the "atlas" registration method, it is possible to supply a custom transformation matrix that is for initialization. To pass this, use the option ``--inittransform``. Note that the transformation must be world-to-world (not voxel-to-voxel!).
  4. If none of the above works, you can estimate the transformation using some other method/software package and pass the resulting transformation directly to CHARM using the ``--usetransform`` option. Note that the transformation must be world-to-world (not voxel-to-voxel!).

* Since SimNIBS version 4.6, CHARM uses a new segmentation atlas that includes subcutaneous fat for improved skull border segmentation by default. If you want to use the previous version of the segmentation atlas, use option *--useatlasv1_0*.

References
-----------

`Puonti O, Van Leemput K, Saturnino GB, Siebner HR, Madsen KH, Thielscher A. (2020). Accurate and robust whole-head segmentation from magnetic resonance images for individualized head modeling. Neuroimage, 219:117044. <https://doi.org/10.1016/j.neuroimage.2020.117044>`_
