# cython: language_level=3
import numpy as np
cimport numpy as np


cdef extern from "_sanlm_source.c":
    void anlm(float* ima, int v, int f, int use_rician, const int* dims)


def sanlm(image, v, f, use_rician=False):
    ''' ANLM Denoising
    Unlike the CAT12 version, does not operate in-place. Instead, it creates a new image
    '''
    cdef np.ndarray[float, ndim=3] image_f = image.astype(np.float32)
    cdef np.ndarray[int, ndim=1] dims = np.array(image.shape, np.int32)
    anlm(&image_f[0, 0, 0], int(v), int(f), int(use_rician), &dims[0])
    return image_f
