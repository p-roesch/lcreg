# lcreg: Efficent rigid and affine 3D image registration
# 
# Copyright (C) 2019  Peter Rösch, Peter.Roesch@hs-augsburg.de
# 
# Organisation:
# Faculty of Computer Science, Augsburg University of Applied Sciences,
# An der Hochschule 1, 86161 Augsburg, Germany
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#cython: language_level=3, boundscheck=False, profile=False, cdivision=True, wraparound=False
cimport numpy as np
import numpy as np

cimport cython
cimport openmp

from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free, abort
cimport lcreg.lcreg_lib

cdef double mirrored_interpolate_1d(double *a, index_t N, double x) nogil:
    """ Calculate interpolated grey values in a 1D float array.

        Args:
            a: 1D numpy float64 array
            N: length of a
            x: position at which the gray value should be interpolated
        Returns:
            interpolated value
    """
    # check if position is within the array including border pixels
    if x <= -0.5 or x >= N - 0.5:
        return 0.0

    # round position for pixel access
    cdef index_t i1 = my_round_index_t(x), i2

    # distance to pixel centre for weight calculation
    cdef double delta = i1 - x

    # choose one of the two neighbors to be incorporated
    # into interpolation
    if delta > 0:
        i2 = i1 - 1
    else:
        i2 = i1 + 1

    # perform mirroring for border pixels
    if i2 < 0:
        i2 = 1
    elif i2 >= N:
        i2 = N - 2

    # calculate weight
    cdef double w = w_linear(delta)

    # return interpolation result (weighted sum of grey values)
    return w * a[i1] + (1 - w) * a[i2]


cpdef double mirrored_interpolate_1d_py(np.float64_t[:] a, double x) nogil:
    """ Calculate interpolated grey values in a 1D float array, Python wrapper.

        Args:
            a: 1D numpy float64 array
            x: position at which the gray value should be interpolated
        Returns:
            interpolated value
    """
    cdef double *a_arr = <double *>malloc(sizeof(double) * a.shape[0])
    if a_arr == NULL:
        abort()
    cdef index_t i
    for i in range(a.shape[0]):
        a_arr[i] = <double> a[i]
    result = mirrored_interpolate_1d(a_arr, a.shape[0], x)
    free(a_arr)
    return result

# Typed memory views are used so that the GIL can be released
cdef double mirrored_interpolate_3d(np.int16_t[:, :, :] a, \
                                     double *x) nogil:
    """
        Perform 3D grey value interpolation.

        Note that the access order of the array is a[z, y, x] whereas
        x is a vector (x, y, z).

        Args:
            a: 3D array of int16 grey values, a[z, y, x]
            x: Position (x, y, z) at which grey value should be interpolated
        Returns:
            interpolated grey value (double)

    """

    # return 0 for positions outside of the image
    if not is_within_image(x, a):
        return 0.0

    cdef double delta

    # matrices containing weights and start indices
    # first index corresponds to the position (voxel 0 and 1)
    # second index corresponds to the axis (x, y, z)
    cdef double w[2][3]
    cdef index_t ind[2][3]

    cdef unsigned char i

    # i==0 corresponds to x, i==2 corresponds to y etc.
    for i in range(3):
        # voxel index for interpolation
        ind[0][i] = my_round_index_t(x[i])
        # argument for weight calculation
        delta = ind[0][i] -x[i]
        # index of second voxel to be used for interpolation
        ind[1][i] = ind[0][i] - 1 if delta > 0 else ind[0][i] + 1
        # perform mirroring
        if ind[1][i] < 0:
            ind[1][i] = 1
        elif ind[1][i] >= a.shape[2-i]:
            ind[1][i] = a.shape[2-i] - 2
        # weight value for fist voxel
        w[0][i] = w_linear(delta)
        # weight for second grey value
        w[1][i] = 1.0 - w[0][i]

    cdef double gval = 0.0
    cdef np.int16_t g
    cdef unsigned char i_x, i_y, i_z

    # iterate over 8 voxels to be considered
    for i_z in range(2):
        for i_y in range(2):
            for i_x in range(2):
                # get image grey value
                g = a[ind[i_z][2], ind[i_y][1], ind[i_x][0]]
                gval = gval+ g * w[i_z][2] * w[i_y][1] * w[i_x][0]
    return gval

cpdef double mirrored_interpolate_3d_py(np.int16_t[:, :, :] a,
        np.float64_t[:] x_np) nogil:
    """
        Python callable wrapper for 3D grey value interpolation.

        Note that the access order of the array is a[z, y, x] whereas
        x is a vector (x, y, z).

        Image Masking is represented implicitly by grey values as follows:
            grey values > 0: Voxels for which the similarity measure is to be
                calculated.
            grey values < 0: Voxels required for interpolation and similarity
                measure calculation in the vicinity of a voxel with
                grey value > 0
            grey value == 0: Background. Interpolation returns 0 for background
                voxels or positions close to the background.

        Args:
            a: 3D array of int16 grey values, a[z, y, x]
            x_np: Position (x, y, z) at which grey value should be interpolated
        Returns:
            interpolated grey value (double)
    """
    cdef double *x = <double *>malloc(3 * sizeof(double))
    if x == NULL:
        abort()

    x[0] = x_np[0]
    x[1] = x_np[1]
    x[2] = x_np[2]
    cdef double result = mirrored_interpolate_3d(a, x)
    free(x)
    return result

# Typed memory views are used so that the GIL can be released
cdef double masked_interpolate_3d(np.int16_t[:, :, :] a, \
                                     double *x) nogil:
    """
        Perform 3D grey value interpolation.

        Note that the access order of the array is a[z, y, x] whereas
        x is a vector (x, y, z).

        Image Masking is represented implicitly by grey values as follows:
            grey values > 0: Voxels for which the similarity measure is to be
                calculated.
            grey values < 0: Voxels required for interpolation and similarity
                measure calculation in the vicinity of a voxel with
                grey value > 0
            grey value == 0: Background. Interpolation returns 0 for background
                voxels or positions close to the background.

        Args:
            a: 3D array of int16 grey values, a[z, y, x]
            x: Position (x, y, z) at which grey value should be interpolated
        Returns:
            interpolated grey value (double)

    """

    # return 0 for positions outside of the image
    if not is_within_image_border_1(x, a):
        return 0.0

    cdef double delta

    # matrices containing weights and start indices
    # first index corresponds to the position (voxel 0 and 1)
    # second index corresponds to the axis (x, y, z)
    cdef double w[2][3]
    cdef index_t ind[2][3]

    cdef unsigned char i

    # i==0 corresponds to x, i==2 corresponds to y etc.
    for i in range(3):
        # voxel index for interpolation
        ind[0][i] = my_round_index_t(x[i])
        # argument for weight calculation
        delta = ind[0][i] - x[i]
        # weight value for fist voxel
        w[0][i] = w_linear(delta)
        # index of second voxel to be used for interpolation
        ind[1][i] = ind[0][i] - 1 if delta > 0 else ind[0][i] + 1
        # weight for second grey value
        w[1][i] = 1.0 - w[0][i]

    cdef double gval = 0.0, sign = -1
    cdef np.int16_t g
    cdef unsigned char i_x, i_y, i_z

    # iterate over 8 voxels to be considered
    for i_z in range(2):
        for i_y in range(2):
            for i_x in range(2):
                # get image grey value, return 0 if zero grey value is
                # encountered
                g = a[ind[i_z][2], ind[i_y][1], ind[i_x][0]]
                # check for masks and run length encoding
                if g == 0:
                    return 0.0
                # if one voxel in the environment posesses a positive
                # grey value, the result is positive to indicate that
                # this position is to be used for similarity measure
                # calculation
                if g > 0:
                    sign = 1
                else:
                    # use positive values for interpolation
                    g = -g
                # add contribution of this neighbor to the interpolation result
                gval = gval + g * w[i_z][2] * w[i_y][1] * w[i_x][0]

    return sign * gval


cpdef double masked_interpolate_3d_py(np.int16_t[:, :, :] a,
        np.float64_t[:] x_np) nogil:
    """
        Python callable wrapper for 3D grey value interpolation.

        Note that the access order of the array is a[z, y, x] whereas
        x is a vector (x, y, z).

        Image Masking is represented implicitly by grey values as follows:
            grey values > 0: Voxels for which the similarity measure is to be
                calculated.
            grey values < 0: Voxels required for interpolation and similarity
                measure calculation in the vicinity of a voxel with
                grey value > 0
            grey value == 0: Background. Interpolation returns 0 for background
                voxels or positions close to the background.

        Args:
            a: 3D array of int16 grey values, a[z, y, x]
            x_np: Position (x, y, z) at which grey value should be interpolated
        Returns:
            interpolated grey value (double)
    """
    cdef double *x = <double *>malloc(3 * sizeof(double))
    if x == NULL:
        abort()

    x[0] = x_np[0]
    x[1] = x_np[1]
    x[2] = x_np[2]
    cdef double result = masked_interpolate_3d(a, x)
    free(x)
    return result

@cython.cdivision(True)
cpdef void rescale_gvals(np.float64_t[:,:,:] in_im, 
        np.int16_t[:,:,:] out_im, double g_min, double g_max,
        np.int16_t out_min, np.int16_t out_max, np.float64_t epsilon,
        bint is_mask):
    cdef index_t x, y, z
    cdef np.int16_t result
    cdef np.float64_t gval_float64
    for z in range(in_im.shape[0]):
        for y in range(in_im.shape[1]):
            for x in range(in_im.shape[2]):
                result = 0
                gval_float64 = in_im[z, y, x]
                if is_mask: 
                    if gval_float64 != 0:
                        result = 1
                else:
                    if gval_float64 <= g_min:
                        result = out_min
                    elif gval_float64 >= g_max:
                        result = out_max
                    else:
                        result = <np.int16_t>((gval_float64 - g_min)
                                              / (g_max - g_min)
                                              * (out_max - out_min)
                                              + out_min + 0.5 + epsilon)
                out_im[z, y, x] = result


@cython.cdivision(True)
cpdef void downsample_XY(np.int16_t[:,:,:] in_im, np.float64_t[:, :, :] out_im,
                  np.float64_t[:] scale_factor,
                  np.float64_t[:] offset, int thread_num = -1) nogil:

    """
        Downsampling of an image in the xy-plane.

        in_im is resampled to the shape of out_im. First, resampling in x
        direction is done, the results are then resampled in y direction.
        The algorithm is based on 1D "integral images" similar to the method
        used in the SURF algorithm. The method allows for arbitrary scaling
        factors. However only downsampling is supported so that the size of
        out_im must not exceed the size of in_im in any direction.

        Args:
            in_im (memoryview, np.int16): input image
            out_im (memoryview, np.float64): output image
            scale_factor (numpy array): 3D scaling factor > 1 for downsampling
            thread_num (int): number of threads to use. Values < 1 indicate
                that all CPU threads should be used.
    """

    # check image sizes
    if in_im.shape[0] > out_im.shape[0]:
        abort()

    # only downsampling is supported
    if in_im.shape[1] < out_im.shape[1] or in_im.shape[2] < out_im.shape[2]:
        abort()

    if thread_num < 1:
        thread_num = openmp.omp_get_max_threads()

    cdef index_t z, y, x
    cdef double *im_slice
    cdef double *im_row_y
    cdef double *im_row_x
    cdef double sum_left, sum_right
    cdef double scale_factor_x = scale_factor[0]
    cdef double scale_factor_y = scale_factor[1]

    with nogil, parallel(num_threads=thread_num):
        im_slice = <double *>malloc(sizeof(double) * \
                    in_im.shape[1] * out_im.shape[2])
        if im_slice == NULL:
            abort()
        im_row_x   = <double *>malloc(sizeof(double) * in_im.shape[2])
        if im_row_x == NULL:
            abort()
        im_row_y   = <double *>malloc(sizeof(double) * in_im.shape[1])
        if im_row_y == NULL:
            abort()
        # work z-slice by z-slice
        for z in prange(in_im.shape[0], schedule='dynamic'):
            # perform row by row x resampling and store result in im_slice
            for y in range(in_im.shape[1]):
                # accumulate image row into buffer
                im_row_x[0] = in_im[z, y, 0]
                for x in range(1, in_im.shape[2]):
                    im_row_x[x] = im_row_x[x-1] + in_im[z, y, x]
                # downsample in x direction into im_slice
                sum_left = mirrored_interpolate_1d(im_row_x,
                            in_im.shape[2], offset[0] - 1.0)
                for x in range(out_im.shape[2]):
                    sum_right = mirrored_interpolate_1d(im_row_x,
                            in_im.shape[2],
                            (x + 1.0) * scale_factor_x - 1.0 + offset[0])
                    im_slice[x + y * out_im.shape[2]] =\
                        (sum_right - sum_left) / scale_factor_x
                    sum_left = sum_right
            # perform colum by column y resampling and store result in out_im
            for x in range(out_im.shape[2]):
                im_row_y[0] = im_slice[x]
                #accumulate into im_row_y
                for y in range(1, in_im.shape[1]):
                    im_row_y[y] = im_row_y[y-1] + \
                            im_slice[x + y * out_im.shape[2]]
                # downsample into out_im
                sum_left = mirrored_interpolate_1d(im_row_y,\
                            in_im.shape[1], offset[1] - 1.0)
                for y in range(out_im.shape[1]):
                    sum_right = mirrored_interpolate_1d(im_row_y,\
                            in_im.shape[1],
                            (y + 1.0) * scale_factor_y - 1.0 + offset[1])
                    out_im[z, y, x] = \
                            <np.float64_t>((sum_right - sum_left) /\
                                scale_factor_y)
                    sum_left = sum_right

        free(im_slice)
        free(im_row_x)
        free(im_row_y)

cpdef void downsample_Z(np.float64_t[:,:,:] in_im,
                        np.int16_t[:, :, :] out_im,
                        np.float64_t[:] scale_factor,
                        np.float64_t[:] offset,
                        int thread_num = -1, 
                        bint is_mask = False) nogil:
    """
        Downsampling of an image in the z direction.

        in_im is resampled to the shape of out_im. Grey values results are
        resampled in z direction.
        The algorithm is based on 1D "integral images" similar to the method
        used in the SURF algorithm. The method allows for arbitrary scaling
        factors. However only downsampling is supported so that the size of
        out_im must not exceed the size of in_im in any direction.

        Args:
            in_im (memoryview, np.float64):  input image
            out_im (memoryview, np.int16): output image
            scale_factor (numpy array): 3D Scaling factor > 1 for downsampling
            offset (memoryview, np.float64): offset to be added to source
                                             image coordinates
            thread_num (int): number of threads to use. Values < 1 indicate
                that all CPU threads should be used.
            is_mask (bool): True if image should be treated as a binary mask
    """

    # check image sizes
    if in_im.shape[1] != out_im.shape[1] or in_im.shape[2] != out_im.shape[2]:
        abort()

    # only downsampling is supported
    if in_im.shape[0] < out_im.shape[0]:
        abort()

    if thread_num < 1:
        thread_num = openmp.omp_get_max_threads()

    cdef index_t x, y, z
    cdef double *im_row_z
    cdef double scale_factor_z = scale_factor[2]
    cdef double sum_left, sum_right, g_val

    with nogil, parallel(num_threads=thread_num):
        im_row_z   = <double *>malloc(sizeof(double) * in_im.shape[0])
        if im_row_z == NULL:
            abort()
        # work y-slice by y-slice
        for y in prange(in_im.shape[1], schedule='dynamic'):
            # perform stack by stack resampling and store result in im_row_z
            for x in range(in_im.shape[2]):
                # accumulate into im_row_z
                im_row_z[0]=in_im[0, y, x]
                for z in range(1, in_im.shape[0]):
                    im_row_z[z] = im_row_z[z-1] + in_im[z, y, x]
                # downsample into out_im
                sum_left = mirrored_interpolate_1d(im_row_z,
                            in_im.shape[0], offset[2] - 1.0)
                for z in range(out_im.shape[0]):
                    sum_right = mirrored_interpolate_1d(im_row_z,
                            in_im.shape[0],
                            (z + 1.0) * scale_factor_z - 1.0 + offset[2])
                    g_val = (sum_right - sum_left) / scale_factor_z
                    if not is_mask:
                        out_im[z, y, x] = my_round_int16(g_val)
                    else:
                        if g_val > 0.25:
                            out_im[z, y, x] = <np.int16_t>1
                        else:
                            out_im[z, y, x] = <np.int16_t>0
                    sum_left = sum_right
        free(im_row_z)

cpdef void downsample(np.int16_t[:,:,:] in_im,
                      np.int16_t[:, :, :] out_im,
                      int thread_num = -1,
                      bint is_mask = False):
    """
        Downsampling of a 3D image.

        in_im is resampled to the shape of out_im.
        The algorithm is based on 1D "integral images" similar to the method
        used in the SURF algorithm. The method allows for arbitrary scaling
        factors. However only downsampling is supported so that the size of
        out_im must not exceed the size of in_im in any direction. Resampling
        is performed in two steps. First, in-slice resampling (x and y) of
        in_im is performed into an intermediate float image which is
        afterwards resampled in z direction into out_im.

        Args:
            in_im:  input image  (memoryview, np.int16)
            out_im: output image (memoryview, np.int16)
            thread_num: number of threads to use. Values < 1 indicate
                that all CPU threads should be used.
            is_mask (bool): True if image should be treated as a binary mask
    """
    cdef np.float64_t[:,:,:] tmp_im = np.zeros((in_im.shape[0], 
        out_im.shape[1], out_im.shape[2]), dtype=np.float64)
    cdef np.float64_t[:] offset = np.zeros(3, dtype=np.float64)
    cdef np.float64_t[:] scale_factor = np.zeros(3, dtype=np.float64)
    cdef int i
    for i in range(3):
        scale_factor[i] = <double>in_im.shape[2-i] / <double>out_im.shape[2-i]
    downsample_XY(in_im, tmp_im, scale_factor, offset, thread_num)
    downsample_Z(tmp_im, out_im, scale_factor, offset, thread_num, is_mask)
    del(tmp_im)


cpdef void downsample_with_offset(np.int16_t[:,:,:] in_im,
                      np.int16_t[:, :, :] out_im,
                      np.float64_t[:, :, :] tmp_im,
                      np.float64_t[:] scale_factor,
                      np.float64_t[:] offset,
                      int thread_num = -1,
                      bint is_mask = False):
    """
        Downsampling of a 3D image part.

        in_im is resampled to the shape of out_im.
        The algorithm is based on 1D "integral images" similar to the method
        used in the SURF algorithm. The method allows for arbitrary scaling
        factors. However only downsampling is supported so that the size of
        out_im must not exceed the size of in_im in any direction. Resampling
        is performed in two steps. First, in-slice resampling (x and y) of
        in_im is performed into an intermediate float image which is
        afterwards resampled in z direction into out_im.

        Args:
            in_im (memoryview, np.int16):  input image
            out_im (memoryview, np.int16): output image
            tmp_im (memoryview, np.float64): temporary image holding in-plane
                                            downsampling results
            scale_factor (numpy array): 3D scaling factor > 1 for downsampling
            offset (memoryview, np.float64): offset to be added to source
                                             image coordinates
            thread_num (int): number of threads to use. Values < 1 indicate
                that all CPU threads should be used.
            is_mask bool): True if image should be treated as a binary mask
    """

    downsample_XY(in_im, tmp_im, scale_factor, offset, thread_num)
    downsample_Z(tmp_im, out_im, scale_factor, offset, thread_num, is_mask)

cpdef void interpolate_block(np.int16_t[:,:,:] fixed_im,
                      np.int16_t[:, :, :] moving_im,
                      np.float64_t[:, :] t_matrix_fixed_to_moving,
                      int thread_num = -1,
                      bint is_mask = False):

    if thread_num < 1:
        thread_num = openmp.omp_get_max_threads()

    cdef index_t x, y, z
    cdef double interpolation_result
    cdef np.int16_t out_grey_val
    with nogil, parallel(num_threads=thread_num):
        p_fixed = <double *>malloc(3 * sizeof(double))
        p_moving = <double *>malloc(3 * sizeof(double))
        for z in prange(fixed_im.shape[0], schedule='dynamic'):
            p_fixed[2] = <double>z
            for y in range(fixed_im.shape[1]):
                p_fixed[1] = <double>y
                for x in range(fixed_im.shape[2]):
                    p_fixed[0] = <double>x
                    transform_point(t_matrix_fixed_to_moving, 
                                    p_fixed, p_moving)
                    interpolation_result = \
                            mirrored_interpolate_3d(moving_im, p_moving)
                    if is_mask:
                        fixed_im[z, y, x] =\
                            1 if interpolation_result > 0.4 else 0
                    else:
                        fixed_im[z, y, x] = \
                                my_round_int16(interpolation_result)
        free(p_fixed)
        free(p_moving)


cpdef void fill_gm_image(np.int16_t[:,:,:] in_block, 
        np.int16_t[:,:,:] mask_block, np.float32_t[:,:,:] gm_block, 
        int thread_num):

    cdef np.float64_t d_x, d_y, d_z
    cdef index_t x, y, z, z_in
    for z in prange(gm_block.shape[0], nogil=True, 
                schedule='dynamic', num_threads=thread_num):
        z_in = z + 1
        for y in range(1, gm_block.shape[1]-1):
            for x in range(1, gm_block.shape[2]-1):
                if mask_block is None or mask_block[z, y, x] != 0:
                    d_x = in_block[z_in, y, x+1] - in_block[z_in, y, x-1]
                    d_y = in_block[z_in, y+1, x] - in_block[z_in, y-1, x]
                    d_z = in_block[z_in+1, y, x] - in_block[z_in-1, y, x]
                    gm_block[z, y, x] = \
                        <np.float32_t> d_x * d_x + d_y * d_y + d_z * d_z

cpdef np.float32_t gm_max(np.float32_t[:,:,:] gm_block): 
    cdef index_t x, y, z
    cdef np.float32_t gm_max = -1
    for z in range(gm_block.shape[0]):
        for y in range(1, gm_block.shape[1]-1):
            for x in range(1, gm_block.shape[2]-1):
                gm_max = max(gm_max, gm_block[z, y, x])
    return gm_max

cpdef void fill_gm_histogram(np.float32_t[:,:,:] gm_block,
        np.uint64_t[:] hist, double gm_max):

    cdef np.float64_t gm
    cdef index_t x, y, z
    cdef size_t bin_nr, n_bins = hist.shape[0]
    for z in range(gm_block.shape[0]):
        for y in range(1, gm_block.shape[1]-1):
            for x in range(1, gm_block.shape[2]-1):
                gm = gm_block[z, y, x]
                bin_nr = my_round_size_t((gm / gm_max) * n_bins)
                bin_nr = min(bin_nr, n_bins-1)
                hist[bin_nr] += 1


cpdef void gm_masking(np.int16_t[:,:,:] in_im, 
        np.int16_t[:,:,:] out_im, np.float32_t[:,:,:] gm_im,
        int z_offset, double gm_limit, 
        bint is_moving): 

    cdef index_t x, y, z, z_out
    
    for z in range(gm_im.shape[0]):
        z_out = z + z_offset
        for y in range(1, gm_im.shape[1]-1):
            for x in range(1, gm_im.shape[2]-1):
                if gm_im[z, y, x] > gm_limit:
                    out_im[z_out, y, x] = in_im[z_out, y, x]
                    set_gm_environment(in_im, out_im, 
                        x, y, z_out, is_moving)


cpdef void do_run_length_encoding(np.int16_t[:,:,:] im, int thread_num) nogil:
    cdef size_t slice_pos=0, rle_start=0, rle_len
    cdef size_t pixels_per_slice = im.shape[1] * im.shape[2]
    cdef bint rle_active=False

    cdef index_t z

    for z in prange(im.shape[0], nogil=True, 
                schedule='dynamic', num_threads=thread_num):
        rle_active = False
        rle_len = 0
        for slice_pos in range(pixels_per_slice-1):
            # switch on rle for two consecutive zero output voxels 
            if not rle_active and \
                    im[z, slice_pos // im.shape[2], 
                      slice_pos % im.shape[2]] == 0 and \
                    im[z, (slice_pos + 1) // im.shape[2], 
                      (slice_pos + 1) % im.shape[2]] == 0 and \
                    slice_pos < pixels_per_slice - 2:
                rle_start = slice_pos
                rle_active = True
                rle_len = 0
            # count rle pixels        
            if rle_active and im[z, slice_pos // im.shape[2],
                                     slice_pos % im.shape[2]] <= 0:
                rle_len = rle_len + 1  
            # store rle result for next positive grey value
            # or if the end of the slice has been reached
            if rle_active and \
                    (im[z, slice_pos // im.shape[2], 
                            slice_pos % im.shape[2]] > 0 or \
                     slice_pos >= pixels_per_slice - 2):
                rle_active = False
                # assume cache line size of 64 and restrict rle
                # usage to certain cache misses
                if rle_len > 32 and rle_len < 2**15 * (2**15 - 2**14):
                    im[z, rle_start // im.shape[2], 
                           rle_start % im.shape[2]] = \
                           <np.int16_t> -(rle_len // 2**15  + 2**14) 
                    im[z, (rle_start + 1) // im.shape[2], 
                           (rle_start+1) % im.shape[2]] = \
                           <np.int16_t> -(rle_len % 2**15)


cpdef void remove_rle(np.int16_t[:, :] in_slice, np.int16_t[:, :] out_slice):
    cdef index_t x, y, x_size = in_slice.shape[1]
    cdef size_t slice_pos=0 
    cdef size_t nr_of_pixels = in_slice.shape[0] * in_slice.shape[1]
    
    while slice_pos < nr_of_pixels:
        y = slice_pos // x_size
        x = slice_pos % x_size
        if in_slice[y, x] > -2**14:
            out_slice[y, x] = in_slice[y, x]
            slice_pos = slice_pos + 1
        else:
            out_slice[y, x] = 0
            out_slice[(slice_pos+1) // x_size, (slice_pos+1) % x_size] = 0
            slice_pos = slice_pos + 2
        


cpdef void add_inertia_tensor_slice(np.int16_t[:, :] im,
        index_t z, np.float64_t[:, :] voxel_to_world_matrix, 
        np.int64_t[:] N, np.float64_t[:] centre, np.float64_t[:, :] TI):
    
    cdef index_t x, y
    cdef double *p_v = <double *>malloc(sizeof(double) * 3)
    cdef double *p_w = <double *>malloc(sizeof(double) * 3)

    p_v[2] = z
    for y in range(1, im.shape[0]-1):
        p_v[1] = y
        for x in range(1, im.shape[1]-1):
            if im[y, x] > 0:
                p_v[0] = x
                transform_point(voxel_to_world_matrix, p_v, p_w)
                N[0] += 1
                centre[0] += p_w[0]
                centre[1] += p_w[1]
                centre[2] += p_w[2]
                TI[0, 0] += p_w[1] * p_w[1] + p_w[2] * p_w[2]
                TI[0, 1] -= p_w[0] * p_w[1]
                TI[0, 2] -= p_w[0] * p_w[2]
                TI[1, 1] += p_w[0] * p_w[0] + p_w[2] * p_w[2]
                TI[1, 2] -= p_w[1] * p_w[2]
                TI[2, 2] += p_w[0] * p_w[0] + p_w[1] * p_w[1]
    free(p_v)
    free(p_w)

cdef void lcc_process_environment(
        np.int16_t[:, :, :] fixed_im, 
        np.int16_t[:, :, :] moving_im,
        np.float64_t[:, :] trans_mat_vox, 
        np.int64_t[:] N_buf,
        np.float64_t[:] lcc_buf, 
        index_t x_f, index_t y_f, index_t z_f, 
        double *p_fixed, double *p_t,
        np.int16_t *b_buf, np.float64_t *t_buf,
        np.int16_t b_val_c, np.float64_t t_val_c) nogil:

    cdef double b, t, b_avg, t_avg, b_i_t_i, b_i_2, t_i_2, denom
    cdef bint pixel_outside
    cdef index_t i, x_l, y_l, z_l, d_z, d_z_y

    i=0
    b_avg = 0.0
    t_avg = 0.0
    pixel_outside = False
    for z_l in range(-1, 2):
        if pixel_outside:
            break
        d_z = my_abs(z_l)
        for y_l in range(-1+d_z, 2-d_z):
            d_z_y = d_z + my_abs(y_l)
            for x_l in range(-1+d_z_y, 2-d_z_y):
                if d_z_y == 0 and x_l == 0:
                    b_buf[i] = b_val_c
                    t_buf[i] = t_val_c
                else:
                    b_buf[i] = my_abs(fixed_im[z_f+z_l, y_f+y_l, x_f+x_l])
                    p_fixed[0] = x_f + x_l
                    p_fixed[1] = y_f + y_l 
                    p_fixed[2] = z_f + z_l
                    transform_point(trans_mat_vox, p_fixed, p_t)
                    t_buf[i] = my_fabs(masked_interpolate_3d( moving_im, p_t))
                if t_buf[i] < 1 or b_buf[i] < 1:
                    pixel_outside = True
                else:
                    b_avg = b_avg + <double> b_buf[i]
                    t_avg = t_avg + t_buf[i]
                    i = i + 1
    if not pixel_outside:
        b_avg = b_avg / 7
        t_avg = t_avg / 7
        b_i_t_i = 0
        b_i_2 = 0
        t_i_2 = 0
        for i in range(7):
            b = <double> b_buf[i] - b_avg
            t = t_buf[i] - t_avg
            b_i_t_i = b_i_t_i + b * t
            b_i_2 = b_i_2 + b * b
            t_i_2 = t_i_2 + t * t
        denom = b_i_2 * t_i_2
        if denom > 1e-20:
            N_buf[z_f] = N_buf[z_f] + 1
            lcc_buf[z_f] = lcc_buf[z_f] +\
                    1.0 - (b_i_t_i * b_i_t_i) / denom

cpdef void lcc_block(
        np.int16_t[:, :, :] fixed_im, 
        np.int16_t[:, :, :] moving_im,
        np.float64_t[:, :] trans_mat_vox, 
        np.int64_t[:] N_buf,
        np.float64_t[:] lcc_buf, 
        np.int32_t thread_num,
        bint use_rle=True):

    cdef index_t z_f, y_f, x_f, x_size = fixed_im.shape[2]
    cdef np.int16_t *b_buf
    cdef np.int16_t b_val_c
    cdef np.float64_t *t_buf
    cdef np.float64_t t_val_c
    cdef double *p_fixed
    cdef double *p_t
    cdef size_t slice_index
    cdef size_t pixels_in_slice = fixed_im.shape[1] * fixed_im.shape[2]

    if thread_num < 1:
        thread_num = min(openmp.omp_get_max_threads(), fixed_im.shape[1]-4)

    with nogil, parallel(num_threads=thread_num):
        b_buf = <np.int16_t *>malloc(7 * sizeof(np.int16_t))
        t_buf = <np.float64_t *>malloc(7 * sizeof(np.float64_t))
        p_fixed = <double *>malloc(3 * sizeof(double))
        p_t = <double *>malloc(3 * sizeof(double))

        for z_f in prange(2, fixed_im.shape[0]-2, schedule='dynamic'):
            if use_rle:
                slice_index = 0
                while slice_index < pixels_in_slice:
                    x_f = slice_index % x_size
                    y_f = slice_index // x_size
                    b_val_c = fixed_im[z_f, y_f, x_f] 
                    if  b_val_c > 0 and \
                            x_f > 1 and x_f < fixed_im.shape[2] - 2 and \
                            y_f > 1 and y_f < fixed_im.shape[1] - 2:
                        p_fixed[2] = <double>z_f
                        p_fixed[1] = <double>y_f
                        p_fixed[0] = <double>x_f
                        transform_point(trans_mat_vox, p_fixed, p_t)
                        t_val_c = masked_interpolate_3d(moving_im, p_t)
                        if t_val_c > 0:
                            lcc_process_environment(fixed_im, moving_im,
                                trans_mat_vox, N_buf, lcc_buf, 
                                x_f, y_f, z_f, p_fixed, p_t,
                                b_buf, t_buf, b_val_c, t_val_c)
                    if b_val_c <= -2**14 and \
                            slice_index < pixels_in_slice - 1: 
                        slice_index = slice_index +\
                                      (-b_val_c - 2**14) * 2**15 -\
                                      fixed_im[z_f, 
                                               (slice_index + 1) // x_size,
                                               (slice_index + 1) % x_size]
                    else:
                        slice_index = slice_index + 1
            else:
                for y_f in range(2, fixed_im.shape[1]-2):
                    for x_f in range(2, fixed_im.shape[2]-2):
                        b_val_c = fixed_im[z_f, y_f, x_f] 
                        if  b_val_c > 0:
                            p_fixed[2] = <double>z_f
                            p_fixed[1] = <double>y_f
                            p_fixed[0] = <double>x_f
                            transform_point(trans_mat_vox, p_fixed, p_t)
                            t_val_c = masked_interpolate_3d(moving_im, p_t)
                            if t_val_c > 0:
                                lcc_process_environment(fixed_im, moving_im,
                                    trans_mat_vox, N_buf, lcc_buf, 
                                    x_f, y_f, z_f, p_fixed, p_t,
                                    b_buf, t_buf, b_val_c, t_val_c)
        free(b_buf)
        free(t_buf)
        free(p_fixed)
        free(p_t)

cdef void lcc_process_environment_with_derivatives(
        np.int16_t[:, :, :] fixed_im, 
        np.int16_t[:, :, :] moving_im,
        np.float64_t[:, :] fixed_v2w_matrix,
        np.float64_t[:, :] trans_mat_vox, 
        np.float64_t[:, :] trans_mat_gradient, 
        np.int32_t n_p,
        np.int64_t[:] N_buf,
        np.float64_t[:] lcc_buf, 
        np.float64_t[:, :] gradient_buf, 
        np.float64_t[:, :, :] hessian_buf, 
        index_t x_f, index_t y_f, index_t z_f, 
        double *p_fixed, double *p_t,
        np.int16_t *t_buf, np.float64_t *b_buf,
        double *grad_v, double *grad_w,
        double *derivative_buf, 
        double *derivative_avg,
        double *t_t_der_sum,
        double *b_t_der_sum,
        double *m_ij_der,
        np.int16_t t_val_c, np.float64_t b_val_c) nogil:

    cdef index_t x_l, y_l, z_l, d_z, d_z_y, x_t, y_t, z_t, i, k, l
    cdef double b_2_sum, t_2_sum, b_t_sum, b_i, m_ij
    cdef double b, t, b_avg, t_avg, b_i_t_i, b_i_2, t_i_2, denom, c_j
    cdef bint pixel_outside
    cdef np.int16_t b_int

    # fill moving buffer (b_buf due to image swapping ...)
    for z_l in range(-1, 2):
        d_z = my_abs(z_l)
        for y_l in range(-1+d_z, 2-d_z):
            d_z_y = d_z + my_abs(y_l)
            for x_l in range(-1+d_z_y, 2-d_z_y):
                if d_z_y == 0 and x_l == 0:
                    t = b_val_c
                else:
                    p_fixed[0] = x_f + x_l
                    p_fixed[1] = y_f + y_l 
                    p_fixed[2] = z_f + z_l
                    transform_point(trans_mat_vox, p_fixed, p_t)
                    t = my_fabs(masked_interpolate_3d( moving_im, p_t))
                if t < 1:
                    return
                set_3d_array_3(b_buf, z_l+1, y_l+1, x_l+1, t)

    # fill fixed buffer (t_buf due to image swapping ...)
    for z_l in range(-2, 3):
        d_z = my_abs(z_l)
        for y_l in range(-2+d_z, 3-d_z):
            d_z_y = d_z + my_abs(y_l)
            for x_l in range(-2+d_z_y, 3-d_z_y):
                if d_z_y == 0 and x_l == 0:
                    b_int = t_val_c
                else:
                    b_int = my_abs(fixed_im[z_f+z_l, y_f+y_l, x_f+x_l])
                if b_int < 1:
                    return
                else:
                    set_3d_array_5(t_buf, z_l+2, y_l+2, x_l+2, b_int)

    # add up values and calculate gradients
    i=0
    t_avg = 0.0
    b_avg = 0.0
    for z_l in range(-1, 2):
        d_z = my_abs(z_l)
        z_t = z_l + 2
        p_fixed[2] = <double>(z_f + z_l)
        for y_l in range(-1+d_z, 2-d_z):
            d_z_y = d_z + my_abs(y_l)
            y_t = y_l + 2
            p_fixed[1] = <double>(y_f + y_l)
            for x_l in range(-1+d_z_y, 2-d_z_y):
                x_t = x_l + 2
                b_avg = b_avg + get_3d_array_3(b_buf, z_l+1, y_l+1, x_l+1)
                t_avg = t_avg + <double> get_3d_array_5(t_buf, z_t, y_t, x_t)
                # gradient in voxel coordinates
                grad_v[0] = <double>(
                                get_3d_array_5(t_buf, z_t, y_t, x_t+1) -\
                                get_3d_array_5(t_buf, z_t, y_t, x_t-1))
                grad_v[1] = <double>(
                                get_3d_array_5(t_buf, z_t, y_t+1, x_t) -\
                                get_3d_array_5(t_buf, z_t, y_t-1, x_t))
                grad_v[2] = <double>(
                                get_3d_array_5(t_buf, z_t+1, y_t, x_t) -\
                                get_3d_array_5(t_buf, z_t-1, y_t, x_t))
                # convert into world coordinates
                transform_vector(trans_mat_gradient, grad_v, grad_w)
                # transform voxel position into world coordinates
                p_fixed[0] = <double>x_f + x_l
                transform_point(fixed_v2w_matrix, p_fixed, p_t)
                # calculate derivatives
                if n_p == 6:
                    # translation
                    for l in range(3):
                        derivative_buf[i * n_p + l] = grad_w[l]
                    # rotation
                        derivative_buf[i * n_p + 3] = \
                            grad_w[2]*p_t[1] - grad_w[1]*p_t[2]
                        derivative_buf[i * n_p + 4] = \
                            grad_w[0]*p_t[2] - grad_w[2]*p_t[0]
                        derivative_buf[i * n_p + 5] = \
                            grad_w[1]*p_t[0] - grad_w[0]*p_t[1]
                else:
                    # homogenous coordinates
                    p_t[3] = 1.0
                    for k in range(n_p):
                        derivative_buf[i * n_p + k] = grad_w[k/4] * p_t[k%4]
                i = i + 1

    # calculate average grey values
    b_avg = b_avg / 7.0
    t_avg = t_avg / 7.0

    # calculate average derivative values
    for k in range(n_p):
        derivative_avg[k] = 0.0
        for i in range(7):
            derivative_avg[k] = derivative_avg[k] +  derivative_buf[i*n_p+k] 
        derivative_avg[k] = derivative_avg[k] / 7.0

    # calculate sums
    b_2_sum = 0.0
    t_2_sum = 0.0
    b_t_sum = 0.0
    for i in range(n_p):
       t_t_der_sum[i] = 0.0
       b_t_der_sum[i] = 0.0
    i=0
    for z_l in range(-1, 2):
        d_z = my_abs(z_l)
        for y_l in range(-1+d_z, 2-d_z):
            d_z_y = d_z + my_abs(y_l)
            for x_l in range(-1+d_z_y, 2-d_z_y):
                b_i = get_3d_array_3(b_buf, z_l+1, y_l+1, x_l+1) - b_avg
                t_i = <double>get_3d_array_5(t_buf, z_l+2, y_l+2, x_l+2) - t_avg
                b_2_sum = b_2_sum + b_i * b_i
                t_2_sum = t_2_sum + t_i * t_i
                b_t_sum = b_t_sum + b_i * t_i
                for k in range(n_p):
                    t_t_der_sum[k] = t_t_der_sum[k] +\
                        t_i * (derivative_buf[i*n_p+k] - derivative_avg[k])
                    b_t_der_sum[k] = + b_t_der_sum[k] +\
                        b_i * (derivative_buf[i*n_p+k] - derivative_avg[k])
                i = i + 1
        
    if b_2_sum < 1e-20 or t_2_sum < 1e-20:
        return
            
    # update lcc, gradient and hessian buffers
    i=0
    for z_l in range(-1, 2):
        d_z = my_abs(z_l)
        for y_l in range(-1+d_z, 2-d_z):
            d_z_y = d_z + my_abs(y_l)
            for x_l in range(-1+d_z_y, 2-d_z_y):
                c_j = 2.0 / b_2_sum
                b_i = get_3d_array_3(b_buf, z_l+1, y_l+1, x_l+1) - b_avg
                t_i = <double>get_3d_array_5(t_buf, z_l+2, y_l+2, x_l+2) - t_avg
                m_ij = b_i - b_t_sum / t_2_sum * t_i
                lcc_buf[z_f] = lcc_buf[z_f] + c_j * m_ij * m_ij
                for k in range(n_p):
                    m_ij_der[k] = 2 * b_t_sum * t_t_der_sum[k] * t_i / \
                                    (t_2_sum * t_2_sum) -\
                        (b_t_der_sum[k] * t_i + b_t_sum *\
                            (derivative_buf[i*n_p+k] - derivative_avg[k])) / \
                        t_2_sum
                for k in range(n_p):
                    gradient_buf[z_f, k] = gradient_buf[z_f, k] + \
                                           c_j * m_ij * m_ij_der[k]
                    for l in range(k, n_p):
                        hessian_buf[z_f, k, l] = hessian_buf[z_f, k, l] + \
                            c_j * m_ij_der[k] * m_ij_der[l]
                i = i + 1     
    N_buf[z_f] = N_buf[z_f] + 1

cpdef void lcc_block_with_derivatives(
        np.int16_t[:, :, :] fixed_im, 
        np.int16_t[:, :, :] moving_im,
        np.float64_t[:, :] fixed_v2w_matrix,
        np.float64_t[:, :] trans_mat_vox, 
        np.float64_t[:,:] trans_mat_gradient,
        np.int32_t n_p,
        np.int64_t[:] N_buf,
        np.float64_t[:] lcc_buf, 
        np.float64_t[:, :] gradient_buf,
        np.float64_t[:, :, :] hessian_buf,
        np.int32_t thread_num,
        bint use_rle=True):

    cdef index_t z_f, y_f, x_f, x_size = fixed_im.shape[2]
    cdef np.int16_t *b_buf
    cdef np.float64_t *t_buf
    cdef double *p_fixed
    cdef double *p_t
    cdef double *grad_v
    cdef double *grad_w
    cdef double *derivative_buf
    cdef double *derivative_avg
    cdef double *t_t_der_sum
    cdef double *b_t_der_sum
    cdef bint pixel_outside
    cdef np.int16_t b_val_c
    cdef np.float64_t t_val_c
    cdef size_t slice_index
    cdef size_t pixels_in_slice = fixed_im.shape[1] * fixed_im.shape[2]

    if thread_num < 1:
        thread_num = min(openmp.omp_get_max_threads(), fixed_im.shape[1]-4)

    with nogil, parallel(num_threads=thread_num):
        b_buf = <np.int16_t *>malloc(5 * 5 * 5 * sizeof(np.int16_t))
        t_buf = <np.float64_t *>malloc(3 * 3 * 3 * sizeof(np.float64_t))
        p_fixed = <double *>malloc(3 * sizeof(double))
        p_t = <double *>malloc(4 * sizeof(double))
        grad_v = <double *>malloc(3 * sizeof(double))
        grad_w = <double *>malloc(3 * sizeof(double))
        derivative_buf = <double *>malloc(n_p * 7 * sizeof(double))
        derivative_avg = <double *>malloc(n_p * sizeof(double))
        t_t_der_sum = <double *>malloc(n_p * sizeof(double))
        b_t_der_sum = <double *>malloc(n_p * sizeof(double))
        m_ij_der = <double *>malloc(n_p * sizeof(double))

        for z_f in prange(2, fixed_im.shape[0]-2, schedule='dynamic'):
            if use_rle:
                slice_index = 0
                while slice_index < pixels_in_slice:
                    x_f = slice_index % x_size
                    y_f = slice_index // x_size
                    b_val_c = fixed_im[z_f, y_f, x_f] 
                    if  b_val_c > 0 and \
                            x_f > 1 and x_f < fixed_im.shape[2] - 2 and \
                            y_f > 1 and y_f < fixed_im.shape[1] - 2:
                        p_fixed[2] = <double>z_f
                        p_fixed[1] = <double>y_f
                        p_fixed[0] = <double>x_f
                        transform_point(trans_mat_vox, p_fixed, p_t)
                        t_val_c = masked_interpolate_3d(moving_im, p_t)
                        if t_val_c > 0:
                            lcc_process_environment_with_derivatives(
                                        fixed_im,  moving_im,
                                        fixed_v2w_matrix,
                                        trans_mat_vox, 
                                        trans_mat_gradient, 
                                        n_p,
                                        N_buf, lcc_buf, 
                                        gradient_buf, hessian_buf, 
                                        x_f, y_f, z_f, 
                                        p_fixed, p_t,
                                        b_buf, t_buf,
                                        grad_v, grad_w,
                                        derivative_buf, 
                                        derivative_avg,
                                        t_t_der_sum,
                                        b_t_der_sum,
                                        m_ij_der,
                                        b_val_c, t_val_c)
                    if b_val_c <= -2**14 and \
                            slice_index < pixels_in_slice - 1: 
                        slice_index = slice_index +\
                                      (-b_val_c - 2**14) * 2**15 -\
                                      fixed_im[z_f, 
                                               (slice_index + 1) // x_size,
                                               (slice_index + 1) % x_size]
                    else:
                        slice_index = slice_index + 1
            else:
                for y_f in range(2, fixed_im.shape[1]-2):
                    for x_f in range(2, fixed_im.shape[2]-2):
                        b_val_c = fixed_im[z_f, y_f, x_f] 
                        if b_val_c > 0:
                            p_fixed[2] = <double>z_f
                            p_fixed[1] = <double>y_f
                            p_fixed[0] = <double>x_f
                            transform_point(trans_mat_vox, p_fixed, p_t)
                            t_val_c = masked_interpolate_3d(moving_im, p_t)
                            if t_val_c > 0:
                                lcc_process_environment_with_derivatives(
                                        fixed_im,  moving_im,
                                        fixed_v2w_matrix,
                                        trans_mat_vox, 
                                        trans_mat_gradient, 
                                        n_p,
                                        N_buf, lcc_buf, 
                                        gradient_buf, hessian_buf, 
                                        x_f, y_f, z_f, 
                                        p_fixed, p_t,
                                        b_buf, t_buf,
                                        grad_v, grad_w,
                                        derivative_buf, 
                                        derivative_avg,
                                        t_t_der_sum,
                                        b_t_der_sum,
                                        m_ij_der,
                                        b_val_c, t_val_c)
        free(b_buf)
        free(t_buf)
        free(p_fixed)
        free(p_t)
        free(grad_v)
        free(grad_w)
        free(derivative_buf)
        free(derivative_avg)
        free(t_t_der_sum)
        free(b_t_der_sum)
        free(m_ij_der)
