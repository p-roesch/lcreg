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
import cython

ctypedef int index_t

cdef np.float64_t HALF_PLUS_EPSILON = 0.5 + 10.0 * np.finfo(np.float64).eps

cdef inline double my_fabs(double x) nogil:
    """
    simple inline fabs method
    """
    return x if x > 0 else -x

cdef inline int my_abs(int x) nogil:
    """
    simple inline abs method
    """
    return x if x > 0 else -x


cdef inline int my_round_index_t(double x) nogil:
    return <index_t>(x + HALF_PLUS_EPSILON)

cdef inline np.int16_t my_round_int16(double x) nogil:
    return <np.int16_t>(x + HALF_PLUS_EPSILON)

cdef inline size_t my_round_size_t(double x) nogil:
    return <size_t> (x + HALF_PLUS_EPSILON)

cdef inline void transform_point(np.float64_t[:, :] m,
                                 double *pi, double *po) nogil:
    po[0] = pi[0] * m[0][0] + pi[1] * m[0][1] + pi[2] * m[0][2] + m[0][3]
    po[1] = pi[0] * m[1][0] + pi[1] * m[1][1] + pi[2] * m[1][2] + m[1][3]
    po[2] = pi[0] * m[2][0] + pi[1] * m[2][1] + pi[2] * m[2][2] + m[2][3]

    
cdef inline void transform_vector(np.float64_t[:, :] m,
                                 double *vi, double *vo) nogil:
    vo[0] = vi[0] * m[0][0] + vi[1] * m[0][1] + vi[2] * m[0][2]
    vo[1] = vi[0] * m[1][0] + vi[1] * m[1][1] + vi[2] * m[1][2]
    vo[2] = vi[0] * m[2][0] + vi[1] * m[2][1] + vi[2] * m[2][2]

cdef inline double w_linear(double x) nogil:
    """
    Kernel for linear interpolation.
    
    Args:
        x: offset value, range [0, 1]
    
    Returns:
        kernel value
    """
    return 1.0 - my_fabs(x)
    
cdef inline bint is_within_image_border_1(double *p,
                                 np.int16_t[:, :, :] a) nogil:
    """Check if voxel coordinate is within image array
    
        Note that the access order of the array is a[z, y, x] whereas
        x is a vector (x, y, z)
    
        Args:
            x: numpy array containing the position
            a: grey value array
        Returns:
            True if x in inside the image excluding 1 voxel border,
            False otherwise
    """

    return p[0] >= 0.5 and p[1] >= 0.5 and p[2] >= 0.5 and \
        p[2] < a.shape[0] - 1.5 and p[1] < a.shape[1] - 1.5 and \
        p[0] < a.shape[2] - 1.5

cdef inline bint is_within_image(double *p,
                                 np.int16_t[:, :, :] a) nogil:
    """Check if voxel coordinate is within image array
    
        Note that the access order of the array is a[z, y, x] whereas
        x is a vector (x, y, z)
    
        Args:
            x: numpy array containing the position
            a: grey value array
        Returns:
            True if x in inside the image, False otherwise
    """

    return p[0] >= -0.5 and p[1] >= -0.5 and p[2] >= -0.5 and \
        p[2] < a.shape[0] - 0.5 and p[1] < a.shape[1] - 0.5 and \
        p[0] < a.shape[2] - 0.5

cdef inline void set_gm_environment(
        np.int16_t[:, :, :] in_im, np.int16_t[:, :, :] out_im,
        np.int32_t x, np.int32_t y, np.int32_t z, bint is_moving):
    cdef index_t x_d, y_d, z_d 
    cdef int delta_z, delta_y, delta_x

    if is_moving:
        for z_d in range(max(z-2, 0), min(z+3, out_im.shape[0])):
            for y_d in range(max(y-2, 0), min(y+3, out_im.shape[1])):
                for x_d in range(max(x-2, 0), min(x+3, out_im.shape[2])):
                    if out_im[z_d, y_d, x_d] == 0:
                        out_im[z_d, y_d, x_d] = -in_im[z_d, y_d, x_d]
    else:
        for z_d in range(max(z-2, 0), min(z+3, out_im.shape[0])):
            delta_z = my_abs(z_d-z)
            for y_d in range(max(y-2+delta_z, 0), 
                             min(y+3-delta_z, out_im.shape[1])):
                delta_y = my_abs(y_d-y)
                for x_d in range(max(x-2+delta_y+delta_z, 0), 
                                 min(x+3-delta_z-delta_y, out_im.shape[2])):
                    if out_im[z_d, y_d, x_d] == 0:
                        out_im[z_d, y_d, x_d] = -in_im[z_d, y_d, x_d]

cdef inline np.int16_t get_3d_array_5(np.int16_t *arr, int z, int y, 
        int x) nogil:
    return arr[x + y * 5 + z * 25]

cdef inline void set_3d_array_5(np.int16_t *arr, int z, int y, int x,
        np.int16_t val) nogil:
    arr[x + y * 5 + z * 25] = val

cdef inline np.float64_t get_3d_array_3(np.float64_t *arr, 
                                        int z, int y, int x) nogil:
    return arr[x + y * 3 + z * 9]

cdef inline void set_3d_array_3(np.float64_t *arr, int z, int y, int x,
        np.float64_t val) nogil:
    arr[x + y * 3 + z * 9] = val


cpdef void rescale_gvals(np.float64_t[:,:,:] in_im,
        np.int16_t[:,:,:] out_im, double g_min, double g_max,
        np.int16_t out_min, np.int16_t out_max, np.float64_t epsilon,
        bint is_mask)
cdef double mirrored_interpolate_1d(double *a, index_t N, double x) nogil
cpdef double mirrored_interpolate_1d_py(np.float64_t[:] a, double x) nogil

cdef double masked_interpolate_3d(np.int16_t[:, :, :] a, double *x) nogil
cpdef double masked_interpolate_3d_py(np.int16_t[:, :, :] a, \
        np.float64_t[:] x) nogil
cdef double mirrored_interpolate_3d(np.int16_t[:, :, :] a, double *x) nogil
cpdef double mirrored_interpolate_3d_py(np.int16_t[:, :, :] a, \
        np.float64_t[:] x) nogil

cpdef void downsample_XY(np.int16_t[:,:,:] in_im,\
                         np.float64_t[:, :, :] out_im,\
                         np.float64_t[:] scale_factor,\
                         np.float64_t[:] offset,\
                         int thread_num=*) nogil
cpdef void downsample_Z(np.float64_t[:,:,:] in_im,\
                        np.int16_t[:, :, :] out_im,\
                         np.float64_t[:] scale_factor,\
                        np.float64_t[:] offset,\
                        int thread_num=*,
                        bint is_mask=*) nogil
cpdef void downsample(np.int16_t[:,:,:] in_im,\
                      np.int16_t[:, :, :] out_im,\
                      int thread_num=*,
                      bint is_mask=*)
cpdef void downsample_with_offset(np.int16_t[:,:,:] in_im,\
                      np.int16_t[:, :, :] out_im,\
                      np.float64_t[:, :, :] tmp_im,\
                      np.float64_t[:] scale_factor,\
                      np.float64_t[:] offset,\
                      int thread_num=*,
                      bint is_mask=*)
cpdef void interpolate_block(np.int16_t[:,:,:] fixed_im,
                      np.int16_t[:, :, :] moving_im,
                      np.float64_t[:, :] t_matrix_fixed_to_moving,
                      int thread_num=*,
                      bint is_mask=*)
cpdef void fill_gm_image(np.int16_t[:,:,:] in_block, 
        np.int16_t[:,:,:] mask_block, np.float32_t[:,:,:] gm_block, 
        int thread_num)
cpdef np.float32_t gm_max(np.float32_t[:,:,:] gm_block) 
cpdef void fill_gm_histogram(np.float32_t[:,:,:] gm_block,
        np.uint64_t[:] hist, double gm_max)
cpdef void gm_masking(np.int16_t[:,:,:] in_block, 
        np.int16_t[:,:,:] out_im, np.float32_t[:,:,:] gm_block, 
        int gm_z_offset, double gm_limit, bint is_moving) 
cpdef void do_run_length_encoding(np.int16_t[:,:,:] im, int thread_num) nogil
cpdef void add_inertia_tensor_slice(np.int16_t[:, :] im,
        index_t z, np.float64_t[:, :] voxel_to_world_matrix, 
        np.int64_t[:] N, np.float64_t[:] centre, np.float64_t[:, :] TI)
cpdef void lcc_block( np.int16_t[:, :, :] fixed_im, 
        np.int16_t[:, :, :] moving_im, np.float64_t[:, :] t_mat_vox,
        np.int64_t[:] N_buf, np.float64_t[:] lcc_buf, np.int32_t thread_num,
        bint use_rle=*)
cpdef void lcc_block_with_derivatives( np.int16_t[:, :, :] fixed_im,
        np.int16_t[:, :, :] moving_im, np.float64_t[:, :] fixed_v2w_matrix,
        np.float64_t[:, :] trans_mat_vox, np.float64_t[:,:] trans_mat_gradient,
        np.int32_t n_p,
        np.int64_t[:] N_buf, np.float64_t[:] lcc_buf, 
        np.float64_t[:, :] gradient_buf, np.float64_t[:, :, :] hessian_buf,
        np.int32_t thread_num, bint use_rle=*)
cpdef void remove_rle(np.int16_t[:, :] in_slice, np.int16_t[:, :] out_slice)
