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

"""
    Similarity measure calculation (Local Correlation Coefficient)
"""
from __future__ import print_function, division
import math
import numpy as np

from lcreg import image3d
from lcreg.lcreg_lib import lcc_block, lcc_block_with_derivatives
from lcreg.lcreg_util import system_memory_available_mb
from lcreg.image_transform import required_transform_mem_mb
from lcreg.image_transform import transformed_offset_shape
from lcreg.image_transform import t_matrix_world_from_p, t_matrix_vox_from_p


def __gradient_matrix(img):
    """
        Create Matrix to transform gradients from voxel to world coordinates

        Args:
            img (Image3D): 3D image

        Returns:
            numpy.array (3x3, numpy.float64): gradient transformation matrix
    """
    return np.dot(img.transform_matrix, np.diag(0.5 / img.spacing))


def __create_N_buf(fixed_im):
    """
        Create empty accumulation buffer for voxel pair counter (N)

        Args:
            fixed_im (Image3D): fixed image

        Returns:
            numpy.array (1xfixed_im.shape[0], numpy.int64): accumulation buffer
    """
    return np.zeros(shape=(fixed_im.shape[0],), dtype=np.int64)


def __create_lcc_buf(fixed_im):
    """
        Create empty accumulation buffer for local correlation sums (lcc)

        Args:
            fixed_im (Image3D): fixed image

        Returns:
            numpy.array (1xfixed_im.shape[0], numpy.float64): Buffer
    """
    return np.zeros(shape=(fixed_im.shape[0],), dtype=np.float64)


def __create_gradient_buf(fixed_im, p):
    """
        Create empty accumulation buffer for lcc gradient sums

        Args:
            fixed_im (Image3D): fixed image
            p (numpy.array, numpy.float64): 6 (rigid) or 12 (affine) parameters

        Returns:
            numpy.array (fixed_im.shape[1]xlen(p), numpy.float64): Buffer
    """
    return np.zeros(shape=(fixed_im.shape[0], len(p)), dtype=np.float64)


def __create_hessian_buf(fixed_im, p):
    """
        Create empty accumulation buffer for lcc approximated hessian sums

        Args:
            fixed_im (Image3D): fixed image
            p (numpy.array, numpy.float64): 6 (rigid) or 12 (affine) parameters

        Returns:
            numpy.array (fixed_im.shape[0] x len(p) x len(p), numpy.float64):
                Buffer
    """
    return np.zeros(
        shape=(fixed_im.shape[0], len(p), len(p)), dtype=np.float64
    )


def __calc_lcc(
    fixed_im,
    moving_im,
    p,
    with_derivatives,
    N_sum,
    lcc_sum,
    gradient_sum,
    hessian_sum,
    thread_num=-1,
    use_rle=True,
):
    """
        Calculate the LCC measure with or without derivatives
            (Python interface)

        This function calls the optimised lcc_block functions implemented
        in cython.

        Args:
            fixed_im (Image3D): fixed image
            moving_im (Image3D): moving image
            p (numpy array, numpy.float64): rigid or affine parameter vector
            with_derivatives (bool): Flag, True if gradient and hessian
                have to be calculated.
            N_sum: (numpy array, numpy.int64, 1 entry): accumulation buffer
                for number of voxels used for LCC calculation
            lcc_sum: (numpy array, numpy.float64, 1 entry): accumulation buffer
                for lcc=(1-LCC) value
            gradient_sum: (numpy array, numpy.float64, 1xlen(p)):
                accumulation buffer for lcc gradient
            hessian_sum: (numpy array, numpy.float64, len(p)xlen(p)):
                accumulation buffer for lcc hessian
            thread_num: Number of threads to be used for calculation or
                -1 to use all cores
            use_rle (bool): use or ignore run length encoding

        Returns:
            tuple (N, lcc, gradient, hessian):
                N: Number of voxels used for LCC calculation
                lcc: Value of (1-LCC)
                if with_derivatives is True:
                    gradient: 1D numpy array with LCC derivatives w.r.t. p
                    hessian: 2D numpy array with aprrox. LCC hessian w.r.t. p
                if not enough corresponding masked voxels could be found,
                    (None, None) is returned
    """
    fixed_data = fixed_im.data[:, :, :]
    moving_data = moving_im.data[:, :, :]
    trans_mat_vox = t_matrix_vox_from_p(fixed_im, moving_im, p)
    gradient_matrix = __gradient_matrix(fixed_im)
    N_buf = __create_N_buf(fixed_im)
    lcc_buf = __create_lcc_buf(fixed_im)
    if with_derivatives:
        gradient_buf = __create_gradient_buf(fixed_im, p)
        hessian_buf = __create_hessian_buf(fixed_im, p)
        lcc_block_with_derivatives(
            fixed_data,
            moving_data,
            fixed_im.voxel_to_world_matrix,
            trans_mat_vox,
            gradient_matrix,
            len(p),
            N_buf,
            lcc_buf,
            gradient_buf,
            hessian_buf,
            thread_num,
            use_rle,
        )
    else:
        lcc_block(
            fixed_data,
            moving_data,
            trans_mat_vox,
            N_buf,
            lcc_buf,
            thread_num,
            use_rle,
        )
    N_sum[0] += N_buf.sum()
    lcc_sum[0] += lcc_buf.sum(0)
    del N_buf
    del lcc_buf
    if with_derivatives:
        gradient_sum += gradient_buf.sum(0)
        hessian_sum += hessian_buf.sum(0)
        del gradient_buf
        del hessian_buf


def lcc(
    fixed_im,
    moving_im,
    p,
    with_derivatives=True,
    mem_limit_mb=-1,
    num_threads=-1,
    use_rle=True,
):
    """
    Perform block wise lcc calculation of an image pair.

    Args:
        fixed_im (Image3D): fixed image
        moving_im (Image3D): moving image
        p (numpy float64 array): 6 (rigid) or 12 (affine)
            transformation parameters
        with_derivatives (bool): True to calculate gradient and hessian
        mem_limit_mb (int): memory limit for local buffers in MB
        num_threads (int): number of threads to be used (-1: use all cores)
        use_rle (bool): use run lenght encoding

    Returns:
        tuple with (N, lcc, ) or (N, lcc, gradient, hessian) for
        with_derivatives==False or with_derivatives==True respectively
    """

    if mem_limit_mb < 0:
        mem_limit_mb = system_memory_available_mb() // 2

    if isinstance(fixed_im.data, np.ndarray):
        mem_limit_mb += fixed_im.mem_mb

    if isinstance(moving_im.data, np.ndarray):
        mem_limit_mb += moving_im.mem_mb

    lcc_sum = np.zeros(1, dtype=np.float64)
    N_sum = np.zeros(1, dtype=np.int64)
    if with_derivatives:
        gradient_sum = np.zeros(len(p), dtype=np.float64)
        hessian_sum = np.zeros((len(p), len(p)), dtype=np.float64)
    else:
        gradient_sum = None
        hessian_sum = None
    if fixed_im.mem_mb + moving_im.mem_mb <= mem_limit_mb:
        __calc_lcc(
            fixed_im,
            moving_im,
            p,
            with_derivatives,
            N_sum,
            lcc_sum,
            gradient_sum,
            hessian_sum,
            num_threads,
            use_rle,
        )
    else:
        use_rle = False
        block_size = np.array((100, 100, 100), dtype=np.int32)
        mem_100 = required_transform_mem_mb(
            block_size, fixed_im, moving_im, t_matrix_world_from_p(p)
        )
        factor = math.pow(mem_limit_mb / mem_100, 1 / 3)
        block_size = (factor * block_size).astype(np.int32)
        # subtract overlap region for both sides of the image
        overlap_size = 2
        for z in range(0, fixed_im.shape[0], block_size[2] - 2 * overlap_size):
            s_z = min(fixed_im.shape[0] - z, block_size[2])
            for y in range(
                0, fixed_im.shape[1], block_size[1] - 2 * overlap_size
            ):
                s_y = min(fixed_im.shape[1] - y, block_size[1])
                for x in range(
                    0, fixed_im.shape[2], block_size[0] - 2 * overlap_size
                ):
                    s_x = min(fixed_im.shape[2] - x, block_size[0])
                    if min(s_x, s_y, s_z) > 0:
                        fixed_edges = np.array(
                            [
                                [x, y, z],
                                [x + s_x, y, z],
                                [x, y + s_y, z],
                                [x, y, z + s_z],
                                [x + s_x, y + s_y, z],
                                [x + s_x, y, z + s_z],
                                [x, y + s_y, z + s_z],
                                [x + s_x, y + s_y, z + s_z],
                            ],
                            dtype=np.float64,
                        ).T
                        moving_offset, moving_shape = transformed_offset_shape(
                            fixed_im,
                            moving_im,
                            fixed_edges,
                            t_matrix_world_from_p(p),
                        )
                        if min(moving_shape) > 0:
                            fixed_part = image3d.image_part(
                                fixed_im,
                                np.array((x, y, z)),
                                np.array((s_x, s_y, s_z)),
                            )
                            moving_part = image3d.image_part(
                                moving_im, moving_offset, moving_shape
                            )
                            __calc_lcc(
                                fixed_part,
                                moving_part,
                                p,
                                with_derivatives,
                                N_sum,
                                lcc_sum,
                                gradient_sum,
                                hessian_sum,
                                num_threads,
                                use_rle,
                            )
                            del fixed_part
                            del moving_part
    N = N_sum[0]
    if N == 0:
        result = (None, None)
    else:
        if with_derivatives:
            # fill hessian (symmetry ...)
            for u in range(len(p)):
                for v in range(u + 1, len(p)):
                    hessian_sum[v, u] = hessian_sum[u, v]
            result = (N, lcc_sum[0] / 2 / N, gradient_sum / N, hessian_sum / N)
        else:
            result = (N, lcc_sum[0] / N)
    return result
