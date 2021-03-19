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
   rigid and affine transformation of image data and locations
"""
from __future__ import print_function, division
import sys
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from lcreg import image3d
from lcreg.lcreg_lib import interpolate_block
from lcreg.lcreg_util import system_memory_available_mb


def euler_from_matrix(m):
    """
        Calculate euler angles from matrix

        Args:
            m (numpy.array (4x4, float64) rotation matrix
        Returns:
            numpy.array: 3 euler angles
        """
    rot = R.from_matrix(m[:3, :3])
    return rot.as_euler("xyz", degrees=True)


def t_matrix_world_from_p(p):
    """
        Create a transformation matrix (world coordinates) from parameters p

        Args:
            p (numpy.array, float64): 6 (rigid) or 12 (affine) parameters
                rigid format: t_x, t_y, t_z, r_x, r_y, r_z
        Returns:
            numpy.array (4x4, float64): transformation matrix (voxel coords.)

    """
    m_world = np.identity(4, dtype=np.float64)
    if len(p) == 6:
        rot = R.from_rotvec(p[3:])
        m_world[:3, :3] = rot.as_matrix()
        m_world[:3, 3] = p[:3]
    else:
        m_world[:3, :4] = p.reshape((3, 4))
    return m_world


def t_matrix_vox_from_p(fixed_im, moving_im, p):

    """
        Create a transformation matrix (voxel coordinates) from parameters p

        Args:
            fixed_im (Image3D): fixed image
            moving_im (Image3D): moving image
            p (numpy.array, float64): 6 (rigid) or 12 (affine) parameters
        Returns:
            numpy.array (4x4, float64): transformation matrix (world coords.)

    """
    return t_matrix_world_to_vox(fixed_im, moving_im, t_matrix_world_from_p(p))


def p_from_t_matrix_world(M, p=None, is_affine=False):
    """
    Extract parameters from transformation matrix.

    Args:
        M (numpy.array, 4x3, numpy.float64): Transformation matrix
        p (numpy.array, 1x6 or 1x12, np.float64): parameter vector
            to fill in or None to create new vector
        is_affine (bool): indicates if transform ist affine or rigid
    Returns:
        numpy.array (1x6 or 1x12, numpy.float64): parameter vector

    """
    n_par = 12 if is_affine else 6
    if p is None:
        p = np.zeros(n_par, dtype=np.float64)
    else:
        if p.shape[0] != n_par:
            raise IndexError("wrong number of parameters for transform type")
    if is_affine:
        p[:] = M[:3, :].flatten()
    else:
        p[:3] = M[:3, 3]
        p[3:] = r_from_matrix(M)
    return p


def r_from_matrix(m):
    """
        calculate rotation vector from 3x3 matrix

        Args:
            m (numpy.array, float64): 3x3 rotation matrix
        Returns:
            np.array (1x3): rotation parameters (r_x, r_y, r_z)

    """

    rot = R.from_matrix(m[:3, :3])
    return rot.as_rotvec()


def swap_RAS_LPS(mat):
    """
    Switch beteween RAS and LPS system transformation

    Args:
        mat (np.array): transformation matrix in RAS/LPS system

    Returns:
        np.array: transformation matrix in LPS/RAS system

    """
    flip_xy = np.identity(4).astype(np.float64)
    flip_xy[0][0] = -1
    flip_xy[1][1] = -1
    return np.dot(flip_xy, np.dot(mat, flip_xy))


def t_matrix_world_to_vox(fixed_im, moving_im, m_world):
    """
    Transform transformation matrix from world to voxel coordinates.

    Args:
        fixed_im (image3d): fixed image (will be overwritten)
        moving_im (image3d): moving image (not modified)
        m_world (np.array, 4x4): transformation matrix fixed -> moving
                                 in world coordinates

    Returns:
        np.array 4x4: transformation matrix fixed -> moving in
                      voxel coordinates
    """
    return np.dot(
        moving_im.world_to_voxel_matrix,
        np.dot(m_world, fixed_im.voxel_to_world_matrix),
    ).astype(np.float64)


def transformed_offset_shape(fixed_im, moving_im, edges, m_world):
    """
    Calculate offset and shape of an image block

    Args:
        fixed_im (image3D): fixed image
        moving_im (image3D): moving image
        edges: (np.array, 3x8): edges of image block in voxel coordinates
        m_world(np.array, 4x4): transformation matrix in world coordinates

    Returns:
        tuple of tuples: offset and shape of the tranformed image region
                         order: xyz
    """
    m_vox = t_matrix_world_to_vox(fixed_im, moving_im, m_world)
    last_row = np.ones((1, edges.shape[1]))
    edges = np.vstack((edges, last_row))
    t_edges = np.dot(m_vox, edges)
    shape_array = moving_im.shape_array_xyz.astype(np.float64)
    t_offset = np.zeros(3, dtype=np.int32)
    t_shape = np.zeros(3, dtype=np.int32)
    for i in range(3):
        t_row = t_edges[i, :]
        t_min = int(min(t_row) + 0.5)
        if t_min <= 1:
            t_offset[i] = 0
        elif t_min > shape_array[i]:
            break
        else:
            t_offset[i] = t_min - 1
        t_max = int(max(t_row) + 0.5)
        if t_max > shape_array[i] - 2:
            t_shape[i] = shape_array[i] - t_offset[i]
        elif t_max < 0:
            break
        else:
            t_shape[i] = t_max - t_offset[i] + 2

    return (t_offset, t_shape)


def required_transform_mem_mb(s_in, fixed_im, moving_im, m_world):
    """
    Calculate required memory for image block transformation

    Args:
        s_in (tuple): size of image region in voxels, order xyz
        fixed_im (image3D): fixed image
        moving_im (image3D): moving image
        m_world(np.array, 4x4): transformation matrix in world coordinates

    Returns:
        int: Total memory required for fixed and moving image blocks in MB
    """
    n_voxels_in = s_in[0] * s_in[1] * s_in[2]
    nx, ny, nz = s_in
    edges_in = np.array(
        [
            [0, 0, 0],
            [nx, 0, 0],
            [0, ny, 0],
            [0, 0, nz],
            [0, ny, nz],
            [nx, 0, nz],
            [nx, ny, 0],
            [nx, ny, nz],
        ],
        dtype=np.float64,
    ).T
    # homogeneous coordinates
    last_row = np.ones((1, edges_in.shape[1]))
    edges_in = np.vstack((edges_in, last_row))
    m_vox = t_matrix_world_to_vox(fixed_im, moving_im, m_world)
    # transformed block edge coordinates
    edges_out = np.dot(m_vox, edges_in)[:3]
    out_min = np.min(edges_out, axis=1).astype(np.uint64) - 1
    out_max = np.max(edges_out, axis=1).astype(np.uint64) + 1
    n_voxels_out = int((out_max - out_min).prod() + 0.5)
    return (n_voxels_in + n_voxels_out) // 2 ** 18


def blockwise_grey_transform(
    in_img_name,
    out_img_name,
    m_world,
    mem_limit_mb=-1,
    n_threads=-1,
    is_mask=False,
):
    """
    Perform block wise geometric transformation of an image

    The transformation matrix m_world given in the world coordinate
    system is applied to in_img. The resampled image is stored in
    the bcolz directory given by out_img_name.

    Args:
        in_img_name (str): input bcolz base name
        out_img_name (str): output bcolz base name
        m_world (np.array): transformation matrix to be applied to in_img
        mem_limit_mb (int): memory limit for resampling in MB
        n_threads (int): number of threads to be used (-1: use all cores)
        is_mask (bool): True if image represents a binary mask

    """
    if mem_limit_mb < 0:
        mem_limit_mb = system_memory_available_mb()

    in_img = image3d.open_image(in_img_name, mode="r")
    out_img = image3d.empty_image_like(in_img, out_img_name)

    if 2 * in_img.mem_mb <= mem_limit_mb:
        in_data = in_img.data[:, :, :]
        out_data = out_img.data[:, :, :]
        t_matrix_vox = t_matrix_world_to_vox(out_img, in_img, m_world)
        interpolate_block(out_data, in_data, t_matrix_vox, n_threads, is_mask)
        out_img.data[:, :, :] = out_data[:, :, :]
        del in_data
        del out_data
    else:
        block_size = np.array((100, 100, 100), dtype=np.int32)
        mem_100 = required_transform_mem_mb(
            block_size, out_img, in_img, m_world
        )
        factor = math.pow(mem_limit_mb / mem_100, 1 / 3)
        block_size = (factor * block_size).astype(np.int32)
        for z in range(0, out_img.shape[0], block_size[2]):
            s_z = min(out_img.shape[0] - z, block_size[2])
            for y in range(0, out_img.shape[1], block_size[1]):
                s_y = min(out_img.shape[1] - y, block_size[1])
                for x in range(0, out_img.shape[2], block_size[0]):
                    s_x = min(out_img.shape[2] - x, block_size[0])
                    if min(s_x, s_y, s_z) > 0:
                        edges = np.array(
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
                        in_offset, in_shape = transformed_offset_shape(
                            out_img, in_img, edges, m_world
                        )
                        if min(in_shape) > 0:
                            out_part = image3d.image_part(
                                out_img,
                                np.array((x, y, z)),
                                np.array((s_x, s_y, s_z)),
                            )
                            in_part = image3d.image_part(
                                in_img, in_offset, in_shape
                            )
                            t_matrix_vox = t_matrix_world_to_vox(
                                out_part, in_part, m_world
                            )
                            interpolate_block(
                                out_part.data,
                                in_part.data,
                                t_matrix_vox,
                                n_threads,
                                is_mask,
                            )
                            del in_part
                            out_img.data[
                                z : z + s_z, y : y + s_y, x : x + s_x
                            ] = out_part.data[:, :, :]
                            del out_part
    image3d.close_image(in_img)
    image3d.close_image(out_img)
