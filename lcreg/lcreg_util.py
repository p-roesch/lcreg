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
    lcreg utility functions
"""
from __future__ import print_function, division
import ast
import psutil
import numpy as np


def array_to_string(arr):
    """
    Convert numpy array into 1D string.

    Args:
        arr (numpy array): numpy array
    Returns:
        str containing space separated numbers corresponding to
        the flattened array
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    string = str(arr.flatten())[1:-1].strip()
    return " ".join(string.split())


def string_to_array(string, dtype=np.float64):
    """
    Convert string in to numpy array.

    Args:
        string (str): String containing space-separated numbers
        dtype (numpy dtype): dtype of numpy array to be created
    Returns:
        numpy array from string with requested dtype

    """
    string = " ".join(string.split())
    arr_str = "[" + string.replace(" ", ",") + "]"
    return np.array(ast.literal_eval(arr_str), dtype=dtype)


def system_nr_of_cores():
    """
    Get number of CPU cores
    """
    return psutil.cpu_count()


def system_memory_available_mb():
    """
    Get 50% of available memory in MB
    """
    mem_limit_mb = int(psutil.virtual_memory().available // 2 ** 21)
    return mem_limit_mb


def corners_vox_x_y_z(im):
    """
    Generate corner positions of an image in voxel coordinates

    Args:
        im (Image3D or numpy array): image
    """
    for z in (0, im.shape[0] - 1):
        for y in (0, im.shape[1] - 1):
            for x in (0, im.shape[2] - 1):
                yield (x, y, z)


def image_corners_vox(im):
    """
    Create matrix containing image edges in voxel coordinates

    Args:
        im (Image3d): image
    """
    corner_list = []
    for x, y, z in corners_vox_x_y_z(im):
        corner_list.append([x, y, z, 1])
    corners_vox = np.array(corner_list, dtype=np.float64).T
    return corners_vox


def image_corners_world(im):
    """
    Create matrix containing image edges in voxel coordinates

    Args:
        im (Image3d): image
    """
    corners_world = np.dot(im.voxel_to_world_matrix, image_corners_vox(im))
    return corners_world
