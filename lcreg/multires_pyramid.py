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
    Multiresolution representation of 3d images.
"""

from __future__ import print_function, division
import logging
import math
import os.path
import numpy as np

from lcreg import image3d
from lcreg import lcreg_lib as cy_ip

from lcreg.lcreg_util import system_memory_available_mb

_SQRT2 = math.sqrt(2)
_SCALE_SQRT2 = np.array([_SQRT2, _SQRT2, _SQRT2], dtype=np.float64)
_SCALE_2 = np.array([2, 2, 2], np.float64)


def _resampling_mem_needed_mb(out_shape, scale_factor):
    n_out = np.array(out_shape, dtype=np.uint64)[::-1]
    n_in = np.ceil((n_out + 1) * scale_factor).astype(np.uint64)
    mem_needed = (
        2 * (n_in.prod() + n_out.prod()) + 4 * n_out[0] * n_out[1] * n_in[2]
    )
    return mem_needed // 2 ** 20


def _blockwise_resample(
    in_im_name,
    out_im_name,
    scale_factor,
    is_mask=False,
    mem_limit_mb=-1,
    thread_num=-1,
):
    """
        Resample an image to another resolution.

        This function resamples an image block by block to another
        resolution using not more than the specified amount of RAM.

        Args:
            in_im_name (str): name of input image
            out_im_name (str): name of output image
            scale_factor (numpy array): 3d scaling factor for
                voxel size. Thus, a scaling factor > 1 corresponds
                to downsampling. Note: order is (x, y, z)
            is_mask (bool): flag switching between grey value and
                mask mode
            mem_limit_mb (int): upper memory limit for internal
                buffers in mb. A value of -1 indicates that all
                available RAM should be used.
            thread_num (int): number of threads to be used for
                the calculation. A value of -1 indicates that all
                available cpu cores should be used
    """
    # open input image
    in_im = image3d.open_image(in_im_name, mode="r")
    # calculate output image size
    out_im_shape_array_xyz = np.round(
        in_im.shape_array_xyz / scale_factor
    ).astype(np.uint32)
    out_im_shape = tuple(out_im_shape_array_xyz[::-1])
    # correct scaling factor w.r.t rounding
    scale_factor = in_im.shape_array_xyz / out_im_shape_array_xyz
    scale_factor = scale_factor.astype(np.float64)
    # spacing of output image
    out_im_spacing = in_im.spacing * scale_factor
    # logging output
    logging.info("  spacing / mm: " + str(list(out_im_spacing)))
    logging.info("  size: " + str(out_im_shape))
    # calculate orign of output image
    out_im_offset_vox = 0.5 * scale_factor - 0.5
    out_im_offset = in_im.transform_voxel_to_world(out_im_offset_vox)

    out_im = image3d.create_image(
        out_im_name,
        out_im_shape,
        out_im_offset,
        out_im_spacing,
        in_im.transform_matrix,
        mode="w",
    )
    out_im.data.flush()

    # get amount of memory to be used
    if mem_limit_mb < 0:
        mem_limit_mb = system_memory_available_mb()

    # calculated memory required for downsampling
    mem_needed_mb = _resampling_mem_needed_mb(out_im.shape, scale_factor)

    # perform resampling
    if mem_needed_mb < mem_limit_mb:
        in_array = in_im.data[:, :, :]
        out_array = out_im.data[:, :, :]
        cy_ip.downsample(in_array, out_array, thread_num, is_mask)
        out_im.data[:, :, :] = out_array[:, :, :]
    else:
        batch_size_z = 1
        while (
            _resampling_mem_needed_mb(
                (batch_size_z, out_im.shape[1], out_im.shape[2]), scale_factor
            )
            < mem_limit_mb
        ):
            batch_size_z += 1
        batch_size_z -= 1
        if batch_size_z == 0:
            raise MemoryError("Insufficient Memory for downsampling")
        # allocate temporary array
        tmp_shape_z = int(np.ceil((batch_size_z + 2) * scale_factor[2]))
        tmp_shape = (tmp_shape_z, out_im.shape[1], out_im.shape[2])
        tmp_array = np.zeros(shape=tmp_shape, dtype=np.float64)
        for z_pos in range(0, out_im.shape[0], batch_size_z):
            tmp_array.fill(0.0)
            z_end = min(z_pos + batch_size_z, out_im.shape[0])
            out_block = out_im.data[z_pos:z_end, :, :]
            in_start = max(int(np.floor(z_pos * scale_factor[2] - 1)), 0)
            in_end = min(
                int(np.ceil((z_end + 1) * scale_factor[2] - 1)), in_im.shape[0]
            )
            in_block = in_im.data[in_start:in_end, :, :]
            in_offset_z = z_pos * scale_factor[2] - in_start
            in_offset = np.array([0, 0, in_offset_z], dtype=np.float64)
            cy_ip.downsample_with_offset(
                in_block,
                out_block,
                tmp_array[0 : in_end - in_start, :, :],
                scale_factor,
                in_offset,
                thread_num,
                is_mask,
            )
            out_im.data[z_pos:z_end, :, :] = out_block[:, :, :]
        del in_block
        del out_block
        del tmp_array

    # close images
    image3d.close_image(in_im)
    image3d.close_image(out_im)


def _create_image_pyramid(
    base_img_name,
    highest_resolution,
    is_sqrt2_pyramid,
    nr_of_levels,
    is_mask,
    mem_limit_mb,
    thread_num,
):
    """
        Args:
            base_img_name: name of image the pyramid is based on
            nr_of_levels (int): number of levels to be generated
            highest_resolution (np.array): highest voxel spacing
                in the pyramid (mm)
            is_mask (bool): flag switching between grey value and
                mask mode
            mem_limit_mb (int): upper memory limit for internal
                buffers in mb. A value of -1 indicates that all
                available RAM should be used.
            thread_num (int): number of threads to be used for
                the calculation. A value of -1 indicates that all
                available cpu cores should be used
        Returns:
            list: image file names starting with highest resolutions
    """
    file_name_list = []
    base_img = image3d.open_image(base_img_name)
    base_spacing = base_img.spacing
    image3d.close_image(base_img)

    # resample base image if required
    if np.allclose(base_img.spacing, highest_resolution):
        file_name_list.append(base_img_name)
    else:
        scale_factor = highest_resolution / base_spacing
        out_img_name = base_img_name + "_l_00"
        logging.info(os.path.basename(out_img_name) + " :")
        _blockwise_resample(
            base_img_name,
            out_img_name,
            scale_factor,
            is_mask,
            mem_limit_mb,
            thread_num,
        )
        file_name_list.append(out_img_name)

    # perform downsampling
    for i in range(1, nr_of_levels + 1):
        if is_sqrt2_pyramid:
            if i == 1:
                scale_factor = _SCALE_SQRT2
                in_img_name = file_name_list[0]
            else:
                scale_factor = _SCALE_2
                in_img_name = file_name_list[i - 2]
        else:
            scale_factor = _SCALE_2
            in_img_name = file_name_list[i - 1]

        out_img_name = "%s_l_%02i" % (base_img_name, i)
        logging.info(os.path.basename(out_img_name) + " :")
        _blockwise_resample(
            in_img_name,
            out_img_name,
            scale_factor,
            is_mask,
            mem_limit_mb,
            thread_num,
        )
        file_name_list.append(out_img_name)
    return file_name_list


def create_pyramid(reg_config):
    """
       Create multi resolution pyramid according to configuration file.

       Args:
        reg_config (configparser.ConfigParser): regpy3d configuration
    """
    logging.info("Started pyramid generation")
    base_spacing = None
    # determine finest resolution
    for n in [
        "fixed_grey_base",
        "moving_grey_base",
        "fixed_mask_base",
        "moving_mask_base",
    ]:
        if reg_config.has_option("pyramid", n):
            img = image3d.open_image(reg_config["pyramid"][n])
            if base_spacing is None:
                base_spacing = img.spacing
            else:
                base_spacing = np.max([base_spacing, img.spacing], axis=0)
            image3d.close_image(img)
    if reg_config.getboolean("multiresolution", "pyramid_cubic_voxels"):
        base_spacing.fill(np.max(base_spacing))
    base_spacing *= reg_config.getfloat(
        "multiresolution", "pyramid_min_scale_factor"
    )
    logging.debug("base_spacing: {} mm".format(base_spacing))
    pyramid_sqrt2_scaling = reg_config.getboolean(
        "multiresolution", "pyramid_sqrt2_scaling"
    )
    # get scale factor
    if pyramid_sqrt2_scaling:
        level_scale_factor = _SQRT2
    else:
        level_scale_factor = float(2)
    # determine number of levels
    level_nr = None
    min_vox_nr = eval(reg_config["multiresolution"]["pyramid_min_voxel_nr"])
    for n in [
        "fixed_grey_base",
        "moving_grey_base",
        "fixed_mask_base",
        "moving_mask_base",
    ]:
        if reg_config.has_option("pyramid", n):
            img = image3d.open_image(reg_config["pyramid"][n])
            factor = np.prod(img.spacing) / np.prod(base_spacing)
            nr_vox_at_base_level = img.nr_of_voxels * factor
            current_level_nr = int(
                math.log(nr_vox_at_base_level / min_vox_nr)
                / math.log(level_scale_factor)
                / 3.0
                + 0.5
            )
            # need at least 7 as mininum nr of voxels in each direction
            dim_level_nr = int(
                math.log(min(img.shape) / 7) / math.log(level_scale_factor)
            )
            if level_nr is None:
                level_nr = min(current_level_nr, dim_level_nr)
            else:
                level_nr = min(level_nr, current_level_nr, dim_level_nr)
            image3d.close_image(img)
    logging.debug("nr of levels: {}".format(level_nr))
    # create levels
    mem_limit_mb = reg_config.getint("DEFAULT", "mem_limit_mb")
    nr_of_threads = reg_config.getint("DEFAULT", "nr_of_threads")
    for n in [
        "fixed_grey_base",
        "moving_grey_base",
        "fixed_mask_base",
        "moving_mask_base",
    ]:
        if reg_config.has_option("pyramid", n):
            im_name = reg_config["pyramid"][n]
            file_name_list = _create_image_pyramid(
                im_name,
                base_spacing,
                pyramid_sqrt2_scaling,
                level_nr,
                "mask" in n,
                mem_limit_mb,
                nr_of_threads,
            )
            list_name = n.replace("base", "files")
            reg_config.set("pyramid", list_name, str(file_name_list))
    logging.info("Pyramid generation done")
