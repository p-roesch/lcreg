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
    Creation of masked images. Masking is based on gradient magnitude and
    optionally on binary images provided by the user,
"""

import logging
import ast
import multiprocessing
import os.path
import numpy as np
import bcolz

from lcreg import image3d
from lcreg import lcreg_lib as cy_ip
from lcreg import lcreg_util


def _fill_gm_buf(in_data, mask_data, gm_buf, block_size, thread_num=-1):
    gm_max = -1.0
    for z in range(0, in_data.shape[0] - 1, block_size):
        if z == 0:
            # keep first gm slice empty
            gm_offset_low = 1
        else:
            # use first gm slice
            gm_offset_low = 0
        if z + block_size >= in_data.shape[0]:
            # calculate number of remaining gm rows
            # last row is left empty
            gm_offset_high = in_data.shape[0] - z - 2
        else:
            # use block_size rows
            gm_offset_high = block_size

        in_block = in_data[
            z + gm_offset_low - 1 : z + gm_offset_high + 1, :, :
        ]
        gm_block = gm_buf[z + gm_offset_low : z + gm_offset_high, :, :]
        if mask_data is None:
            mask_block = None
        else:
            mask_block = mask_data[
                z + gm_offset_low : z + gm_offset_high, :, :
            ]
        cy_ip.fill_gm_image(in_block, mask_block, gm_block, thread_num)
        gm_buf[z + gm_offset_low : z + gm_offset_high, :, :] = gm_block[
            :, :, :
        ]
        gm_max = max(gm_max, cy_ip.gm_max(gm_block))
    gm_buf.flush()
    return gm_max


def _create_gm_histogram(gm_buf, gm_max, block_size):
    gm_hist = np.zeros(shape=(2 ** 20,), dtype=np.uint64)
    for z in range(1, gm_buf.shape[0] - 1, block_size):
        gm_block = gm_buf[z : min(z + block_size, gm_buf.shape[0] - 1), :, :]
        cy_ip.fill_gm_histogram(gm_block, gm_hist, gm_max)
    return gm_hist


def _fill_masked_image(
    in_data, out_data, gm_buf, gm_limit, is_moving, block_size, thread_num=-1
):
    for z in range(1, gm_buf.shape[0] - 1, block_size):
        gm_block = gm_buf[z : min(z + block_size, gm_buf.shape[0] - 1), :, :]
        z_low_img = max(0, z - 2)
        z_high_img = min(z + block_size + 2, in_data.shape[0])
        in_block = in_data[z_low_img:z_high_img, :, :]
        out_block = out_data[z_low_img:z_high_img, :, :]
        out_data[z_low_img:z_high_img, :, :] = out_block[:, :, :]
        z_offset = z - z_low_img
        cy_ip.gm_masking(
            in_block, out_block, gm_block, z_offset, gm_limit, is_moving
        )
        out_data[z_low_img:z_high_img, :, :] = out_block[:, :, :]
    if not is_moving:
        for z in range(1, out_data.shape[0] - 1, block_size):
            z_max = min(z + block_size, out_data.shape[0] - 1)
            out_block = out_data[z:z_max, :, :]
            cy_ip.do_run_length_encoding(out_block, thread_num)
    out_data.flush()


def _mask_image(
    in_im_name, mask_name, gm_surface, mem_limit_mb, thread_num, out_im_name
):
    """
        Create mask image in internal format

        Args:
            in_im_name (str): Name of input image rootdir
            in_mask_name (str): Name of input mask rootdir
            gm_surface (float): desired gm surface in mm^2
            mem_limit_mb (int): Amount of memory to use for masking
            thread_num (int): Nr. of threads
            out_im_name (int): Name of output grey image
            remove_temporary_images (bool): remove temporary images if True
    """
    # initialise thread_num and mem limit
    if thread_num <= 0:
        thread_num = lcreg_util.system_nr_of_cores()
    if mem_limit_mb <= 0:
        mem_limit_mb = lcreg_util.system_memory_available_mb()
    # open/create images and initialise data reference
    # input image
    in_im = image3d.open_image(in_im_name, mode="r")
    in_data = in_im.data
    # calculate gm_fraction
    gm_fraction = gm_surface / in_im.total_voxel_surface
    # one thread per image line
    thread_num = min(thread_num, in_im.shape[1])
    # output image
    out_im = image3d.empty_image_like(in_im, out_im_name)
    out_data = out_im.data
    # mask image
    if mask_name != "None":
        mask_im = image3d.open_image(mask_name, mode="r")
        mask_data = mask_im.data
    else:
        mask_im = None
        mask_data = None
    #
    # determine block size
    bytes_per_voxel = 2 + 4 + 4
    if mask_im is not None:
        bytes_per_voxel += 2
    bytes_per_slice = (
        bytes_per_voxel * int(in_im.shape[1]) * int(in_im.shape[2])
    )
    block_size = min(in_im.shape[0], mem_limit_mb * 2 ** 20 // bytes_per_slice)
    #
    # allocate float buffer for gm values
    buf_slice = np.zeros(
        shape=(1, in_im.shape[1], in_im.shape[2]), dtype=np.float32
    )
    gm_buf_size_mb = in_im.mem_mb * 2
    if gm_buf_size_mb < image3d.BCOLZ_COMPRESSION_LIMIT_MB:
        c_params = bcolz.cparams(clevel=1, shuffle=bcolz.NOSHUFFLE)
    else:
        c_params = None
    gm_buf_name = in_im_name + "_gm_buf"
    gm_buf = bcolz.carray(
        buf_slice,
        rootdir=gm_buf_name,
        cparams=c_params,
        expectedlen=in_im.nr_of_voxels,
        mode="w",
    )
    for _ in range(in_im.shape[0] - 1):
        gm_buf.append(buf_slice)
    gm_buf.flush()
    #
    # fill float image and calculate maximum gm value
    gm_max = _fill_gm_buf(in_data, mask_data, gm_buf, block_size, thread_num)
    #
    # reopen gm_buf in read only mode
    gm_buf.flush()
    del gm_buf
    gm_buf = bcolz.open(gm_buf_name, mode="r")
    #
    # close mask image
    if mask_im is not None:
        image3d.close_image(mask_im)
    #
    # create gm_histogram
    hist = _create_gm_histogram(gm_buf, gm_max, block_size)
    #
    # calculate gm_limit
    n_elem = np.sum(hist, dtype=np.uint64)
    count_limit = int(n_elem * (1 - gm_fraction) + 0.5)
    cum_hist = np.cumsum(hist)
    bin_index = np.argmin(np.abs(cum_hist - count_limit))
    gm_limit = gm_max * bin_index / hist.shape[0]
    #
    #
    # fill in the output image
    is_moving = "moving" in out_im_name
    _fill_masked_image(
        in_data, out_data, gm_buf, gm_limit, is_moving, block_size, thread_num
    )
    logging.debug(
        os.path.basename(out_im_name)
        + " compression ratio: "
        + str(out_data.nbytes / out_data.cbytes)
    )
    # cleanup
    # delete gm buffer
    gm_buf.purge()
    # close images
    image3d.close_image(in_im)
    image3d.close_image(out_im)


def mask_level(reg_config, level):
    """
        Create masked images for a certain level.

        Args:
            reg_config (ConfigParser): Registration configuration
            level (int): level for which masked image is to be created

        Returns:
            tuple(str, str): Names of created fixed and moving masked image
    """
    mem_limit_mb = reg_config.getint("DEFAULT", "mem_limit_mb")
    # memory available
    if mem_limit_mb < 0:
        mem_limit_mb = lcreg_util.system_memory_available_mb()
    # number of threads to use
    thread_num = reg_config.getint("DEFAULT", "nr_of_threads")
    if thread_num < 0:
        thread_num = multiprocessing.cpu_count()
    #
    # mask fixed image
    in_im_name = ast.literal_eval(reg_config["pyramid"]["fixed_grey_files"])[
        level
    ]
    if reg_config["DEFAULT"]["fixed_mask_image_name"] != "None":
        mask_name = ast.literal_eval(
            reg_config["pyramid"]["fixed_mask_files"]
        )[level]
    else:
        mask_name = "None"
    gm_surface = reg_config.getfloat("masking", "fixed_gm_surface")
    fixed_out_im_name = in_im_name + "_masked"
    _mask_image(
        in_im_name,
        mask_name,
        gm_surface,
        mem_limit_mb,
        thread_num,
        fixed_out_im_name,
    )
    #
    # mask moving image
    in_im_name = ast.literal_eval(reg_config["pyramid"]["moving_grey_files"])[
        level
    ]
    if reg_config["DEFAULT"]["moving_mask_image_name"] != "None":
        mask_name = ast.literal_eval(
            reg_config["pyramid"]["moving_mask_files"]
        )[level]
    else:
        mask_name = "None"
    gm_surface = reg_config.getfloat("masking", "moving_gm_surface")
    moving_out_im_name = in_im_name + "_masked"
    _mask_image(
        in_im_name,
        mask_name,
        gm_surface,
        mem_limit_mb,
        thread_num,
        moving_out_im_name,
    )
    #
    # return names of created images
    return (fixed_out_im_name, moving_out_im_name)
