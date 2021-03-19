#!/usr/bin/env python3

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
Convert Scanco ISQ file into mhd format without copying grey value data
"""

import os
import sys
from collections import OrderedDict
import numpy as np

# positions of relevant parameters in ISQ header. Numbers correspond
# to indices in a np.int32 array.
# Source: http://www.scanco.ch/en/support/customer-login/\
#                           faq-customers/faq-customers-general.html
__ISQ_OFFSETS_INT_4 = {
    "dimx_p": 11,
    "dimy_p": 12,
    "dimz_p": 13,
    "dimx_um": 14,
    "dimy_um": 15,
    "dimz_um": 16,
    "slice_thickness": 17,
    "slice_increment_um": 18,
    "slice_1_pos_um": 19,
    "min_data_value": 20,
    "max_data_value": 21,
    "mu_scaling": 22,
    "energy": 42,
    "intensity": 43,
    "data_offset": -1,
}

# Default values for mhd file parameters
__MHD_DEFAULTS = OrderedDict(
    [
        ("ObjectType", "Image"),
        ("NDims", "3"),
        ("BinaryData", "True"),
        ("BinaryDataByteOrderMSB", "False"),
        ("CompressedData", "False"),
        ("TransformMatrix", "1 0 0 0 1 0 0 0 1"),
        ("Offset", "0 0 0"),
        ("CenterOfRotation", "0 0 0"),
        ("AnatomicalOrientation", "RAI"),
        ("ElementSpacing", "1 1 1"),
        ("DimSize", "1 1 1"),
        ("HeaderSize", "-1"),
        ("ElementType", "MET_SHORT"),
    ]
)


def _read_isq_param(in_file_name):
    """
    Read parameters from ISQ file into OrderedDict

    Args:
        in_file_name (str): Input file name including suffix
    Returns:
        tuple of
            OrderedDict containing parameters in mhd style,
            offset of ISQ file data part in bytes
            grey value range (tuple)
    """
    param = __MHD_DEFAULTS.copy()
    isq_header = np.fromfile(in_file_name, np.int32, 128)
    # swap bytes if required
    if sys.byteorder == "big":
        isq_header = isq_header.byteswap()
    dim_p_str = ""
    element_spacing_str = ""
    # load image dimensions and calculate element spacing
    for i in range(3):
        dim_p = isq_header[__ISQ_OFFSETS_INT_4["dimx_p"] + i]
        dim_um = isq_header[__ISQ_OFFSETS_INT_4["dimx_um"] + i]
        dim_p_str += str(dim_p) + " "
        element_spacing_mm = dim_um / dim_p / 1000.0
        element_spacing_str += str(element_spacing_mm) + " "
    # get grey value range for windowing
    grey_min = isq_header[__ISQ_OFFSETS_INT_4["min_data_value"]]
    grey_max = isq_header[__ISQ_OFFSETS_INT_4["max_data_value"]]
    # set mhd parameters
    param["DimSize"] = dim_p_str[:-1]
    param["ElementSpacing"] = element_spacing_str[:-1]
    param["ElementType"] = "MET_SHORT"
    param["HeaderSize"] = "-1"
    param["ISQ_slice_thickness_um"] = str(
        isq_header[__ISQ_OFFSETS_INT_4["slice_thickness"]]
    )
    param["ISQ_slice_increment_um"] = str(
        isq_header[__ISQ_OFFSETS_INT_4["slice_increment_um"]]
    )
    param["ISQ_slice_1_pos_um"] = str(
        isq_header[__ISQ_OFFSETS_INT_4["slice_1_pos_um"]]
    )
    param["ISQ_min_data_value"] = str(grey_min)
    param["ISQ_max_data_value"] = str(grey_max)
    param["ISQ_mu_scaling"] = str(
        isq_header[__ISQ_OFFSETS_INT_4["mu_scaling"]]
    )
    param["ISQ_energy_V"] = str(isq_header[__ISQ_OFFSETS_INT_4["energy"]])
    param["ISQ_intensity_muA"] = str(
        isq_header[__ISQ_OFFSETS_INT_4["intensity"]]
    )
    param["ElementDataFile"] = os.path.basename(in_file_name)
    # calculate offset for grey value loading
    offset = (isq_header[__ISQ_OFFSETS_INT_4["data_offset"]] + 1) * 512
    grey_range = (grey_min, grey_max)
    return param, offset, grey_range


def isq_to_mhd(isq_file_name, mhd_file_name):
    """
        Convert ISQ file into meta image file
        ARGS:
            isq_file_name (str): full path name of isq image file
            mhd_file_name (str): name of mhd file to be written
    """
    mhd_param, offset, grey_range = _read_isq_param(isq_file_name)
    if os.sep in mhd_file_name:
        mhd_param["ElementDataFile"] = os.path.abspath(isq_file_name)
    else:
        mhd_param["ElementDataFile"] = isq_file_name
    with open(mhd_file_name, "w") as out_file:
        for i in mhd_param.items():
            out_file.write(i[0] + " = " + i[1] + "\n")


def main():
    if len(sys.argv) != 3:
        print("usage: isq_to_mhd isq_file_name mhd_file_name")
        sys.exit(0)
    else:
        isq_to_mhd(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
