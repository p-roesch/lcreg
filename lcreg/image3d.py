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
Module defining the Image3D class, Image conversion and I/O functionality.
Internally, images are represented in terms of bcolz arrays using compression.
The internal representation can be converted into meta image files (mhd).
"""
from __future__ import print_function, division
import sys
import os
import gc
import time
import logging
from collections import OrderedDict
import numpy as np
import bcolz
from lcreg.lcreg_lib import remove_rle, rescale_gvals
from lcreg.lcreg_util import array_to_string, string_to_array

BCOLZ_COMPRESSION_LIMIT_MB = 128


class Image3D(object):
    """
        Image3D class containing image data (numpy or on-disk bcolz array)
        and geometry. Essential properties required for the
        creation of meta image files (mhd) are stored in bcolz.attrs.
    """

    def __init__(
        self, in_data, offset=None, spacing=None, transform_matrix=None
    ):
        """
        Image3D constructor.

        Initialises Image3D geometry using the passed offset, spacing
        and transform matrix (rotation).

        Args:
           in_data (bcolz.carray or numpy array): grey value data.
           offset (numpy array): 3D image offset in mm (world coordinates).
                Default is [0.0, 0.0, 0.0]
           spacing (numpy array): 3D voxel spacing in mm. Default is [1,1,1]
           transform_matrix (numpy array): 3x3 image rotation matrix.
                Default is the 3x3 identity.
        """
        self._data = in_data
        if isinstance(self._data, bcolz.carray_ext.carray):
            self._mhd_param = OrderedDict(in_data.attrs["mhd_param"])
        else:
            self._mhd_param = MHD_DEFAULTS.copy()
        self._mhd_param["DimSize"] = array_to_string(
            np.array(in_data.shape[::-1], dtype=np.int64)
        )

        if offset is None:
            self._offset = np.zeros(3, dtype=np.float64)
        else:
            self._offset = offset

        if spacing is None:
            self._spacing = np.ones(3, dtype=np.float64)
        else:
            self._spacing = spacing

        if transform_matrix is None:
            self._transform_matrix = np.identity(3, dtype=np.float64)
        else:
            self._transform_matrix = transform_matrix

        self._voxel_to_world_matrix = np.identity(4, dtype=np.float64)
        self._world_to_voxel_matrix = np.identity(4, dtype=np.float64)

        self._update_geometry()

    def _update_geometry(self):
        """
        Initialise internal transformation matrices between world and
        voxel ccordinates from current offset, transformation matrix
        and spacing. If the current transformation matrix is not
        invertible, an exception is risen.
        Note that the mhd orientation is transposed according to
        the ITK conventions.
        """
        self._mhd_param["Offset"] = array_to_string(self._offset)
        self._mhd_param["TransformMatrix"] = array_to_string(
            self._transform_matrix.T
        )
        self._mhd_param["ElementSpacing"] = array_to_string(self._spacing)

        if (
            isinstance(self._data, bcolz.carray_ext.carray)
            and self._data.mode != "r"
        ):
            self._data.attrs["mhd_param"] = list(self._mhd_param.items())

        try:
            self._voxel_to_world_matrix.fill(0.0)
            self._voxel_to_world_matrix[0:3, 3] = self._offset
            self._voxel_to_world_matrix[3, 3] = 1
            self._voxel_to_world_matrix[:3, :3] = np.dot(
                self._transform_matrix, np.diag(self._spacing)
            )
            self._world_to_voxel_matrix = np.linalg.inv(
                self._voxel_to_world_matrix
            )
        except np.linalg.LinAlgError:
            raise ValueError("Illegal geometry settings")

    @property
    def mhd_param(self):
        """
        Getter for mhd parameters.

        Returns:
            OrderedDict mhd parameters
        """
        return self._mhd_param.copy()

    @property
    def offset(self):
        """
        Getter for 3D image offset.

        Returns:
            numpy array, length 3: Image offset in mm
        """
        return self._offset[:]

    @offset.setter
    def offset(self, new_offset):
        """
        Setter for 3D image offset.

        Changes offset and updates transformation matrices and
        bcolz.attribs for mhd export.

        Args:
            new_offset (numpy array): 3D offset in mm.
        """
        self._offset = new_offset[:]
        self._update_geometry()

    @property
    def transform_matrix(self):
        """
        Getter for image transformation matrix (rotation).

        Returns:
            3x3 numpy array: Image rotation matrix

        """
        return self._transform_matrix

    @transform_matrix.setter
    def transform_matrix(self, new_transform_matrix):
        """
            Setter for image rotation matrix.

            Changes rotation matrix and updates transformation matrices and
            bcolz.attribs for mhd export.

            Args:
                new_transform_matrix (3x3 numpy array): New rotation matrix.

        """
        self._transform_matrix = new_transform_matrix
        self._update_geometry()

    @property
    def spacing(self):
        """
        Getter for 3D image spacing.

        Returns:
            numpy array, length 3: Voxel size in mm
        """
        return self._spacing[:]

    @spacing.setter
    def spacing(self, new_spacing):
        """
        Setter for 3D image spacing.

        Changes spacing and updates transformation matrices and
        bcolz.attribs for mhd export.

        Args:
            new_spacing (numpy array): Voxel size in mm.
        """
        self._spacing = new_spacing
        self._update_geometry()

    @property
    def data(self):
        """
        Getter for 3D image data.

        Returns:
            bcolz.carray: Voxel size in mm
        """
        return self._data

    def clear_data(self):
        """
        Flush data to disk if required and free internal data
        Set data to None
        """
        if isinstance(self.data, bcolz.carray_ext.carray):
            if self.data.mode != "r":
                self.data.flush()
        # free data array
        del self._data

    @property
    def shape(self):
        """
        Getter for 3D image dimensions.

        Returns:
            tuple: Shape of 3D image data, order zyx (!)
        """
        return self._data.shape[:]

    @property
    def shape_array_xyz(self):
        """
        Getter for 3D image dimensions.

        Returns:
            numpy array: Shape of 3D image data, order xyz (!)
        """
        return np.array(self._data.shape[::-1], dtype=np.uint64)

    @property
    def nr_of_voxels(self):
        """
        Getter for 3D image number of voxels.

        Returns:
            python int: number of voxels
        """
        vox_num = 1
        for s in self._data.shape:
            vox_num *= int(s)
        return vox_num

    @property
    def mem_mb(self):
        """
        Getter for 3D image size in mb.

        Returns:
            int: Image size in mb
        """
        return 2 * self.nr_of_voxels // 2 ** 20

    @property
    def total_voxel_surface(self):
        """
        Calculate surface of all voxels

        Returns:
            float: total voxel surface in mm^2
        """

        single_voxel_surface = 2.0 * (self.spacing ** 2).sum()
        return self.nr_of_voxels * single_voxel_surface

    @property
    def world_to_voxel_matrix(self):
        """
        Getter for world to voxel coordinate transformation matrix.

        Returns:
            4x4 numpy array: World to voxel coordinate transformation
                matrix in homogeneous coordinates.
        """
        return self._world_to_voxel_matrix[:, :]

    @property
    def voxel_to_world_matrix(self):
        """
        Getter for voxel to world coordinate transformation matrix.

        Returns:
            4x4 numpy array: Voxel to world coordinate transformation
                matrix in homogeneous coordinates.
        """
        return self._voxel_to_world_matrix[:, :]

    def transform_voxel_to_world(self, vox_pos):
        """
        Transform voxel coordinate into world coordinate system.

        Args:
            vox_pos (numpy array, length 3): position in voxel coordinates
        Returns:
            numpy array, length 3: position in world coordinates
        """
        vox_pos_homogeneous = np.append(vox_pos, 1)
        return np.dot(self._voxel_to_world_matrix, vox_pos_homogeneous)[:3]

    def transform_world_to_voxel(self, world_pos):
        """
        Transform world coordinate into voxel coordinate system.

        Args:
            world_pos (numpy array, length 3): position in world coordinates
        Returns:
            numpy array, length 3: position in voxel coordinates
        """
        world_pos_homogeneous = np.append(world_pos, 1)
        return np.dot(self._world_to_voxel_matrix, world_pos_homogeneous)[:3]

    def uncompress(self):
        """
        Uncompress bcolz array in place. If the data is already uncompressed,
        the command does nothing.
        """
        if isinstance(self._data, bcolz.carray_ext.carray):
            self._data = self._data[:, :, :]


# data type for bcolz arrays
BCOLZ_DTYPE = np.int16

# Default values for mhd file parameters
MHD_DEFAULTS = OrderedDict(
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
        ("ElementType", "MET_SHORT"),
        ("HeaderSize", "0"),
        ("ElementDataFile", "binary.raw"),
    ]
)


def _calculate_header_size(raw_file_name, param):
    header_size = int(param["HeaderSize"])
    if header_size == -1:
        type_str = _MHD_TYPES[param["ElementType"]][1]
        dim_size = string_to_array(param["DimSize"], np.uint64)
        array_size_in_bytes = (
            int(np.prod(dim_size)) * np.dtype(type_str).itemsize
        )
        header_size = os.path.getsize(raw_file_name) - array_size_in_bytes
    return header_size


def read_mhd_param(in_file_name):
    """
    Read meta image header file into ordered dictionary

    Args:
        in_file_name (str): Name of meta image file including suffix.
    Returns:
        OrderedDict containing mhd settings from file.
    """
    mhd_param = MHD_DEFAULTS.copy()
    with open(in_file_name, "r") as in_file:
        for line in in_file:
            mhd_item = line[:-1].split("=")
            mhd_item[0] = mhd_item[0].strip()
            mhd_item[1] = mhd_item[1].strip()
            if mhd_item[0] in mhd_param.keys():
                mhd_param[mhd_item[0]] = mhd_item[1][:]
    return mhd_param


def write_mhd_file(mhd_param, out_file_name):
    with open(out_file_name, "w") as out_file:
        for i in mhd_param.items():
            out_file.write(i[0] + " = " + i[1] + "\n")


# conversion from mhd to numpy data types
_MHD_TYPES = {
    "MET_UCHAR": (np.uint8, "u1"),
    "MET_CHAR": (np.int8, "i1"),
    "MET_USHORT": (np.uint16, "u2"),
    "MET_SHORT": (np.int16, "i2"),
    "MET_UINT": (np.uint32, "u4"),
    "MET_INT": (np.int32, "i4"),
    "MET_ULONG": (np.uint64, "u8"),
    "MET_LONG": (np.int64, "i8"),
    "MET_FLOAT": (np.float32, "f4"),
    "MET_DOUBLE": (np.float64, "f8"),
}


def _raw_to_bcolz(
    raw_file_name,
    file_byteorder,
    dim_size,
    dtype,
    root_dir_name,
    n_bits,
    grey_range,
    is_mask,
    header_size,
):
    """
    Read contents of a binary file into a compressed bcolz array.

    A binary file containing grey values is read into a compressed bcolz
    array. Only grey values in the range specified by grey_range are used
    (clampling is applied). The number of bits to be used in the bcolz
    array can be specified resulting in an output grey value range from
    1 to 2**n_bits-1. If the byte order of the file is different from the
    system byte order, byte swapping is applied.

    Args:
        raw_file_name (str): Name of input file containing raw data
        file_byteorder (str): byte order of data in file ('big' or 'little')
        dim_size (numpy array): image dimensions
        dtype (numpy dtype): dtype of data in file
        root_dir_name (str): directory for bcolz on-disk array
        n_bits (int): number of bits to use for image grey values
        grey_range (tuple): minimum and maximum grey value to use
        is_mask (bool): flag for mask images
        header_size(int): header size in bytes
    """
    # input grey value range
    g_min, g_max = grey_range
    # output grey value range
    out_min, out_max = 1.0, 2.0 ** n_bits - 1.0
    # total number of voxels
    expected_len = int(dim_size[0]) * int(dim_size[1]) * int(dim_size[2])
    bcolz_array = None
    # voxels per slice
    swap_needed = file_byteorder != sys.byteorder
    # epsilon offset for rounding
    epsilon = np.finfo(np.float64).eps
    # read slices and append them to bcolz carray
    with open(raw_file_name, "r") as bin_file:
        # ignore header (size is given in byes)
        np.fromfile(bin_file, np.dtype("u1"), header_size)
        items_per_slice = dim_size[0] * dim_size[1]
        out_slice = np.zeros(
            shape=(1, dim_size[1], dim_size[0]), dtype=BCOLZ_DTYPE
        )
        for _ in range(dim_size[2]):
            im_slice = np.fromfile(bin_file, dtype, items_per_slice)
            im_slice = im_slice.reshape(1, dim_size[1], dim_size[0])
            if swap_needed:
                im_slice = im_slice.byteswap()
            im_slice = im_slice.astype(np.float64)
            rescale_gvals(
                im_slice,
                out_slice,
                g_min,
                g_max,
                out_min,
                out_max,
                epsilon,
                is_mask,
            )
            if bcolz_array is None:
                # create new bcolz array
                # use defaults defined in lcreg.py
                c_params = None
                if (2 * expected_len) // 2 ** 20 <= BCOLZ_COMPRESSION_LIMIT_MB:
                    c_params = bcolz.cparams(clevel=1, shuffle=bcolz.NOSHUFFLE)
                bcolz_array = bcolz.carray(
                    out_slice,
                    rootdir=root_dir_name,
                    mode="w",
                    expectedlen=expected_len,
                    cparams=c_params,
                )
            else:
                # append data
                bcolz_array.append(out_slice)

    # cleanup
    bcolz_array.flush()
    return bcolz_array


def open_image(root_dir_name, mode="r"):
    """
    Open bcolz-compressed image

    Args:
        root_dir_name (str): Name of bcolz root dir
        mode (str): Mode for bcolz file

    Returns:
        Image3D created from bcolz carray
    """
    bcolz_array = bcolz.open(root_dir_name, mode=mode)
    param = OrderedDict(bcolz_array.attrs["mhd_param"])
    offset = string_to_array(param["Offset"])
    spacing = string_to_array(param["ElementSpacing"])
    transform_matrix = (
        string_to_array(param["TransformMatrix"]).reshape((3, 3)).T
    )
    return Image3D(bcolz_array, offset, spacing, transform_matrix)


def close_image(image):
    """
    Flush image file, free internal cache and delete image

    Args:
        image (Image3D): Image to close
    """
    image.clear_data()
    del image
    gc.collect()


def import_image(
    in_file_name, root_dir_name, n_bits=None, grey_range=None, is_mask=False
):
    """
    Import mhd image into internal format based on compressed
    bcolz arrays.

    Args:
       in_file_name (str): Name of meta image file to import
       root_dir_name (str): Name of bcolz root dir
       n_bits (int): Number of bits to use for internal grey value
            representation.  Default is to determine bits from grey_range
       grey_range (2-tuple): range of grey values to be used in the bcolz
            array. Values outside this range are clipped
       is_mask (bool): Flag indicating whether image is a binary mask
    """
    # read image parameters
    if in_file_name[-3:].lower() == "mhd":
        param = read_mhd_param(in_file_name)
    else:
        raise TypeError("Unsupported image type")
    # compressed images are not supported
    if param["CompressedData"] == "True":
        raise TypeError("Compressed raw data is not supported")
    # no support for input file lists
    if (
        param["ElementDataFile"] == "LIST"
        or len(param["ElementDataFile"].split()) > 1
    ):
        raise TypeError("One-Slice-Per-File Data is not supported")
    dtype = _MHD_TYPES[param["ElementType"]][0]
    # get image dimensions
    dim_size = string_to_array(param["DimSize"], np.uint64)
    # check if byte swapping is required
    if param["BinaryDataByteOrderMSB"] == "True":
        data_byteorder = "big"
    else:
        data_byteorder = "little"
    swap_needed = data_byteorder != sys.byteorder
    data_bits = n_bits
    dir_name = os.path.dirname(in_file_name)
    raw_file_name = os.path.join(dir_name, param["ElementDataFile"])
    # calculate header size
    header_size = _calculate_header_size(raw_file_name, param)
    # for float input images, the default accuracy is 10 bit
    if dtype in (np.float32, np.float64):
        g_min = np.finfo(dtype).max
        g_max = np.finfo(dtype).min
        if data_bits is None:
            data_bits = 10
    else:
        g_min = np.iinfo(dtype).max
        g_max = np.iinfo(dtype).min
    if grey_range is not None:
        # apply user defined grey value range
        g_min, g_max = grey_range
    else:
        # determine grey value range from image data
        im_dtype = _MHD_TYPES[param["ElementType"]][1]
        if param["BinaryDataByteOrderMSB"] == "True":
            im_dtype = ">" + im_dtype
        else:
            im_dtype = "<" + im_dtype
        im_data = np.memmap(
            raw_file_name,
            mode="r",
            dtype=np.dtype(im_dtype),
            offset=header_size,
        )
        g_min = im_data.min()
        g_max = im_data.max()
        del im_data
    grey_range = (g_min, g_max)
    logging.debug(
        "{} gVal range: {} - {}, offset: {}".format(
            in_file_name, g_min, g_max, header_size
        )
    )

    if data_bits is None:
        # get Nr of bits from integer grey value range
        delta = 1 - np.finfo(np.float64).eps
        data_bits = int(np.log(g_max - g_min) / np.log(2) + delta)

    # convert raw data file into bcolz array
    bcolz_array = _raw_to_bcolz(
        raw_file_name,
        data_byteorder,
        dim_size,
        dtype,
        root_dir_name,
        data_bits,
        grey_range,
        is_mask,
        header_size,
    )
    # update mhd parameters
    param["ElementDataFile"] = os.path.basename(root_dir_name)
    param["CompressedData"] = "True"
    if sys.byteorder == "big":
        param["BinaryDataByteOrderMSB"] = "True"
    else:
        param["BinaryDataByteOrderMSB"] = "False"
    param["ElementType"] = "MET_SHORT"
    # store mhd parameters in bcolz attributes
    bcolz_array.attrs["mhd_param"] = list(param.items())
    # write first log entry for this image
    bcolz_array.attrs["log"] = [
        time.asctime() + ": " + 'imported from file "' + in_file_name + '"',
    ]
    if not is_mask:
        bcolz_array.attrs["lv_max"] = str((2 ** data_bits - 1) ** 2)
    # cleanup
    bcolz_array.flush()
    del bcolz_array


def export_image(im, file_name):
    """
    Export an image to mhd format.

    Args:
        im (Image3D): image
        file_name: name of output file
    """
    param = im.mhd_param.copy()
    param["ElementDataFile"] = os.path.basename(file_name)[:-4] + ".raw"
    write_mhd_file(param, file_name)
    data_file_name = os.path.join(
        os.path.dirname(file_name), param["ElementDataFile"]
    )
    with open(data_file_name, "wb") as out_file:
        out_file.write(im.data[:, :, :].copy())


def export_bcolz_image(root_dir_name, out_file_name, take_abs=False):
    """
    Export internal image representation to meta image

    Args:
        root_dir_name (str): bcolz root dir name
        out_file_name(str): name of output meta image including suffix
    """
    bcolz_array = bcolz.open(root_dir_name, mode="r")
    # get mhd parameters
    param = OrderedDict(bcolz_array.attrs["mhd_param"])
    param["CompressedData"] = "False"
    # name of mhd raw file
    param["ElementDataFile"] = os.path.basename(out_file_name)[:-3] + "raw"
    write_mhd_file(param, out_file_name)
    data_file_name = os.path.join(
        os.path.dirname(out_file_name), param["ElementDataFile"]
    )
    # store image data slice by slice into raw file
    with open(data_file_name, "wb") as out_file:
        out_slice = np.zeros(
            (bcolz_array.shape[1], bcolz_array.shape[2]), dtype=np.int16
        )
        for slice_ctr in range(bcolz_array.shape[0]):
            remove_rle(bcolz_array[slice_ctr, :, :], out_slice)
            if take_abs:
                out_file.write(np.abs(out_slice))
            else:
                out_file.write(out_slice)
    del bcolz_array


def export_bcolz_image_main():
    #
    # usage message
    if len(sys.argv) != 3:
        print("Export compressed image to mhd format")
        print("usage: compressed_to_mhd bcolz_dir_name out_mhd_name")
    else:
        export_bcolz_image(sys.argv[1], sys.argv[2])


def create_image(
    root_dir_name,
    dim_size,
    offset=None,
    spacing=None,
    transform_matrix=None,
    mode="w",
):
    """
    Create Image3D including a new compressed bcolz file.

    Args:
        root_dir_name (str): bcolz root dir name
        dim_size (numpy array): 3D image dimensions
        offset (numpy array): 3D offset (geometry)
        spacing (numpy array): 3D voxel size
        transform_matrix (3x3 numpy array): rotation matrix
        mode (str): image file mode

    Returns:
        Image3D image object
    """
    expected_len = int(dim_size[0]) * int(dim_size[1]) * int(dim_size[2])
    # create emtpy slice
    im_slice = np.zeros(shape=(1, dim_size[1], dim_size[2]), dtype=BCOLZ_DTYPE)

    # use default defined in lcreg.py
    c_params = None
    if (2 * expected_len) // 2 ** 20 <= BCOLZ_COMPRESSION_LIMIT_MB:
        c_params = bcolz.cparams(clevel=1, shuffle=bcolz.NOSHUFFLE)
    data = bcolz.carray(
        im_slice,
        rootdir=root_dir_name,
        cparams=c_params,
        expectedlen=expected_len,
        mode=mode,
    )
    # append missing sizes
    for _ in range(dim_size[0] - 1):
        data.append(im_slice)
    # create and fill in mhd parameters
    param = MHD_DEFAULTS.copy()
    param["ElementDataFile"] = os.path.basename(root_dir_name)
    param["DimSize"] = array_to_string(dim_size[::-1])
    if offset is not None:
        param["Offset"] = array_to_string(offset)
    if spacing is not None:
        param["ElementSpacing"] = array_to_string(spacing)
    if transform_matrix is not None:
        param["TransformMatrix"] = array_to_string(transform_matrix.T)
    data.attrs["mhd_param"] = list(param.items())
    # create log attribute
    data.attrs["log"] = [
        time.asctime()
        + ": "
        + 'image created with root_dir_name "'
        + root_dir_name
        + '", dim_size='
        + array_to_string(dim_size)
    ]
    data.flush()
    return Image3D(data, offset, spacing, transform_matrix)


def empty_image_like(template, root_dir_name):
    """
    Create an image with grey values set to zero from a template.

    Args:
        template (Image3D): template image defining image size and geometry.

    Returns:
        Image3D with same size and geometry as template but with all
        grey values set to zero.

    """
    return create_image(
        root_dir_name,
        template.data.shape,
        template.offset,
        template.spacing,
        template.transform_matrix,
    )


def image_part(source_im, offset_voxel, size_voxel):
    """
    Extract and uncompress a subimage.

    Args:
        source_im (Image3D): source to extract part from
        offset_voxel (numpy int array of length 3): 3D offset in voxels
        size_voxel   (numpy int array of length 3): 3D subimage size in voxels

    Returns:
        Image3D image with correct geometry information
    """
    # calculate new offset
    # (Position of voxel 0 of the image part in world coordinates
    new_world_offset = source_im.transform_voxel_to_world(offset_voxel)
    new_max_vox = offset_voxel + size_voxel
    new_data = source_im.data[
        offset_voxel[2] : new_max_vox[2],
        offset_voxel[1] : new_max_vox[1],
        offset_voxel[0] : new_max_vox[0],
    ]
    # create new image3d with adapted data and geometry
    return Image3D(
        new_data,
        new_world_offset,
        source_im.spacing,
        source_im.transform_matrix,
    )


def reduce_mhd(in_im_name, out_im_name, skip_3d):
    """
        Downsample an image by simply skipping grey values.

        Args:
            in_im_name (str): Name of input mhd image
            out_im_name (str): Name of output mhd image
            skip_3d (np.array): integer skipping factors, order: xyz
    """
    in_mhd = read_mhd_param(in_im_name)
    in_dtype = _MHD_TYPES[in_mhd["ElementType"]][0]
    in_shape = string_to_array(in_mhd["DimSize"], np.uint64)[::-1]
    in_data_file_name = os.path.join(
        os.path.dirname(in_im_name), in_mhd["ElementDataFile"]
    )
    in_data = np.memmap(
        in_data_file_name, dtype=in_dtype, mode="r", shape=tuple(in_shape)
    )
    out_shape = in_data[:: skip_3d[2], :: skip_3d[1], :: skip_3d[0]].shape
    out_spacing = string_to_array(in_mhd["ElementSpacing"]) * skip_3d
    out_mhd = in_mhd.copy()
    out_mhd["ElementSpacing"] = array_to_string(out_spacing)
    out_dtype = in_dtype
    out_data_name = os.path.basename(out_im_name)[:-3] + "raw"
    out_raw_name = os.path.join(os.path.dirname(out_im_name), out_data_name)
    out_mhd["ElementDataFile"] = os.path.basename(out_data_name)
    out_mhd["DimSize"] = array_to_string(out_shape[::-1])
    write_mhd_file(out_mhd, out_im_name)
    out_data = np.memmap(
        out_raw_name, dtype=out_dtype, shape=tuple(out_shape), mode="write"
    )
    out_data[:, :, :] = in_data[:: skip_3d[2], :: skip_3d[1], :: skip_3d[0]]
    out_data.flush()
    del in_data
    del out_data


def resample_mhd(
    fixed_mhd_file_name,
    moving_mhd_file_name,
    out_mhd_file_name,
    matrix,
    offset,
):
    """
        Resample an mhd image using scipy and np.memmap

        Images do not need to fit into memory

        Args:
            fixed_mhd_file_name (str): name of fixed image mhd file
            moving_mhd_file_name (str): name of moving image mhd file
            out_mhd_file_name (str): name of output image mhd file
            matrix (3x3 np.array): rotation matrix
            offset (1x3 np.array): offset vector
    """
    try:
        from scipy.ndimage import affine_transform
    except ImportError:
        logging.error("scipy not installaed")
        print("image resampling requires scipy")
        return
    # read mhd files
    fixed_mhd = read_mhd_param(fixed_mhd_file_name)
    out_mhd = fixed_mhd.copy()
    moving_mhd = read_mhd_param(moving_mhd_file_name)
    # get shapes and dtypes
    out_shape = string_to_array(out_mhd["DimSize"], np.uint64)[::-1]
    out_dtype = _MHD_TYPES[out_mhd["ElementType"]][1]
    if out_mhd["BinaryDataByteOrderMSB"] == "True":
        out_dtype = ">" + out_dtype
    else:
        out_dtype = "<" + out_dtype
    moving_shape = string_to_array(moving_mhd["DimSize"], np.uint64)[::-1]
    moving_dtype = _MHD_TYPES[moving_mhd["ElementType"]][1]
    if moving_mhd["BinaryDataByteOrderMSB"] == "True":
        moving_dtype = ">" + moving_dtype
    else:
        moving_dtype = "<" + moving_dtype
    # update and store output mhd parameters
    out_data_name = os.path.basename(out_mhd_file_name)[:-3] + "raw"
    out_raw_name = os.path.join(
        os.path.dirname(out_mhd_file_name), out_data_name
    )
    out_mhd["ElementDataFile"] = out_data_name
    out_mhd["HeaderSize"] = "0"
    write_mhd_file(out_mhd, out_mhd_file_name)
    # open mmaps
    moving_data_file_name = os.path.join(
        os.path.dirname(moving_mhd_file_name), moving_mhd["ElementDataFile"]
    )
    moving_offset = _calculate_header_size(moving_data_file_name, moving_mhd)
    moving_data = np.memmap(
        moving_data_file_name,
        dtype=np.dtype(moving_dtype),
        mode="r",
        shape=tuple(moving_shape),
        offset=moving_offset,
    )
    out_data = np.memmap(
        out_raw_name,
        dtype=np.dtype(out_dtype),
        shape=tuple(out_shape),
        mode="write",
        offset=0,
    )
    # perform transform using linear interpolation
    affine_transform(
        moving_data,
        matrix,
        offset,
        output_shape=out_data.shape,
        output=out_data,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    # cleanup
    out_data.flush()
    del moving_data
    del out_data


def difference_image_mhd(
    im1_mhd_file_name, im2_mhd_file_name, out_mhd_file_name, calculate_abs=True
):
    """
        Calculate absolute voxel by voxel difference betwen two images.

        Image sizes must agree but images need not to fit into memory.

        Args:
            im1_mhd_file_name (str): name of first image mhd file
            im2_mhd_file_name (str): name of second image mhd file
            out_mhd_file_name (str): name of output image mhd file
            calculate_abs (boolean): flag for absolute value calculation
    """
    # read mhd files
    im1_mhd = read_mhd_param(im1_mhd_file_name)
    out_mhd = im1_mhd.copy()
    im2_mhd = read_mhd_param(im2_mhd_file_name)
    # get shapes and dtypes
    out_shape = string_to_array(out_mhd["DimSize"], np.uint64)[::-1]
    out_dtype = _MHD_TYPES[out_mhd["ElementType"]][1]
    if out_mhd["BinaryDataByteOrderMSB"] == "True":
        out_dtype = ">" + out_dtype
    else:
        out_dtype = "<" + out_dtype
    im1_shape = string_to_array(im1_mhd["DimSize"], np.uint64)[::-1]
    im1_dtype = _MHD_TYPES[im1_mhd["ElementType"]][1]
    if im1_mhd["BinaryDataByteOrderMSB"] == "True":
        im1_dtype = ">" + im1_dtype
    else:
        im1_dtype = "<" + im1_dtype
    im2_shape = string_to_array(im2_mhd["DimSize"], np.uint64)[::-1]
    im2_dtype = _MHD_TYPES[im2_mhd["ElementType"]][1]
    if im2_mhd["BinaryDataByteOrderMSB"] == "True":
        im2_dtype = ">" + im2_dtype
    else:
        im2_dtype = "<" + im2_dtype
    # check sizes
    if not np.array_equal(im2_shape, im1_shape):
        print("ERROR: Image sizes must agree\n")
        sys.exit(0)
    else:
        # update and store output mhd parameters
        out_data_name = os.path.basename(out_mhd_file_name)[:-3] + "raw"
        out_raw_name = os.path.join(
            os.path.dirname(out_mhd_file_name), out_data_name
        )
        out_mhd["ElementDataFile"] = out_data_name
        out_mhd["HeaderSize"] = "0"
        write_mhd_file(out_mhd, out_mhd_file_name)
        # open mmaps
        im1_data_file_name = os.path.join(
            os.path.dirname(im1_mhd_file_name), im1_mhd["ElementDataFile"]
        )
        im1_offset = _calculate_header_size(im1_data_file_name, im1_mhd)
        im1_data = np.memmap(
            im1_data_file_name,
            dtype=np.dtype(im1_dtype),
            mode="r",
            shape=tuple(im1_shape),
            offset=im1_offset,
        )
        im2_data_file_name = os.path.join(
            os.path.dirname(im2_mhd_file_name), im2_mhd["ElementDataFile"]
        )
        im2_offset = _calculate_header_size(im2_data_file_name, im2_mhd)
        im2_data = np.memmap(
            im2_data_file_name,
            dtype=np.dtype(im2_dtype),
            mode="r",
            shape=tuple(im2_shape),
            offset=im2_offset,
        )
        out_data = np.memmap(
            out_raw_name,
            dtype=np.dtype(out_dtype),
            shape=tuple(out_shape),
            mode="write",
            offset=0,
        )
        # calculate difference
        if calculate_abs:
            np.abs(im1_data - im2_data, out=out_data)
        else:
            np.subtract(im1_data, im2_data, out=out_data)
        # cleanup
        out_data.flush()
        del im1_data
        del im2_data
        del out_data


def abs_difference_mhd_main():
    #
    # usage message
    if len(sys.argv) < 3:
        print("calculate absolute difference between two mhd images")
        print("usage: abs_difference_mhd im1_name im2_name out_im_name")
    else:
        difference_image_mhd(
            sys.argv[1], sys.argv[2], sys.argv[3], calculate_abs=True
        )


def difference_mhd_main():
    #
    # usage message
    if len(sys.argv) < 3:
        print("calculate signed difference between two mhd images")
        print("usage: difference_mhd im1_name im2_name out_im_name")
    else:
        difference_image_mhd(
            sys.argv[1], sys.argv[2], sys.argv[3], calculate_abs=False
        )
