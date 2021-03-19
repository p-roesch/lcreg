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
main program for 3d image registration
"""
import sys
import logging
import configparser
import os
import ast
import multiprocessing
from sys import modules, argv
import time
import datetime
import shutil
import stat
import platform
import lcreg
import psutil

import numpy as np
import scipy
import bcolz

from lcreg import image3d
from lcreg import multires_pyramid
from lcreg.image_transform import (
    swap_RAS_LPS,
    p_from_t_matrix_world,
    t_matrix_world_from_p,
)
from lcreg import masking
from lcreg.optimisation import (
    simple_gradient,
    levenberg_marquardt,
    get_initial_transform_from_moments,
)
from lcreg.lcreg_util import system_memory_available_mb
from lcreg.lcreg_util import array_to_string
from lcreg.lcreg_util import image_corners_world
from lcreg.image_transform import euler_from_matrix


def create_output_directories(reg_config):
    """
    Create required output directories
    """
    for dir_name in (
        reg_config["DEFAULT"]["working_directory_name"],
        reg_config["DEFAULT"]["output_directory_name"],
    ):
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError:
                print(
                    'Could not create directory "' + dir_name + '" exiting ...'
                )
                exit(0)


def set_up_logging(reg_config, argv):
    """
    Initialise logging and print general information into log file
    """
    log_config = None
    log_level = reg_config["DEFAULT"]["log_level"]
    if log_level not in logging.__all__:
        raise ValueError("Illegal log level " + log_level)
    else:
        level_value = eval("logging." + log_level)
        if not isinstance(level_value, int):
            raise ValueError("Illegal log level " + log_level)
        else:
            log_format = "%(asctime)s - %(levelname)s: %(message)s"
            out_dir_name = reg_config["DEFAULT"]["output_directory_name"]
            log_filename = os.path.join(out_dir_name, "lcreg.log")
            log_config = {
                "format": log_format,
                "level": log_level,
                "filename": log_filename,
            }
            logging.basicConfig(**log_config)
            #
            # print system information
            logging.info("### lcreg " + argv[1])
            logging.info("")
            #
            # print CPU name if cpuinfo is installed
            try:
                import cpuinfo

                cpu_info = cpuinfo.get_cpu_info()
                logging.info("CPU: {}".format(cpu_info["brand_raw"]))
            except Exception:
                logging.warn("CPU: unknown type")
            #
            # print maximum CPU frequency if value is available
            freq = psutil.cpu_freq()
            if freq is not None:
                max_f = freq[2]
                logging.info("CPU max. frequency: {} GHz".format(max_f / 1000))
            #
            # nr of threads
            logging.info("CPU threads: {}".format(psutil.cpu_count()))
            #
            # print total physical memory if value is available
            mem = psutil.virtual_memory()
            if mem is not None:
                mem_total_gb = mem[0] / 2 ** 30
                logging.info("RAM: {:.1f} GiB".format(mem_total_gb))
            logging.info("")
            #
            # print Python and library versions
            logging.info(platform.platform())
            logging.info("Python {}".format(sys.version))
            logging.info("lcreg {}".format(lcreg.__version__))
            logging.info("numpy {}".format(np.__version__))
            logging.info("scipy {}".format(scipy.__version__))
            logging.info("bcolz {}".format(bcolz.__version__))
            logging.info("psutil {}".format(psutil.__version__))
            logging.info("")

    return log_config


def set_up_bcolz(reg_config):
    """
    Set global bcolz parameters
    """
    num_threads = reg_config.getint("DEFAULT", "nr_of_threads")
    if num_threads > 0:
        bcolz.blosc_set_nthreads(num_threads)
    bcolz.cparams.setdefaults(
        clevel=reg_config.getint("compression", "clevel"),
        shuffle=eval(reg_config["compression"]["shuffle"]),
        cname=reg_config["compression"]["cname"],
    )


def __import_image(t):
    """
    Import image

    Args:
        t (tuple): reg_config, name, mask_flag, log_config
    Returns:
        output name (str): name of bcolz image directory
    """
    reg_config, name, mask_flag, log_config = t
    # required for windows if multiple processes are used
    logging.basicConfig(**log_config)
    output_dir = reg_config["DEFAULT"]["working_directory_name"]
    output_name = os.path.join(output_dir, name)
    do_reuse = reg_config["DEFAULT"]["reuse_existing_input_images"] == "True"
    if do_reuse and os.path.isdir(output_name):
        logging.debug("reusing previously imported " + name + " image")
    else:
        grey_masking_range = None
        bits = None
        if not mask_flag:
            bits = reg_config.getint(
                "compression", name + "_image_bits_per_pixel"
            )
            range_str = reg_config["masking"][name + "_grey_masking_range"]
            if range_str != "None":
                values = range_str.split()
                low = ast.literal_eval(values[0])
                high = ast.literal_eval(values[1])
                grey_masking_range = (low, high)

        image3d.import_image(
            reg_config["DEFAULT"][name + "_image_name"],
            output_name,
            n_bits=bits,
            grey_range=grey_masking_range,
            is_mask=mask_flag,
        )
    return output_name


def __get_geometry(img_name):
    """
    Get geometry information of image

    Args:
        img_name (str): Name of bcolz image directory
    Returns:
        (tuple): shape, spacing, offset, transform_matrix
    """
    img = image3d.open_image(img_name)
    result = (
        img.shape_array_xyz,
        img.spacing,
        img.offset,
        img.transform_matrix,
    )
    image3d.close_image(img)
    return result


def __get_total_voxel_surface(img_name):
    """
    Get total voxel surface of image

    Args:
        img_name (str): Name of bcolz image directory
    Returns:
        (float): total voxel surface in mm^2
    """
    img = image3d.open_image(img_name)
    result = img.total_voxel_surface
    image3d.close_image(img)
    return result


def import_input_images(reg_config, log_config):
    """
    Import input images

    Args:
        rec_conig (ConfigParser): Configuration
    """
    logging.info("Image import started")
    arg_list = []
    arg_list.append((reg_config, "fixed", False, log_config))
    arg_list.append((reg_config, "moving", False, log_config))

    if reg_config["DEFAULT"]["fixed_mask_image_name"] != "None":
        arg_list.append((reg_config, "fixed_mask", True, log_config))
    if reg_config["DEFAULT"]["moving_mask_image_name"] != "None":
        arg_list.append((reg_config, "moving_mask", True, log_config))
    # import images in parallel only if not profiling
    if "profile" in modules or "cProfile" in modules:
        name_list = list(map(__import_image, arg_list))
    else:
        nr_of_threads = reg_config.getint("DEFAULT", "nr_of_threads")
        if nr_of_threads <= 0:
            nr_of_processes = min(multiprocessing.cpu_count(), len(arg_list))
        else:
            nr_of_processes = nr_of_threads
        pool = multiprocessing.Pool(nr_of_processes)
        name_list = pool.map(__import_image, arg_list)
        pool.close()
    #
    name_list.reverse()
    #
    # import fixed image
    fixed_name = name_list.pop()
    reg_config.set("import", "fixed_name", fixed_name)
    reg_config.set("pyramid", "fixed_grey_base", fixed_name)
    fixed_geometry = __get_geometry(fixed_name)
    if min(fixed_geometry[0]) < 6:
        logging.error("dimensions of fixed image too small")
        print("ERROR: dimensions of fixed image too small")
        exit()
    #
    # import moving image
    moving_name = name_list.pop()
    reg_config.set("import", "moving_name", moving_name)
    reg_config.set("pyramid", "moving_grey_base", moving_name)
    moving_geometry = __get_geometry(moving_name)
    if min(moving_geometry[0]) < 6:
        logging.error("dimensions of moving image too small")
        print("ERROR: dimensions of moving image too small")
        exit()
    #
    # import fixed mask and compare geometry with fixed_image
    if reg_config["DEFAULT"]["fixed_mask_image_name"] != "None":
        fixed_mask_name = name_list.pop()
        reg_config.set("import", "fixed_mask_name", fixed_mask_name)
        reg_config.set("pyramid", "fixed_mask_base", fixed_mask_name)
        fixed_mask_geometry = __get_geometry(fixed_mask_name)
        for i in range(len(fixed_mask_geometry)):
            if not np.allclose(fixed_geometry[i], fixed_mask_geometry[i]):
                logging.error("geometry of fixed mask and image differ")
                print("ERROR: geometry of fixed mask and image differ")
                exit()

    #
    # import moving mask and compare geometry with moving_image
    if reg_config["DEFAULT"]["moving_mask_image_name"] != "None":
        moving_mask_name = name_list.pop()
        reg_config.set("import", "moving_mask_name", moving_mask_name)
        reg_config.set("pyramid", "moving_mask_base", moving_mask_name)
        moving_mask_geometry = __get_geometry(moving_mask_name)
        for i in range(len(moving_mask_geometry)):
            if not np.allclose(moving_geometry[i], moving_mask_geometry[i]):
                logging.error("geometry of moving mask and image differ")
                print("ERROR: geometry of moving mask and image differ")
                exit()
    logging.info("Image import done")


def initialise_transform(reg_config):
    """
    Initialise tranformation

    Args:
        reg_config (ConfigParser): Confiuration
    """
    # check for illegal settings
    mat = np.identity(4, dtype=np.float64)
    in_file_name = reg_config["transformation"][
        "initial_transformation_file_name"
    ]
    if in_file_name != "None":
        if reg_config.getboolean(
            "transformation", "initialise_from_edge_moments"
        ):
            logging.error(
                "Two incompatible options for transfomation "
                + "initialisation chosen"
            )
            print(
                "ERROR: Two incompatible options for transfomation "
                + "initialisation chosen"
            )
            exit(0)
        if os.path.isfile(in_file_name):
            mat = np.loadtxt(in_file_name, dtype=np.float64)
        else:
            logging.error("could not find transformation file " + in_file_name)
    # convert RAS transform to LPS system
    if reg_config.getboolean("transformation", "initial_transform_is_RAS"):
        logging.info("converting initial transform from RAS to LPS system")
        mat = swap_RAS_LPS(mat)
    # invert initial transform if required
    if reg_config.getboolean(
        "transformation", "initial_transform_is_inverted"
    ):
        logging.info("inverting initial transform")
        mat = np.linalg.inv(mat)
    reg_config.set(
        "optimisation",
        "initial_transform",
        "np." + repr(mat).replace("=float64", "=np.float64"),
    )
    logging.debug("initial_transform: \n" + str(mat))


def create_multi_resolution_pyramid(reg_config):
    """
    Create multi resolution pyramid of fixed and moving images and masks

    Args:
        reg_config (ConfigParser): Configuration
    """
    multires_pyramid.create_pyramid(reg_config)


def perform_masking(reg_config):
    """
    Perform image masking for all resolutions available

    Args:
        reg_config (ConfigParser): Configuration
    """
    fixed_masked_images = []
    moving_masked_images = []
    nr_of_levels = len(
        ast.literal_eval(reg_config["pyramid"]["fixed_grey_files"])
    )
    fixed_gm_surface = reg_config.getfloat(
        "masking", "fixed_used_gradient_magnitude_fraction"
    ) * __get_total_voxel_surface(reg_config["import"]["fixed_name"])
    reg_config.set("masking", "fixed_gm_surface", str(fixed_gm_surface))
    moving_gm_surface = reg_config.getfloat(
        "masking", "moving_used_gradient_magnitude_fraction"
    ) * __get_total_voxel_surface(reg_config["import"]["moving_name"])

    reg_config.set("masking", "moving_gm_surface", str(moving_gm_surface))
    for l in range(nr_of_levels):
        out_files = masking.mask_level(reg_config, l)
        fixed_masked_images.append(out_files[0])
        moving_masked_images.append(out_files[1])

    reg_config["optimisation"]["fixed_masked_images"] = str(
        fixed_masked_images
    )
    reg_config["optimisation"]["moving_masked_images"] = str(
        moving_masked_images
    )


def initialise_from_moments(reg_config):
    """
    Initialise initial transformation from edge moments

    Args:
        reg_config (ConfigParser): Configuration
    """
    if reg_config.getboolean("transformation", "initialise_from_edge_moments"):
        get_initial_transform_from_moments(reg_config)


def perform_multi_resolution_registration(reg_config):
    """
    Multi resolution rigid or affine 3D registration

    Args:
        reg_config (ConfigParser): Configuration
    """
    logging.info("Multi resolution optimisation started")
    # get initial transformation parameters
    M_initial = eval(reg_config["optimisation"]["initial_transform"])
    is_affine = reg_config["transformation"]["transformation_type"] == "affine"
    p_initial = p_from_t_matrix_world(M_initial, None, is_affine)
    # get lists of image names
    fixed_images = ast.literal_eval(
        reg_config["optimisation"]["fixed_masked_images"]
    )
    moving_images = ast.literal_eval(
        reg_config["optimisation"]["moving_masked_images"]
    )

    p_opt = p_initial.copy()
    level = 0
    mem_limit_mb = ast.literal_eval(reg_config["DEFAULT"]["mem_limit_mb"])
    nr_of_threads = ast.literal_eval(reg_config["DEFAULT"]["nr_of_threads"])
    if mem_limit_mb < 0:
        mem_available = system_memory_available_mb()
    else:
        mem_available = min(system_memory_available_mb(), mem_limit_mb)

    # get levenberg_marquard parameters
    def get_config_LM(val_name):
        section = "levenberg_marquardt"
        return ast.literal_eval(reg_config[section][val_name])

    l_initial = float(get_config_LM("initial_lambda"))
    l_upscale_factor = get_config_LM("lambda_upscale_factor")
    l_downscale_factor = get_config_LM("lambda_downscale_factor")
    max_nr_of_iterations = get_config_LM("max_nr_of_iterations")
    max_nr_of_upscale_steps = get_config_LM("max_nr_of_upscale_steps")
    min_edge_voxel_displacement = get_config_LM("min_edge_voxel_displacement")
    p_opt = p_initial[:]

    level_results = []

    for fixed_name, moving_name in zip(
        fixed_images[::-1], moving_images[::-1]
    ):
        fixed_im = image3d.open_image(fixed_name)
        moving_im = image3d.open_image(moving_name)
        logging.info(
            "{:s} / {:s}".format(
                os.path.basename(fixed_name), os.path.basename(moving_name)
            )
        )
        if fixed_im.mem_mb + moving_im.mem_mb < 0.9 * mem_available:
            logging.info("  uncompressing images")
            fixed_im.uncompress()
            moving_im.uncompress()
        if reg_config.getboolean("simple_gradient", "active") and level == 0:
            delta_v_initial = ast.literal_eval(
                reg_config["simple_gradient"]["delta_v_initial"]
            )
            delta_v_min = ast.literal_eval(
                reg_config["simple_gradient"]["delta_v_min"]
            )
            c_rot = None
            v_rot = None
            if reg_config.getboolean(
                "transformation", "initialise_from_edge_moments"
            ):
                c_rot = eval(reg_config["optimisation"]["rotation_centre"])
                if not reg_config.getboolean(
                    "transformation", "initialise_shift_only"
                ):
                    v_rot = eval(reg_config["optimisation"]["rotation_axes"])

            result = simple_gradient(
                fixed_im,
                moving_im,
                p_opt,
                c_rot,
                v_rot,
                delta_v_initial,
                delta_v_min,
                mem_limit_mb,
                nr_of_threads,
            )
            logging.info(
                "simple grad., lcc_opt: {:f}, p_opt: {:s} n_steps: {:d}".format(
                    result[0], str(list(result[1])), result[2]
                )
            )
            p_opt = result[1].copy()

        result = levenberg_marquardt(
            fixed_im,
            moving_im,
            p_opt,
            l_initial,
            l_downscale_factor,
            l_upscale_factor,
            max_nr_of_iterations,
            min_edge_voxel_displacement,
            max_nr_of_upscale_steps,
            mem_limit_mb,
            nr_of_threads,
        )
        logging.info(
            "LM, lcc_opt: {:f}, p_opt: {:s}, n_steps: {:d}".format(
                result[0], str(list(result[1])), result[2]
            )
        )
        mat = t_matrix_world_from_p(result[1])
        logging.debug("m_opt:\n {:s}".format(str(mat)))
        logging.debug(
            "final euler angles (xyz, deg):\n" + str(euler_from_matrix(mat))
        )
        logging.debug("final translation vector:\n" + str(mat[:3, 3]))

        lcc_opt = result[0]
        p_opt = result[1][:]
        level_results.append((level, result))
        image3d.close_image(fixed_im)
        image3d.close_image(moving_im)
        level = level + 1
    reg_config.set(
        "optimisation",
        "level_results",
        repr(level_results)
        .replace("array", "np.array")
        .replace("float64", "np.float64"),
    )
    reg_config.set(
        "optimisation",
        "p_opt",
        repr(p_opt)
        .replace("array", "np.array")
        .replace("float64", "np.float64"),
    )
    reg_config.set("optimisation", "lcc_opt", str(lcc_opt))


def create_transformed_mhd(reg_config):
    """
    Create moving image mhd file with geometry information corresponding
        to fixed image

    Args:
        reg_config (ConfigParser): Configuration
    """

    # check if mhd transformation makes sense
    if reg_config["transformation"]["transformation_type"] != "rigid":
        return

    logging.info("create transformed mhd file")

    # get voxel to world matrix of moving image
    moving_name = reg_config["import"]["moving_name"]
    moving_image = image3d.open_image(moving_name, "r")
    moving_v2w_old = moving_image.voxel_to_world_matrix
    moving_spacing = moving_image.spacing

    # apply transform
    p_opt = eval(reg_config["optimisation"]["p_opt"])
    m_opt_inv = np.linalg.inv(t_matrix_world_from_p(p_opt))
    moving_v2w_new = np.dot(m_opt_inv, moving_v2w_old)
    # collect values
    new_offset = moving_v2w_new[0:3, 3]
    # correct for scaling
    new_transform_matrix = np.dot(
        moving_v2w_new[:3, :3], np.diag(1.0 / moving_spacing)
    )

    moving_mhd_name = reg_config["DEFAULT"]["moving_image_name"]
    moving_mhd = image3d.read_mhd_param(moving_mhd_name)

    # update mhd
    if os.path.isabs(moving_mhd["ElementDataFile"]):
        data_file_name = moving_mhd["ElementDataFile"]
    else:
        data_file_name = os.path.abspath(
            os.path.join(
                os.path.dirname(moving_mhd_name), moving_mhd["ElementDataFile"]
            )
        )

    moving_mhd["ElementDataFile"] = data_file_name
    moving_mhd["Offset"] = array_to_string(new_offset)
    # transpose rotation matrix for mhd export
    moving_mhd["TransformMatrix"] = array_to_string(new_transform_matrix.T)
    # store result
    mhd_out_name = os.path.join(
        reg_config["DEFAULT"]["output_directory_name"], "moving_registered.mhd"
    )
    image3d.write_mhd_file(moving_mhd, mhd_out_name)


def export_results(reg_config, script_suffix):
    """
    Store registration results

    Write transformation matrix, config file, c3d call script and
    transformation file according to RIRE. Perform moving image transformation
    if specified in reg_config

    Args:
        reg_config (ConfigParser): Configuration
        script_suffix: 'bat' or 'sh depending on os
    """
    # store matrix
    logging.info("storing transformation matrix")
    p_opt = eval(reg_config["optimisation"]["p_opt"])
    m_opt = t_matrix_world_from_p(p_opt)
    if reg_config.getboolean("transformation", "store_as_RAS"):
        m_opt_out = swap_RAS_LPS(m_opt)
    else:
        m_opt_out = m_opt
    out_file_name = os.path.join(
        reg_config["DEFAULT"]["output_directory_name"], "lcreg_result.mat"
    )
    np.savetxt(out_file_name, m_opt_out)
    #
    # store config file
    logging.info("storing config file")
    config_out_name = os.path.join(
        reg_config["DEFAULT"]["output_directory_name"], "lcreg_results.ini"
    )
    with open(config_out_name, "w") as out_file:
        reg_config.write(out_file)

    # store c3d call
    logging.info("storing c3d call file")
    c3d_call = "c3d "
    c3d_call += os.path.abspath(reg_config["DEFAULT"]["fixed_image_name"])
    c3d_call += " "
    c3d_call += os.path.abspath(reg_config["DEFAULT"]["moving_image_name"])
    c3d_call += " "
    c3d_call += "-reslice-matrix"
    c3d_call += " "
    c3d_call += os.path.abspath(
        os.path.join(
            reg_config["DEFAULT"]["output_directory_name"], "lcreg_result.mat"
        )
    )
    c3d_call += " "
    c3d_call += "-int Cubic -o"
    c3d_call += " "
    c3d_call += os.path.abspath(
        os.path.join(
            reg_config["DEFAULT"]["output_directory_name"],
            "resampled_moving_image_c3d.mhd",
        )
    )
    file_name = os.path.join(
        reg_config["DEFAULT"]["output_directory_name"],
        "c3d_call." + script_suffix,
    )
    with open(file_name, "w") as out_file:
        out_file.write(c3d_call)
    os.chmod(file_name, stat.S_IRWXU)

    if reg_config.getboolean("output", "resample_moving_image"):
        logging.info("resampling image with scipy.affine_transform")
        fixed_im = image3d.open_image(reg_config["import"]["fixed_name"], "r")
        moving_im = image3d.open_image(
            reg_config["import"]["moving_name"], "r"
        )
        # numpy array access order is z, y, x
        swap_xz = np.array(
            [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        t_mat_vox = np.dot(fixed_im.voxel_to_world_matrix, swap_xz)
        t_mat_vox = np.dot(m_opt, t_mat_vox)
        t_mat_vox = np.dot(moving_im.world_to_voxel_matrix, t_mat_vox)
        t_mat_vox = np.dot(swap_xz.T, t_mat_vox)
        matrix = t_mat_vox[:3, :3]
        offset = t_mat_vox[:3, 3]
        image3d.close_image(fixed_im)
        image3d.close_image(moving_im)
        image3d.resample_mhd(
            reg_config["DEFAULT"]["fixed_image_name"],
            reg_config["DEFAULT"]["moving_image_name"],
            os.path.join(
                reg_config["DEFAULT"]["output_directory_name"],
                "resampled_moving_image_scipy.mhd",
            ),
            matrix,
            offset,
        )

    if reg_config.getboolean("output", "store_RIRE_validation_file"):
        logging.info("storing RIRE validation file")
        reg_config.set("output", "RIRE_date", time.strftime("%d %B %Y"))
        reg_config.set(
            "output",
            "RIRE_from",
            reg_config.get("DEFAULT", "fixed_modality").upper(),
        )
        reg_config.set(
            "output", "RIRE_to", reg_config.get("DEFAULT", "moving_modality")
        )
        reg_config.set(
            "output",
            "RIRE_patient_number",
            reg_config.get("DEFAULT", "patient_id")[-3:],
        )
        fixed_im = image3d.open_image(reg_config["import"]["fixed_name"], "r")
        R_mat = image_corners_world(fixed_im)
        image3d.close_image(fixed_im)
        R_mat_transformed = np.dot(m_opt, R_mat)
        combined_mat = np.hstack((R_mat[:3, :].T, R_mat_transformed[:3, :].T))
        out_file_name = (
            "RIRE_"
            + reg_config.get("DEFAULT", "patient_id")
            + "_"
            + reg_config.get("DEFAULT", "modality_combination")
            + ".transformation"
        )
        file_name = os.path.join(
            reg_config["DEFAULT"]["output_directory_name"], out_file_name
        )
        with open(file_name, "w") as out_file:
            out_file.write(reg_config.get("output", "RIRE_header"))
            out_file.write("\n\n")
            for row in range(8):
                row_str = "{:3d}  ".format(row + 1)
                for col in range(6):
                    row_str += "{:11.4f}".format(combined_mat[row, col])
                out_file.write(row_str + "\n")
            out_file.write("\n(All distances are in millimeters.)\n")
            out_file.write("-" * 73 + "\n")


def cleanup(reg_config):
    """
    Remove temporary files if specified in reg_config

    Args:
        reg_config (ConfigParser): Configuration
    """

    base_file_keys = (
        ("pyramid", "fixed_grey_base"),
        ("pyramid", "moving_grey_base"),
        ("pyramid", "fixed_mask_base"),
        ("pyramid", "moving_mask_base"),
    )

    pyramid_file_keys = (
        ("optimisation", "fixed_masked_images"),
        ("optimisation", "moving_masked_images"),
        ("pyramid", "fixed_grey_files"),
        ("pyramid", "moving_grey_files"),
        ("pyramid", "fixed_mask_files"),
        ("pyramid", "moving_mask_files"),
    )

    if reg_config.getboolean("DEFAULT", "remove_temporary_images"):
        for key in base_file_keys:
            if reg_config.has_option(key[0], key[1]):
                img_name = reg_config.get(key[0], key[1])
                if os.path.isdir(img_name):
                    shutil.rmtree(img_name)
        for key in pyramid_file_keys:
            if reg_config.has_option(key[0], key[1]):
                for img_name in ast.literal_eval(
                    reg_config.get(key[0], key[1])
                ):
                    if os.path.isdir(img_name):
                        shutil.rmtree(img_name)


def create_view_scripts(reg_config, script_suffix):
    """
    Create shell scripts for visualisation of registration results

    Args:
        reg_config (ConfigParser): Configuration
        script_suffix: 'bat' or 'sh' depending on os
    """
    fixed_image_name = reg_config.get("DEFAULT", "fixed_image_name")
    moving_image_name = reg_config.get("DEFAULT", "moving_image_name")
    output_directory_name = reg_config.get("DEFAULT", "output_directory_name")
    #
    # create snap commands for overlay
    if sys.platform.startswith("win"):
        snap_exe_name = "ITK-SNAP"
    else:
        snap_exe_name = "itksnap"
    # script filenames
    snap_unreg_call_file_name = os.path.join(
        output_directory_name, "snap_unreg." + script_suffix
    )
    snap_reg_call_file_name = os.path.join(
        output_directory_name, "snap_reg." + script_suffix
    )
    # write scripts
    with open(snap_unreg_call_file_name, "w") as out_file:
        out_file.write(
            snap_exe_name
            + " -g "
            + os.path.abspath(fixed_image_name)
            + " -o "
            + os.path.abspath(moving_image_name)
        )
    if reg_config["transformation"]["transformation_type"] == "rigid":
        with open(snap_reg_call_file_name, "w") as out_file:
            out_file.write(
                snap_exe_name
                + " -g "
                + os.path.abspath(fixed_image_name)
                + " -o "
                + os.path.abspath(
                    os.path.join(
                        output_directory_name, "moving_registered.mhd"
                    )
                )
            )
    #
    # create view_bcolz command for masked pyramids
    bcolz_call_file_name = os.path.join(
        output_directory_name, "view_masked_pyramid." + script_suffix
    )
    if not reg_config.getboolean("DEFAULT", "remove_temporary_images"):
        fixed_masked_images = ast.literal_eval(
            reg_config["optimisation"]["fixed_masked_images"]
        )
        moving_masked_images = ast.literal_eval(
            reg_config["optimisation"]["moving_masked_images"]
        )
        with open(bcolz_call_file_name, "w") as out_file:
            cmd = "view_compressed_images"
            cmd = cmd + " " + os.path.abspath(output_directory_name)
            for d_name in fixed_masked_images + moving_masked_images:
                cmd = cmd + " " + os.path.abspath(d_name)
            out_file.write(cmd)
    # set execute flag on linux
    if not sys.platform.startswith("win"):
        for file_name in (
            bcolz_call_file_name,
            snap_unreg_call_file_name,
            snap_reg_call_file_name,
        ):
            if os.path.isfile(file_name):
                os.chmod(file_name, stat.S_IRWXU)


def main(parameter_file_name=None):
    """
    main function

    Args:
        parameter_file_name (str): Parameter file name
    """
    start_time = time.time()
    #
    # check command line argument
    if parameter_file_name is None and len(argv) != 2:
        print("\nusage: lcreg parameter_file_name\n")
        exit(0)
    else:
        parameter_file_name = argv[1]
    #
    # read config file
    reg_config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    try:
        reg_config.read_file(open(parameter_file_name))
    except FileNotFoundError:
        print(
            'Configuration file "'
            + parameter_file_name
            + '" not found, exiting.'
        )
        exit(0)
    #
    # add new sections
    reg_config.add_section("import")
    reg_config.add_section("pyramid")
    reg_config.add_section("optimisation")
    #
    create_output_directories(reg_config)
    #
    log_config = set_up_logging(reg_config, argv)
    #
    set_up_bcolz(reg_config)
    #
    import_input_images(reg_config, log_config)
    #
    initialise_transform(reg_config)
    #
    create_multi_resolution_pyramid(reg_config)
    #
    perform_masking(reg_config)
    #
    initialise_from_moments(reg_config)
    #
    perform_multi_resolution_registration(reg_config)
    #
    create_transformed_mhd(reg_config)
    #
    # set suffix for sh/bat files
    script_suffix = "bat" if sys.platform.startswith("win") else "sh"
    #
    export_results(reg_config, script_suffix)
    #
    create_view_scripts(reg_config, script_suffix)
    #
    cleanup(reg_config)
    #
    end_time = time.time()
    delta_t = datetime.timedelta(seconds=end_time - start_time)
    logging.info("*** lcreg stopped after {}\n".format(delta_t))


def profiling_main():
    import pstats
    import cProfile

    #
    # check command line argument
    if len(sys.argv) != 2:
        print("\nusage: python lcreg_profile parameter_file_name\n")
        sys.exit(0)
    #
    # call registration
    cProfile.runctx("main()", globals(), locals(), "lcreg.prof")
    #
    stats = pstats.Stats("lcreg.prof")
    stats.strip_dirs().sort_stats("cumtime").print_stats()


if __name__ == "__main__":
    # call registration
    main()
