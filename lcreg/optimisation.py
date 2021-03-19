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
    Levenberg-Marquardt and simple gradient optimisation
"""

import logging
import ast
from itertools import permutations
import numpy as np
from lcreg import image3d
from lcreg import similarity
from lcreg.image_transform import t_matrix_world_from_p, p_from_t_matrix_world
from lcreg.lcreg_util import image_corners_world, corners_vox_x_y_z
from lcreg.image_transform import euler_from_matrix
import lcreg.lcreg_lib as cy_ip


def inertia_tensor(im):
    N = np.zeros(1, dtype=np.int64)
    centre = np.zeros(3, dtype=np.float64)
    TI = np.zeros((3, 3), dtype=np.float64)
    for z in range(1, im.shape[0] - 1):
        cy_ip.add_inertia_tensor_slice(
            im.data[z, :, :], z, im.voxel_to_world_matrix, N, centre, TI
        )
    # fill lower part of symmetric tensor
    for i in range(0, 3):
        for k in range(i + 1, 3):
            TI[k, i] = TI[i, k]
    centre /= N[0]
    I3 = np.identity(3, dtype=np.float64)
    TI -= N[0] * (np.dot(centre, centre) * I3 - np.outer(centre, centre))
    return centre, TI


def get_initial_transform_from_moments(reg_config):
    """
    Calculate intial transform from inertia tensors of masked images.
    """
    # find and open masked images in the middle of the pyramid
    fixed_masked_list_str = reg_config["optimisation"]["fixed_masked_images"]
    fixed_masked_list = ast.literal_eval(fixed_masked_list_str)
    im_index = len(fixed_masked_list) // 2
    fixed_im_name = fixed_masked_list[im_index]
    fixed_im = image3d.open_image(fixed_im_name)
    moving_masked_list_str = reg_config["optimisation"]["moving_masked_images"]
    moving_masked_list = ast.literal_eval(moving_masked_list_str)
    im_index = len(moving_masked_list) // 2
    moving_im_name = moving_masked_list[im_index]
    moving_im = image3d.open_image(moving_im_name)

    # calculate inertia tensor slice by slice
    c_fixed, TI_fixed = inertia_tensor(fixed_im)
    c_moving, TI_moving = inertia_tensor(moving_im)
    # calculate eigenvalues and eigenvectors
    evect_fixed = np.linalg.eigh(TI_fixed)[1]
    evect_moving = np.linalg.eigh(TI_moving)[1]
    # start with identity transform
    p_opt = np.zeros(6, dtype=np.float64)
    lcc_opt = similarity.lcc(fixed_im, moving_im, p_opt, False)[1]
    logging.debug("lcc identity transform: " + str(lcc_opt))
    p_shift = p_opt.copy()
    p_shift[:3] = c_moving[:] - c_fixed[:]
    lcc_shift = similarity.lcc(fixed_im, moving_im, p_shift, False)[1]
    logging.debug("lcc shift only:" + str(lcc_shift))
    if lcc_shift < lcc_opt:
        lcc_opt = lcc_shift
        p_opt[:] = p_shift[:]
    if not reg_config.getboolean("transformation", "initialise_shift_only"):
        # point set in fixed image
        pt_fixed = np.ones((4, 4), dtype=np.float64)
        for i in range(3):
            pt_fixed[:3, i] = c_fixed[:] + evect_fixed[:, i]
        pt_fixed[:3, 3] = c_fixed[:]
        pt_fixed_inv = np.linalg.inv(pt_fixed)
        pt_moving = np.ones((4, 4), dtype=np.float64)
        pt_moving[:3, 3] = c_moving[:]
        #
        # try all permutations and signs for moving image points
        result_list = []
        s = [1, 1, 1]
        for a in permutations((0, 1, 2), 3):
            for s[0] in (-1, 1):
                for s[1] in (-1, 1):
                    for s[2] in (-1, 1):
                        for i in range(3):
                            pt_moving[:3, i] = (
                                c_moving[:] + s[i] * evect_moving[:, a[i]]
                            )
                            M = np.dot(pt_moving, pt_fixed_inv)
                            # check for real rotation
                            if abs(np.linalg.det(M[:3, :3]) - 1) < 1e-3:
                                p = p_from_t_matrix_world(M)
                                N, lcc = similarity.lcc(
                                    fixed_im, moving_im, p, False
                                )
                                result_list.append((N, lcc, p))
        # ignore solutions with very small overlap
        N_max = max([t[0] for t in result_list if t[0] is not None])
        N_limit = int(N_max * 0.75)
        # find settings corresponding to minimum lcc
        for N, lcc, p in result_list:
            if N is not None and N >= N_limit and lcc < lcc_opt:
                lcc_opt, p_opt = lcc, p

    # store result in reg_config
    m_opt = t_matrix_world_from_p(p_opt)
    reg_config["optimisation"]["initial_transform"] = (
        repr(m_opt)
        .replace("array", "np.array")
        .replace("float64", "np.float64")
    )
    reg_config["optimisation"]["rotation_centre"] = (
        repr(c_fixed)
        .replace("array", "np.array")
        .replace("float64", "np.float64")
    )
    reg_config["optimisation"]["rotation_axes"] = (
        repr(evect_fixed)
        .replace("array", "np.array")
        .replace("float64", "np.float64")
    )
    logging.debug("initial_transform from moments: \n" + str(m_opt))
    logging.debug(
        "initial euler angles (xyz, deg):\n" + str(euler_from_matrix(m_opt))
    )
    logging.debug("initial translation vector:\n" + str(m_opt[:3, 3]))
    logging.debug("initial lcc value: " + str(lcc_opt))
    image3d.close_image(fixed_im)
    image3d.close_image(moving_im)


def __image_centre_world(im):
    """
    Image centre in world coordinates

    Args:
        im (Image3D): image
    """
    return im.transform_voxel_to_world((im.shape_array_xyz[::-1] - 1) / 2)


def __delta_r_from_delta_v(im, delta_v, c_rot):
    """
    Calculate approxiate gradient descent step with for rotation parameters.

    It is assumed that delta_v corresponds to small rotation angles.

    Args:
        im (Image3D): image
        delta_v (float): step with in units of image voxel diagonals
            for image corners
        c_rot (numpy array): centre of rotation

    Returns:
        rotation parameter step width for gradient descent

    """
    # centre of image in world coordinates
    r_w_max = -1
    for x, y, z in corners_vox_x_y_z(im):
        # find image corner with largest distance to the centre
        # of the coordinate system (world coordinates)
        p_w = im.transform_voxel_to_world(
            np.array([x, y, z], dtype=np.float64)
        )
        r_w = np.linalg.norm(p_w - c_rot)
        if r_w > r_w_max:
            r_w_max = r_w
    # assume that r_x = r_y = r_z for rotation parameters
    delta_r = delta_v * np.min(im.spacing) / r_w_max
    return delta_r


def __delta_v_from_delta_M(moving_im, M, delta_M, R_c_w):
    """
    Calculate maximum additional displacement of image edges resulting from
    transformation update.

    Args:
        moving_im (Image3D): moving image
        M (numpy array, 4x4): Transformation matrix
        delta_M (numpy array, 4x4): Transformation update matrix
        R_c_w: (numpy.array, 4x8): Matrix containing image edges
            in world coordinates
    """

    M_delta = np.dot(delta_M - np.identity(4), M)
    d_world = np.dot(M_delta, R_c_w)
    d_vox = np.dot(moving_im.world_to_voxel_matrix, d_world)
    d_vox_abs = np.linalg.norm(d_vox[:3, :], axis=0)
    return max(d_vox_abs)


def __affine_matrix_from_p(p):
    """
    Create affine transformation matrix from a vector with 12 entries.

    Args:
        p (numpy.array, 1x12, np.float64): parameter vector

    Returns:
        numpy.array, 4x4, numpy.float64; transformation matrix
    """
    M = np.identity(4, dtype=np.float64)
    M[:3, :] = p.reshape(3, 4)
    return M


def simple_gradient(
    fixed_im,
    moving_im,
    p_initial,
    c_rot=None,
    v_rot=None,
    d_v_initial=2,
    d_v_min=1,
    mem_limit_mb=-1,
    num_threads=-1,
):
    """
    Perform simple gradient descent as described in
        Studholme, C., Hill, D.L.G., Hawkes, D.J.:
        Automated 3-D registration of MR and CT images of the head.
        Med. Image Anal. 1 (1996) 163--175

    Args:
        fixed_im (Image3D): fixed image
        moving_im(Image3D): moving image
        p_initial (numpy.array): initial rigid (6) parameters
        c_rot (numpy.array): rotation centre, image centre if None is passed
        v_rot (numpy.array, 3x3): rotation axes, identity if None is passed
        d_v_initial: initial step width in units of moving_im voxel diameters.
        d_v_min: minimum step width in units of moving_im voxel diameters.
        mem_limit_mb (int): memory limit for optimisation in MB
        num_threads (int): number of threads to be used or -1 (use all cores)

    Returns:
        tuple (lcc_min, p_opt, n_steps):
            lcc_min: minimum (1-LCC) value
            p_opt: parameter values yielding minimum lcc
            n_steps: number of optimisation steps performed
    """
    if len(p_initial) != 6:
        err_str = "simple gradient optimisation requires rigid transform"
        logging.error(err_str)
        raise NotImplementedError(err_str)

    if c_rot is None:
        c_rot = __image_centre_world(fixed_im)
    if v_rot is None:
        v_rot = np.identity(3, dtype=np.float64)
    # matrix to shift rotation centre to origin
    T_c = np.identity(4, dtype=np.float64)
    T_c[:3, 3] = -c_rot
    # ... and back
    T_c_inv = np.identity(4, dtype=np.float64)
    T_c_inv[:3, 3] = c_rot
    # parameter step size
    delta_t = d_v_initial * np.linalg.norm(moving_im.spacing)
    delta_t_min = d_v_min * np.linalg.norm(moving_im.spacing)
    delta_r = __delta_r_from_delta_v(moving_im, d_v_initial, c_rot)
    # initial lcc value
    N, lcc_min = similarity.lcc(
        fixed_im, moving_im, p_initial, False, mem_limit_mb, num_threads
    )
    if N is None or lcc_min is None:
        logging.error("Insufficient overlap")
        return (None, None)

    # parameter offset
    offset = np.zeros(6, dtype=np.float64)
    # initial transformation matrix
    T_initial = t_matrix_world_from_p(p_initial)
    # optimum offset matrix
    d_T_opt = np.identity(4, dtype=np.float64)
    # optimum offset matrix for current step
    d_T_opt_step = np.identity(4, dtype=np.float64)
    # transformation parameters
    p_l = p_initial.copy()

    n_steps = 0
    logging.debug("lcc_initial: " + str(lcc_min))
    while delta_t >= delta_t_min:
        lcc_min_step = lcc_min
        # loop over parameters
        sign = 1
        for k in range(12):
            offset.fill(0)
            if k < 6:
                # translation
                offset[:3] = sign * delta_t * v_rot[:, (k // 2) % 3]
            else:
                # rotation
                offset[3:] = sign * delta_r * v_rot[:, (k // 2) % 3]

            # transformation offset
            d_T = np.dot(t_matrix_world_from_p(offset), d_T_opt)
            # current transformation matrix
            T_l = np.dot(
                T_initial, np.dot(T_c_inv, np.dot(np.linalg.inv(d_T), T_c))
            )
            p_l = p_from_t_matrix_world(T_l)
            N, lcc = similarity.lcc(
                fixed_im, moving_im, p_l, False, mem_limit_mb, num_threads
            )
            if N is not None and lcc is not None:
                if lcc < lcc_min_step:
                    lcc_min_step = lcc
                    d_T_opt_step[:, :] = d_T[:, :]
            sign *= -1

        if lcc_min_step < lcc_min:
            lcc_min = lcc_min_step
            # update optimum offset value
            d_T_opt[:, :] = d_T_opt_step[:, :]
            logging.debug(
                "lcc_min: {:f}, d_p_opt: {:s}".format(
                    lcc_min, str(list(p_from_t_matrix_world(d_T_opt)))
                )
            )
        else:
            delta_t = delta_t / 2.0
            delta_r = delta_r / 2.0
        n_steps += 1

    # get parameters corresponding to d_T_opt
    T_opt = np.dot(
        T_initial, np.dot(T_c_inv, np.dot(np.linalg.inv(d_T_opt), T_c))
    )
    p_opt = p_from_t_matrix_world(T_opt)

    return lcc_min, p_opt, n_steps


def _update_p(moving_im, R_c_w, p, grad, H, l, p_new):
    """
    Update parameter vector using LM algorithm

    Args:
        moving_im (Image3D) moving image
        R_c_w (np.array, 4x8, float64): Corners of fixed
                                image in world coordinates
        p (np.array, 1x6 or 1x12, float64): current parameter vector
        grad (np.array, 1x6 or 1x12, float64): gradient vector
        H (np.array, 6x6 or 12x12): Hessian matrix
        l (float64): LM parameter lambda
        p_new (np.array, 1x6 or 1x12, float64): Vector to store result in
    Returns:
        Shift of image corners in units of voxel diagonals 
        resulting from parameter update
    """

    # LM step
    D = np.diag(np.diag(H))
    delta_p = -np.dot(np.linalg.inv(H + l * D), grad)
    # update parameter vector
    if len(p) == 6:
        # rigid transformation
        M_offset_inv = np.linalg.inv(t_matrix_world_from_p(delta_p))
        M = t_matrix_world_from_p(p)
        M_new = np.dot(M, M_offset_inv)
        p_from_t_matrix_world(M_new, p_new, False)
    else:
        # affine transformation
        M_offset_inv = t_matrix_world_from_p(-delta_p)
        M = t_matrix_world_from_p(p)
        p_new[:] = p - delta_p
    return __delta_v_from_delta_M(moving_im, M, M_offset_inv, R_c_w)


def levenberg_marquardt(
    fixed_im,
    moving_im,
    p_initial,
    l_initial=1.0,
    l_downscale_factor=1.0 / 10.0,
    l_upscale_factor=10,
    max_step_num=50,
    min_vox_delta=0.1,
    max_nr_of_upscale_steps=4,
    mem_limit_mb=-1,
    num_threads=-1,
):
    """
    Perform Levenberg-Marquardt optimisation of transformation parameters

    Args:
        fixed_im (Image3D): fixed image
        moving_im(Image3D): moving image
        p_initial (numpy.array): initial rigid (6) or affine (12) parameters
        l_inital (float): Initial value for LM lambda parameter
        l_downscale_factor (float): lambda is multiplied with this factor
            after a step which led to a decrease of lcc
        l_upscale_factor (float): lambda is multiplied with this factor
            after a step which led to an increase of lcc
        max_step_num (int): maximum number of optimisation steps before
            optimisation terminates
        min_vox_delta (float): optimisation terminates if the parameter
            update resulting in a decrease of lcc corresponds to a
            maximum displacement of image corners smaller than min_vox_delta
        max_nr_of_upscale_steps (int): maximum number of steps resulting in
            lcc increase before optimisation terminates
        mem_limit_mb (int): memory limit for optimisation in MB
        num_threads (int): number of threads to be used or -1 (use all cores)

    Returns:
        tuple (lcc_min, p_opt, n_steps):
            lcc_min: minimum (1-LCC) value
            p_opt: parameter values yielding minimum lcc
            n_steps: number of optimisation steps performed
    """
    R_c_w = image_corners_world(fixed_im)
    l = l_initial
    p = p_initial.copy()
    p_opt = p_initial.copy()

    # initialise values
    lcc_val_opt, grad_opt, H_opt = similarity.lcc(
        fixed_im, moving_im, p_opt, True, mem_limit_mb, num_threads
    )[1:]
    logging.debug("initial lcc: {:f}, lambda: {:e}".format(lcc_val_opt, l))
    # first LM step
    _update_p(moving_im, R_c_w, p_opt, grad_opt, H_opt, l, p)
    nr_of_upscale_steps = 0

    is_finished = False
    # optimisation loop
    step_opt = 0
    for step in range(1, max_step_num + 1):
        lcc_val, grad, H = similarity.lcc(
            fixed_im, moving_im, p, True, mem_limit_mb, num_threads
        )[1:]
        logging.debug(
            "step: {:d} lcc: {:f}, lambda: {:e}".format(step, lcc_val, l)
        )
        #
        downhill = lcc_val <= lcc_val_opt
        if downhill:
            lcc_val_opt = lcc_val
            step_opt = step
            l = l * l_downscale_factor
            nr_of_upscale_steps = 0
            p_opt = p[:]
            grad_opt[:] = grad[:]
            H_opt[:, :] = H[:, :]
        else:
            l = l * l_upscale_factor
            nr_of_upscale_steps += 1
        #
        # parameter update
        delta_v = _update_p(moving_im, R_c_w, p_opt, grad_opt, H_opt, l, p)
        #
        # check stop conditions
        if step > 1:
            if min_vox_delta is not None:
                if downhill and delta_v < min_vox_delta:
                    logging.debug(
                        "LM termination: delta_v: {:f}".format(delta_v)
                    )
                    is_finished = True
            if nr_of_upscale_steps >= max_nr_of_upscale_steps:
                logging.debug(
                    "LM termination: {:d} upscale steps".format(
                        nr_of_upscale_steps
                    )
                )
                is_finished = True
        if is_finished:
            break
    return (lcc_val_opt, p_opt, step_opt)
