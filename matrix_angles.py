import numpy as np

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
VALID_AXES = [AXIS_X, AXIS_Y, AXIS_Z]

d = 3.355e-10
wavelength = 4.905e-10
theta = np.arcsin(wavelength / (2 * d))


def _angles_incoming(axis, delta_i):
    """
    return the two turning angles of the polarisation of the incoming beam
    :param axis: x, y or z, the incoming polarisation in the scattering frame
    :param delta_i: the angle between ki and Q vectors, equal to theta at an elastic scattering
    :return: alpha_i and beta_i, the turning angles around yi and zi axes, respectively
    """
    if axis == AXIS_X:
        alpha_i = np.pi / 2.0
        beta_i = -delta_i
    elif axis == AXIS_Y:
        alpha_i = np.pi / 2.0
        beta_i = np.pi / 2.0 - delta_i
    elif axis == AXIS_Z:
        alpha_i = 0
        beta_i = 0
    else:
        raise ValueError("Invalid axis of the incoming polarisation given: {}".format(axis))
    return alpha_i, beta_i


def _angles_outgoing(axis, delta_f):
    """
    return the two turning angles of the polarisation of the outgoing beam
    :param axis: x, y or z, the outgoing polarisation in the scattering frame
    :param delta_f: the angle between kf and Q vectors, equal to theta at an elastic scattering
    :return: alpha_f and beta_f, the turning angles around yf and zf axes, respectively
    """
    if axis == AXIS_X:
        alpha_f = -np.pi / 2.0
        beta_f = delta_f
    elif axis == AXIS_Y:
        alpha_f = np.pi / 2.0
        beta_f = np.pi / 2.0 + delta_f
    elif axis == AXIS_Z:
        alpha_f = 0
        beta_f = 0
    else:
        raise ValueError("Invalid axis of the outgoing polarisation given: {}".format(axis))
    return alpha_f, beta_f


def _adjust_angle(angle):
    while angle < 0:
        angle += 2 * np.pi
    while angle > 2 * np.pi:
        angle -= 2 * np.pi
    return angle


# def polarisation2angles(incoming_axis, outgoing_axis, delta_i, delta_f):
def polarisation2angles(pol_axes, deltas):
    """
    return the four turning angles of the polarisation of the incoming and outgoing beams
    :param pol_axes: tuple of incoming and outgoing polarisation axes: x, y or z
    :param deltas: delta_i and delta_f: angles from ki-vector to Q and from Q to kf
    :return: alpha_i, beta_i, alpha_f and beta_f, the turning angles around yi, zi, yf and zf axes, respectively
    """
    axis_in, axis_out = pol_axes
    delta_i, delta_f = deltas
    if axis_in not in VALID_AXES:
        raise ValueError(
            "Invalid axis of the incoming polarisation given: {}. It has to be one of the followings: {}".format(
                axis_in, VALID_AXES))
    if axis_out not in VALID_AXES:
        raise ValueError(
            "Invalid axis of the outgoing polarisation given: {}. It has to be one of the followings: {}".format(
                axis_out, VALID_AXES))
    alpha_i, beta_i = _angles_incoming(axis=axis_in, delta_i=delta_i)
    alpha_f, beta_f = _angles_outgoing(axis=axis_out, delta_f=delta_f)
    # alpha_f = alpha_f
    # beta_f = beta_f
    return _adjust_angle(alpha_i), _adjust_angle(alpha_f), _adjust_angle(beta_i), _adjust_angle(beta_f)


def phase2current(precession_phase, prefactor, shift):
    current = (precession_phase - shift) / prefactor
    return current


def polarisation2coil(pol_axes, deltas, prefactors, shifts):
    # pi, pf = pol_axes
    # delta_i, delta_f = deltas
    # pre_oi, pre_of, pre_ii, pre_if = prefactors
    # shift_oi, shift_of, shift_ii, shift_if = shifts
    # alpha_i, alpha_f, beta_i, beta_f = polarisation2angles(pi, pf, delta_i, delta_f)
    angles = polarisation2angles(pol_axes, deltas)
    currents = np.empty(4)
    for i in range(currents.shape[0]):
        currents[i] = phase2current(precession_phase=angles[i], prefactor=prefactors[i], shift=shifts[i])
    # c_oi_mtx = phase2current(precession_phase=alpha_i, prefactor=pre_oi, shift=shift_oi)
    # c_of_mtx = phase2current(precession_phase=alpha_f, prefactor=pre_of, shift=shift_of)
    # c_ii_mtx = phase2current(precession_phase=beta_i, prefactor=pre_ii, shift=shift_iif)
    # c_if_mtx = phase2current(precession_phase=beta_f, prefactor=pre_if, shift=shift_iif)
    return currents
