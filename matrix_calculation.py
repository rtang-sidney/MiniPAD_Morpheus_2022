import numpy as np
import matplotlib.pyplot as plt
from matrix_angles import polarisation2angles
import universal_params as univ

plt.rcParams.update({'font.size': 18})
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Arial']
# np.set_printoptions(threshold=sys.maxsize)

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

HKL_011 = [0, 1, 1]
KAPPA_1 = [0.027, 0.017, 0.017]
HKLS = [HKL_011]
KAPPAS = [KAPPA_1]


def phase2current(precession_phase, prefactor, shift):
    if abs(precession_phase) < 1e-4:
        return 0
    current = (precession_phase - shift) / prefactor
    return current


def current_adjust(coil, period):
    if coil < 3 - period:
        coil += period
    elif coil > 3:
        coil -= period
    else:
        pass
    return coil


def matrix2current(pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, delta_i, delta_f):
    f = open("Current_di{:.0f}_df{:.0f}.dat".format(np.rad2deg(delta_i), np.rad2deg(delta_f)), "w+")
    f.write(
        "pi, pf, coil_oi (°), coil_oi (A), coil_of (°), coil_of (A), coil_ii (°), coil_ii (A), coil_if (°), coil_if (A)\n")
    period_oi = abs(2 * np.pi / pre_oi)
    period_of = abs(2 * np.pi / pre_of)
    period_ii = abs(2 * np.pi / pre_ii)
    period_if = abs(2 * np.pi / pre_if)
    for pi in AXES:
        for pf in AXES:
            alpha_i, alpha_f, beta_i, beta_f = polarisation2angles(pi, pf, delta_i, delta_f)
            c_oi = phase2current(precession_phase=alpha_i, prefactor=pre_oi, shift=shift_oi)
            c_of = phase2current(precession_phase=alpha_f, prefactor=pre_of, shift=shift_of)
            c_ii = phase2current(precession_phase=beta_i, prefactor=pre_ii, shift=shift_ii)
            c_if = phase2current(precession_phase=beta_f, prefactor=pre_if, shift=shift_if)
            c_oi = current_adjust(c_oi, period_oi)
            c_ii = current_adjust(c_ii, period_ii)
            c_if = current_adjust(c_if, period_if)
            c_of = current_adjust(c_of, period_of)
            f.write("{}, {}, {:.3f}°, {:.3f} A, {:.3f}°, {:.3f} A, {:.3f}°, {:.3f} A, {:.3f}°, {:.3f} A\n".format(
                pi, pf, np.rad2deg(alpha_i), c_oi, np.rad2deg(alpha_f), c_of, np.rad2deg(beta_i), c_ii,
                np.rad2deg(beta_f), c_if))
    f.close()


def polarisation_matrix(hkl, kappa, ixx, ixy, ixz, iyx, iyy, iyz, izx, izy, izz):
    if hkl not in HKLS:
        raise ValueError("Invalid hkl base position.")
    if kappa not in KAPPAS:
        raise ValueError("Invalid kappa vector.")
    pxx = univ.count2pol(ixx[0], ixx[1])
    pxy = univ.count2pol(ixy[0], ixy[1])
    pxz = univ.count2pol(ixz[0], ixz[1])
    pyx = univ.count2pol(iyx[0], iyx[1])
    pyy = univ.count2pol(iyy[0], iyy[1])
    pyz = univ.count2pol(iyz[0], iyz[1])
    pzx = univ.count2pol(izx[0], izx[1])
    pzy = univ.count2pol(izy[0], izy[1])
    pzz = univ.count2pol(izz[0], izz[1])
    matrix = np.array([[pxx, pyx, pzx], [pxy, pyy, pzy], [pxz, pyz, pzz]])
    fname = "{:d}{:d}{:d}_{:.3f},{:.3f},{:.3f}".format(*hkl, *kappa)
    f = open(fname, "w+")
    f.write("Incoming (i), final (j), p_ij \n")
    for j in AXES:
        for i in AXES:
            f.write("{:s}, {:s}, {:.f}\n".format(i, j, matrix[j, i]))
    f.close()


coil_params = np.loadtxt(univ.fname_parameters, skiprows=1, delimiter=",")
pre_oi, pre_of, shift_oi, shift_of = coil_params[0, :]
pre_ii, pre_if, shift_ii, shift_if = coil_params[1, :]
# theta = np.arcsin(univ.wavelength / (2 * univ.d_mnsi011))
twotheta = np.deg2rad(99.0)  # 99.0
theta = twotheta / 2.0
delta_i = np.pi / 2.0 - theta
delta_f = np.pi / 2.0 + theta

matrix2current(pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, delta_i, delta_f)
print()
