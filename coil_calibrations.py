import numpy as np
from raw_data import RawData
import matplotlib.pyplot as plt
from lmfit import Model
import universal_params as univ
from matrix_angles import polarisation2coil

import sys

np.set_printoptions(threshold=sys.maxsize)

plt.rcParams.update({'font.size': 18})
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Arial']
# SAMPLE_PG = "PG"
PEAK_NUC = (0, 1, 1)
PEAK_MAG = tuple(map(sum, zip((0, 1, 1), (-0.0155, -0.0155, -0.0155))))
# SAMPLES = [SAMPLE_PG, SAMPLE_MNSI]

FOLDER_1D_MNSI = "Scan_1D_MnSi"

FOLDER_SCAN_2D = "Scan_2D"

OBJ_OI = univ.oi_posn
OBJ_OF = univ.of_posn
OBJ_II = univ.ii_posn
OBJ_IF = univ.if_posn
OBJ_SINGLE = [OBJ_OI, OBJ_OF, OBJ_II, OBJ_IF]
OBJ_FIT = "FitAll"
OBJ_CNT = "CountAll"
OBJ_MAG = "MagPeak"

FIT_POL = "Polarisation"
FIT_CNT = "Count"

VAR_C_OI = "c_oi"
VAR_C_OF = "c_of"
VAR_C_II = "c_ii"
VAR_C_IF = "c_if"
VAR_COILS = [VAR_C_OI, VAR_C_OF, VAR_C_II, VAR_C_IF]

VAR_PRE_OI = 'pre_oi'
VAR_PRE_OF = 'pre_of'
VAR_PRE_II = 'pre_ii'
VAR_PRE_IF = 'pre_if'
VAR_PREFACTORS = [VAR_PRE_OI, VAR_PRE_OF, VAR_PRE_II, VAR_PRE_IF]

VAR_SHIFT_OI = 'shift_oi'
VAR_SHIFT_OF = 'shift_of'
VAR_SHIFT_II = 'shift_ii'
VAR_SHIFT_IF = 'shift_if'
VAR_SHIFT_IIF = 'shift_iif'
VAR_SHIFTS = [VAR_SHIFT_OI, VAR_SHIFT_OF, VAR_SHIFT_II, VAR_SHIFT_IF]

VAR_MXX = "m_xx"
VAR_MYX = "m_yx"
VAR_MZX = "m_zx"
VAR_MXY = "m_xy"
VAR_MYY = "m_yy"
VAR_MZY = "m_zy"
VAR_MXZ = "m_xz"
VAR_MYZ = "m_yz"
VAR_MZZ = "m_zz"
VAR_MATRIX = [VAR_MXX, VAR_MYX, VAR_MZX, VAR_MXY, VAR_MYY, VAR_MZY, VAR_MXZ, VAR_MYZ, VAR_MZZ]

VAR_P1 = "p1"
VAR_P2 = "p2"
VAR_P3 = "p3"
VAR_PARR = [VAR_P1, VAR_P2, VAR_P3]

VAR_NUC = "nuc"
VAR_MAG_Y = "mag_y"
VAR_MAG_Z = "mag_z"
VAR_INTFR_Y = "intfr_y"
VAR_INTFR_Z = "intfr_z"
VAR_MAG_MIX = "mag_mix"
VAR_CHIRAL = "chiral"
VAR_RE_INTFR_Y = "re_intfr_y"
VAR_RE_INTFR_Z = "re_intfr_z"
VAR_TENSOR = [VAR_NUC, VAR_MAG_Y, VAR_MAG_Z, VAR_INTFR_Y, VAR_INTFR_Z, VAR_MAG_MIX, VAR_CHIRAL, VAR_RE_INTFR_Y,
              VAR_RE_INTFR_Z]

VAR_DELTA_I = 'delta_i'
VAR_DELTA_F = 'delta_f'
VAR_P_INIT = 'p_init'
VAR_AMP = "amp"
VAR_SIGN = "pm"

SIGN_NSF = 1
SIGN_SF = -1
COLOUR_NSF = "blue"
COLOUR_SF = "orange"

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

SHIFT_LIM = 0.1 * np.pi


class Measurement:
    def __init__(self, peak):
        self.peak = peak
        if self.peak == PEAK_NUC:
            self.nos_oi_nsf = [3173]
            self.nos_oi_sf = [3174]

            self.nos_of_nsf = [3175]
            self.nos_of_sf = [3176]

            self.nos_ii_nsf = [3179]
            self.nos_ii_sf = [3180]

            self.nos_if_nsf = [3181, 3183]
            self.nos_if_sf = [3182, 3184]

            self.nos_fit_nsf = [3173, 3175] + list(range(3179, 3216 + 1, 2)) + list(range(3218, 3229 + 1, 2)) + list(
                range(3230, 3257 + 1, 4)) + list(range(3231, 3257 + 1, 4))
            self.nos_fit_sf = [3174, 3176] + list(range(3180, 3216 + 1, 2)) + list(range(3219, 3229 + 1, 2)) + list(
                range(3232, 3257 + 1, 4)) + list(range(3233, 3257 + 1, 4))
            self.nos_fit_nsf.sort()
            self.nos_fit_sf.sort()

            self.nos_crt_nsf = [3173, 3175]
            self.nos_crt_sf = [3174, 3176]
            self.nos_ref_nsf = list(range(3185, 3216 + 1, 2)) + list(range(3218, 3223 + 1, 2))
            self.nos_ref_sf = list(range(3186, 3216 + 1, 2)) + list(range(3219, 3223 + 1, 2))
            self.nos_ref_nsf.sort()
            self.nos_ref_sf.sort()

            self.theta = np.deg2rad(99.0 / 2.0)

            self.matrix = np.identity(3)
            self.p_arr = np.zeros(3)
        elif self.peak == PEAK_MAG:
            self.nos_mag_nsf = list(range(3259, 3286 + 1, 4)) + list(range(3260, 3286 + 1, 4)) + list(
                range(3287, 3328 + 1, 2))
            self.nos_mag_sf = list(range(3261, 3286 + 1, 4)) + list(range(3262, 3286 + 1, 4)) + list(
                range(3288, 3328 + 1, 2))
            self.nos_mag_nsf.sort()
            self.nos_mag_sf.sort()

            self.theta = np.deg2rad(97.12 / 2.0)

            self.matrix = np.array([[0, -1, -1], [0, 0, -1], [0, -1, 0]])
            # self.matrix[0] = 1
            self.p_arr = np.array([0, 0, 0])
        else:
            raise RuntimeError("Invalid measured peak position given: {}".format(self.peak))

        self.delta_i = np.pi / 2.0 - self.theta
        self.delta_f = np.pi / 2.0 + self.theta


class FitParams:
    def __init__(self, obj, pre_oi=0.75, pre_of=-0.75, pre_ii=1.4, pre_if=1.0, shift_oi=0.0, shift_of=0.0,
                 shift_ii=0.0, shift_if=0.0, p_init=0.793, amp=None, sign=None):
        self.obj = obj
        self.pre_oi = pre_oi
        self.pre_of = pre_of
        self.pre_ii = pre_ii
        self.pre_if = pre_if
        self.shift_oi = shift_oi
        self.shift_of = shift_of
        self.shift_ii = shift_ii
        self.shift_if = shift_if
        self.p_init = p_init
        self.amp = amp
        self.sign = sign
        self.pol_mat_flat = np.empty(9)
        self.p_cr8 = np.empty(3)  # the extra polarisation created by the scattering

    def lmfit_loader(self, meas, monitor, currents, data_obj):
        prefactors = [self.pre_oi, self.pre_of, self.pre_ii, self.pre_if]
        shifts = [self.shift_oi, self.shift_of, self.shift_ii, self.shift_if]
        if self.obj in OBJ_SINGLE:
            c_ind = OBJ_SINGLE.index(self.obj)  # gives the index of the coil that is the scan object
            fmodel = Model(coil2pol, independent_vars=[VAR_COILS[c_ind]])
            coils_scanned = [False, False, False, False]
            coils_scanned[c_ind] = True
            coil_fitted = True
            print("Model ready for {}".format(self.obj))
        elif self.obj in [OBJ_FIT, OBJ_CNT, OBJ_MAG]:
            coils_scanned = [True, True, True, True]
            coil_fitted = True
            if self.obj == OBJ_MAG:
                coil_fitted = False
                fmodel = Model(coil2pol, independent_vars=VAR_COILS)
            elif self.obj == OBJ_FIT:
                fmodel = Model(coil2pol, independent_vars=VAR_COILS)
            else:
                fmodel = Model(coil2count, independent_vars=VAR_COILS)
                amp_init = (np.max(data_obj) - np.min(data_obj)) / 2.0
                fmodel.set_param_hint(VAR_AMP, value=amp_init, min=amp_init * 0.9, max=amp_init * 1.1)
                if self.sign is None:
                    raise RuntimeError("The sign has to be given for fitting neutron counts")
                fmodel.set_param_hint(VAR_SIGN, value=self.sign, vary=False)
            print(
                "All coils are scanned for {}, the independent variables: {}".format(self.obj, fmodel.independent_vars))
        else:
            raise RuntimeError("Invalid scan object given: {}".format(self.obj))

        for i in range(4):
            set_coil(model=fmodel, coil_scanned=coils_scanned[i], coil_fitted=coil_fitted, index=i, currents=currents,
                     prefactors=prefactors, shifts=shifts)

        fmodel.set_param_hint(VAR_DELTA_I, value=meas.delta_i, vary=False)
        fmodel.set_param_hint(VAR_DELTA_F, value=meas.delta_f, vary=False)
        set_pinit(model=fmodel, meas=meas, p_init=self.p_init)
        set_matrix(model=fmodel, meas=meas)
        params = fmodel.make_params()

        if self.obj == OBJ_IF:
            params[VAR_SHIFT_II].expr = VAR_SHIFT_IF
        else:
            params[VAR_SHIFT_IF].expr = VAR_SHIFT_II
        print(params)

        print("All coils are scanned for {}, the independent variables: {}".format(self.obj, fmodel.independent_vars))

        if self.obj == OBJ_OI:
            result = fmodel.fit(data_obj, params, c_oi=currents[0], weights=monitor)
        elif self.obj == OBJ_OF:
            result = fmodel.fit(data_obj, params, c_of=currents[1], weights=monitor)
        elif self.obj == OBJ_II:
            result = fmodel.fit(data_obj, params, c_ii=currents[2], weights=monitor)
        elif self.obj == OBJ_IF:
            result = fmodel.fit(data_obj, params, c_if=currents[3], weights=monitor)
        else:
            result = fmodel.fit(data_obj, params, c_oi=currents[0], c_of=currents[1], c_ii=currents[2],
                                c_if=currents[3], weights=monitor)
            print(self.obj, fmodel.independent_vars)
            if self.obj == OBJ_CNT:
                self.amp = result.params[VAR_AMP].value
            if self.obj == OBJ_MAG:
                self.pol_mat_flat = [result.params[var].value for var in VAR_MATRIX]
            # # print("Variables matrix: {}".format(VAR_MATRIX))
            # # print("Matrix: {}".format(self.matrix))
            # print(["Matrix element {}, value {}".format(var, result.params[var].value) for var in VAR_MATRIX])
            # print(VAR_MATRIX)
        print(result.fit_report())
        prefactors = [result.params[var].value for var in VAR_PREFACTORS]
        shifts = [result.params[var].value for var in VAR_SHIFTS]

        self.pre_oi, self.pre_of, self.pre_ii, self.pre_if = prefactors
        self.shift_oi, self.shift_of, self.shift_ii, self.shift_if = shifts
        self.p_init = result.params[VAR_P_INIT].value
        if self.obj != OBJ_CNT:
            f = open(univ.fname_parameters, "w+")
            f.write("# Pre-factors outer incoming, outer final, inner incoming, inner final\n")
            f.write("# Shift outer incoming, outer final, inner incoming, inner final\n")
            f.write("{:f}, {:f}, {:f}, {:f}\n".format(*prefactors))
            f.write("{:f}, {:f}, {:f}, {:f}\n".format(*shifts))
            f.close()


def pol_rot_in(alpha_i, beta_i, delta_i, p_init):
    pix = np.cos(delta_i + beta_i) * np.sin(alpha_i)
    piy = np.sin(delta_i + beta_i) * np.sin(alpha_i)
    piz = np.cos(alpha_i)
    pi = np.array([pix, piy, piz]) * p_init
    return pi


def pol_rot_out(alpha_f, beta_f, delta_f, pf_scatt):
    a, b, c = pf_scatt
    pf_out = np.sin(alpha_f) * (-a * np.cos(beta_f - delta_f) + b * np.sin(beta_f - delta_f)) + c * np.cos(alpha_f)
    return pf_out


def coils2angles(currents, prefactors, shifts):
    return list(map(lambda i: coil2angle(prefactors[i], currents[i], shifts[i]), range(len(currents))))


def coil2angle(current, prefactor, shift):
    return prefactor * current + shift


def pol_mat_cal(px, py, pz, nuc, mag_y, mag_z, intfr_y, intfr_z, mag_mix, chiral, re_intfr_y, re_intfr_z):
    sig_x = nuc + mag_y + mag_z + px * abs(chiral)
    sig_y = nuc + mag_y + mag_z + py * re_intfr_y
    sig_z = nuc + mag_y + mag_z + pz * re_intfr_z
    sig_tot = nuc + mag_y + mag_z + px * abs(chiral) + py * re_intfr_y + pz * re_intfr_z
    tensor_xx = (nuc - mag_y - mag_z) / sig_x
    tensor_yx = intfr_z / sig_y
    tensor_zx = -intfr_y / sig_z
    tensor_xy = -intfr_z / sig_x
    tensor_yy = (nuc + mag_y - mag_z) / sig_y
    tensor_zy = -mag_mix / sig_z
    tensor_xz = intfr_y / sig_x
    tensor_yz = mag_mix / sig_y
    tensor_zz = (nuc - mag_y + mag_z) / sig_z
    created_x = chiral / sig_tot
    created_y = re_intfr_y / sig_tot
    created_z = re_intfr_z / sig_tot
    pol_tensor = np.array(
        [[tensor_xx, tensor_yx, tensor_zx], [tensor_xy, tensor_yy, tensor_zy], [tensor_xz, tensor_yz, tensor_zz]])
    created_pol = np.array([created_x, created_y, created_z])
    return pol_tensor, created_pol


def coil2pol(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, delta_i,
             delta_f, p_init, m_xx, m_yx, m_zx, m_xy, m_yy, m_zy, m_xz, m_yz, m_zz, p1, p2, p3):  #
    # calculate the polarisation given the coil parameters
    # The shifts of the inner coils are entangled and hence denoted by only one parameter
    trf_mtx = np.array([[m_xx, m_yx, m_zx], [m_xy, m_yy, m_zy], [m_xz, m_yz, m_zz]])
    indep_arr = np.array([p1, p2, p3])
    currents = (c_oi, c_of, c_ii, c_if)
    prefactors = (pre_oi, pre_of, pre_ii, pre_if)
    shifts = (shift_oi, shift_of, shift_ii, shift_if)
    alpha_i, alpha_f, beta_i, beta_f = coils2angles(currents, prefactors, shifts)

    pi_scatt = pol_rot_in(alpha_i=alpha_i, beta_i=beta_i, delta_i=delta_i, p_init=p_init)
    pf_scatt = np.matmul(trf_mtx, pi_scatt)
    for i in range(pf_scatt.shape[0]):
        pf_scatt[i] += indep_arr[i]
    pf_out_z = pol_rot_out(alpha_f=alpha_f, beta_f=beta_f, delta_f=delta_f, pf_scatt=pf_scatt)
    return pf_out_z


def coil2count(amp, pm, c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
               delta_i, delta_f, p_init, m_xx, m_yx, m_zx, m_xy, m_yy, m_zy, m_xz, m_yz, m_zz, p1, p2, p3):  #
    # gives the neutron count providing the coil parameters
    # pm is the sign
    pol = coil2pol(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
                   delta_i, delta_f, p_init, m_xx, m_yx, m_zx, m_xy, m_yy, m_zy, m_xz, m_yz, m_zz, p1, p2, p3)  #
    return amp * (pol * pm + 1)


def count2norm(count, monitor):
    # norm the count according to the monitor count = 1e6
    return count / monitor * 1e6


def norm2count(count_normed, monitor):
    # gives back the real count from the normed count
    return count_normed * 1e-6 * monitor


def set_prefactor(model, name, value, vary=True):
    if vary is True:
        model.set_param_hint(name, value=value, min=value - abs(value) * 0.5, max=value + abs(value) * 0.5)
    else:
        model.set_param_hint(name, value=value, vary=False)


def set_shift(model, name, value, vary=True):
    if vary is True:
        model.set_param_hint(name, value=value, min=value - SHIFT_LIM, max=value + SHIFT_LIM)
    else:
        model.set_param_hint(name, value=value, vary=False)


def set_current(model, name, value, vary=False):
    if vary is False:
        model.set_param_hint(name, value=np.round(np.mean(value), 1), vary=False)
    else:
        raise RuntimeError("The current of {:s} can be varied and should be an independent variable.".format(name))


def set_coil(model, coil_scanned, coil_fitted, index, currents, prefactors, shifts):
    if coil_scanned is True:
        if VAR_COILS[index] in model.param_names:
            model.param_names.remove(VAR_COILS[index])
    else:
        set_current(model=model, name=VAR_COILS[index], value=currents[index], vary=False)
    set_prefactor(model=model, name=VAR_PREFACTORS[index], value=prefactors[index], vary=coil_scanned and coil_fitted)
    set_shift(model=model, name=VAR_SHIFTS[index], value=shifts[index], vary=coil_scanned and coil_fitted)


def set_matrix(model, meas):
    if meas.peak == PEAK_NUC:
        for i in range(len(VAR_MATRIX)):
            model.set_param_hint(VAR_MATRIX[i], value=meas.pol_mat_flat.flatten()[i], vary=False)
        for i in range(len(VAR_PARR)):
            model.set_param_hint(VAR_PARR[i], value=meas.p_cr8[i], vary=False)
    elif meas.peak == PEAK_MAG:
        for i in range(len(VAR_MATRIX)):
            model.set_param_hint(VAR_MATRIX[i], value=meas.pol_mat_flat.flatten()[i], min=-1.1,
                                 max=1.1)  # min=-1.1, max=1.1
        for i in range(len(VAR_PARR)):
            model.set_param_hint(VAR_PARR[i], value=meas.p_cr8[i], min=-1.1, max=1.1)  # min=-0.01, max=0.01
    else:
        raise RuntimeError("The meas object {} does not belong to any known classes".format(meas))


def set_pinit(model, meas, p_init):
    if isinstance(meas, Measurement):
        model.set_param_hint(VAR_P_INIT, value=p_init, min=0.7, max=0.85)
    else:
        model.set_param_hint(VAR_P_INIT, value=p_init, vary=False)
    # model.set_param_hint(VAR_P_INIT, value=(961 - 111) / (961 + 111), vary=False)


def current_adjust(coil, period):
    if coil < 4 - period:
        coil += period
    elif coil > period:
        coil -= period
    else:
        pass
    return coil


def data_loader(number):
    # print(number)
    scan_coils = np.empty((4, 0))
    # for number in number_list:
    data_file = RawData(number)
    data = data_file.scan_data
    if len(data.shape) == 1:
        data_4d = np.zeros((4, data.shape[0]))
        index = univ.positions.index(data_file.scan_posn)
        data_4d[index, :] = data
        scan_coils = np.append(scan_coils, data_4d, axis=1)
    else:
        scan_coils = np.append(scan_coils, data, axis=1)
    counts = data_file.scan_count
    monitors = np.repeat(data_file.monitor_count, counts.shape[0])
    return scan_coils, counts, monitors


def data_combine(numbers):
    scan_coils_tot = np.empty((4, 0))
    counts_tot = np.array([], dtype=int)
    monitors_tot = np.array([], dtype=int)
    for number in numbers:
        data_4d, counts, monitors = data_loader(number)
        scan_coils_tot = np.append(scan_coils_tot, data_4d, axis=1)
        counts_tot = np.append(counts_tot, counts)
        monitors_tot = np.append(monitors_tot, monitors)
    return scan_coils_tot, counts_tot, monitors_tot


def get_correction_factor(counts, counts_ref, coils_scan, coils_ref, monitors, monitors_ref):
    correct_factor = np.array([])
    for i in range(counts.shape[0]):
        for j in range(counts_ref.shape[0]):
            if abs(coils_scan[0, i] - coils_ref[0, j]) < 9e-2 and abs(coils_scan[1, i] - coils_ref[1, j]) < 9e-2:
                norm_counts = count2norm(count=counts[i], monitor=monitors[i])
                norm_counts_ref = count2norm(count=counts_ref[j], monitor=monitors_ref[j])
                correct_factor = np.append(correct_factor, norm_counts_ref / norm_counts)
                continue
    correct_factor = np.mean(correct_factor)
    return correct_factor


def correct_count(correct_obj, correct_ref):
    scan_coils, counts, monitors = data_loader(correct_obj)
    scan_coils_ref, counts_ref, monitors_ref = data_combine(correct_ref)

    correct_factor = get_correction_factor(counts=counts, counts_ref=counts_ref, coils_scan=scan_coils,
                                           coils_ref=scan_coils_ref, monitors=monitors, monitors_ref=monitors_ref)
    counts_corrected = np.round(norm2count(counts * correct_factor, monitors))
    # print(counts, "\n", counts_corrected)
    return counts_corrected


def plot_1d(meas, nos_p, nos_m, params_pol, params_nsf=None, params_sf=None):
    if_scan = False
    prefactors = (params_pol.pre_oi, params_pol.pre_of, params_pol.pre_ii, params_pol.pre_if)
    shifts = (params_pol.shift_oi, params_pol.shift_of, params_pol.shift_ii, params_pol.shift_if)
    for numor, scan_number_p in enumerate(nos_p):
        data_file_p = RawData(scan_number_p)
        index = univ.positions.index(data_file_p.scan_posn)
        coils = data_file_p.scan_data
        if isinstance(meas, Measurement):
            if scan_number_p in meas.nos_crt_nsf:
                counts_p = correct_count(scan_number_p, meas.nos_ref_nsf)
            else:
                counts_p = data_file_p.scan_count
        else:
            counts_p = data_file_p.scan_count
        monitor = data_file_p.monitor_count
        scan_x = data_file_p.scan_x
        if if_scan is False:
            if data_file_p.scan_posn == univ.if_posn:
                if_scan = True
                file_extra = RawData(nos_p[numor + 1])
                coils = np.append(coils, file_extra.scan_data, axis=1)
                counts_p = np.append(counts_p, file_extra.scan_count)

                scan_x = np.append(scan_x, file_extra.scan_x)
                # monitor = np.append(counts_p, file_extra.monitor_count)
        else:
            if_scan = False
            continue
        if coils.ndim == 1:
            coil_4d = np.zeros((4, coils.shape[0]))
            coil_4d[index] = coils
        else:
            coil_4d = coils
        scan_number_m = nos_m[numor]
        data_file_m = RawData(scan_number_m)
        if isinstance(meas, Measurement):
            if scan_number_m in meas.nos_crt_sf:
                counts_m = correct_count(scan_number_m, meas.nos_ref_sf)
            else:
                counts_m = data_file_m.scan_count
        else:
            counts_m = data_file_m.scan_count

        if if_scan is True:
            file_extra = RawData(nos_m[numor + 1])
            counts_m = np.append(counts_m, file_extra.scan_count)
        # print(scan_number_p, data_file_p.scan_posn, counts_p.shape, counts_m.shape)
        pols = univ.count2pol(counts_p, counts_m)

        coil_plot = np.linspace(-3, 3, 61)
        coils_plot = []
        ttl_coil = []
        ttl_ang = []
        for posn in OBJ_SINGLE:
            if data_file_p.scan_posn == posn:
                coils_plot.append(coil_plot)
            else:
                cindex = OBJ_SINGLE.index(posn)
                current_now = np.round(np.mean(coil_4d[cindex]), 2)
                coils_plot.append(np.repeat(current_now, coil_plot.shape[0]))
                ang_now = coil2angle(current=current_now, prefactor=prefactors[cindex], shift=shifts[cindex])
                ttl_coil.append(posn)
                ttl_ang.append(ang_now)
        if len(ttl_coil) == 3 and len(ttl_ang) == 3:
            # plt_ttl = "{}: {:.2f}°, {}: {:.2f}°, {}: {:.2f}°".format(ttl_coil[0], np.rad2deg(ttl_ang[0]), ttl_coil[1],
            #                                                          np.rad2deg(ttl_ang[1]), ttl_coil[2],
            #                                                          np.rad2deg(ttl_ang[2]))
            plt_ttl = "{}: {:.2f}, {}: {:.2f}, {}: {:.2f}".format(ttl_coil[0], ttl_ang[0], ttl_coil[1], ttl_ang[1],
                                                                  ttl_coil[2], ttl_ang[2])
        else:
            raise RuntimeError("Failed to find the coils and/or currents.")
        c_oi, c_of, c_ii, c_if = coils_plot[0], coils_plot[1], coils_plot[2], coils_plot[3]

        fit_pol = coil2pol(c_oi, c_of, c_ii, c_if, params_pol.pre_oi, params_pol.pre_of, params_pol.pre_ii,
                           params_pol.pre_if, params_pol.shift_oi, params_pol.shift_of, params_pol.shift_ii,
                           params_pol.shift_if, meas.delta_i, meas.delta_f, params_pol.p_init,
                           *params_pol.pol_mat_flat, *params_pol.p_cr8)  #

        filename = "Scan{:d}_{:d}.png".format(scan_number_p, scan_number_m)
        filename = "/".join([FOLDER_1D_MNSI, filename])
        fig, ax = plt.subplots(figsize=(15, 10))

        sort_ind = np.argsort(scan_x)
        scan_x = scan_x[sort_ind]
        counts_p = counts_p[sort_ind]
        counts_m = counts_m[sort_ind]
        pols = pols[sort_ind]
        ax.errorbar(scan_x, counts_p, yerr=np.sqrt(counts_p), color=COLOUR_NSF, fmt="1", markersize=10,
                    label="Meas NSF")
        ax.errorbar(scan_x, counts_m, yerr=np.sqrt(counts_m), color=COLOUR_SF, fmt="2", markersize=10, label="MeasSF")
        ax.set_xlabel("{} (A)".format(data_file_p.scan_posn))
        ax.set_ylabel("Counts")
        ax.set_title(plt_ttl)
        ax.tick_params(axis="both", direction="in")
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        colour_ax2 = "green"
        ax2.plot(scan_x, pols, "3", markersize=10, color=colour_ax2, label="MeasPol")
        if params_nsf and params_sf:
            fit_nsf = coil2count(params_nsf.amp, params_nsf.sign, c_oi, c_of, c_ii, c_if, params_nsf.pre_oi,
                                 params_nsf.pre_of, params_nsf.pre_ii, params_nsf.pre_if, params_nsf.shift_oi,
                                 params_nsf.shift_of, params_nsf.shift_ii, params_nsf.shift_if, meas.delta_i,
                                 meas.delta_f, params_nsf.p_init, *params_pol.pol_mat_flat, *params_pol.p_cr8)  #
            # print(scan_number_p, "\n", pol_nsf, "\n", fit_nsf)
            fit_sf = coil2count(params_sf.amp, params_sf.sign, c_oi, c_of, c_ii, c_if, params_sf.pre_oi,
                                params_sf.pre_of, params_sf.pre_ii, params_sf.pre_if, params_sf.shift_oi,
                                params_sf.shift_of, params_sf.shift_ii, params_sf.shift_if, meas.delta_i, meas.delta_f,
                                params_sf.p_init, *params_pol.pol_mat_flat, *params_pol.p_cr8)  #
            fit_nsf = norm2count(fit_nsf, monitor)
            fit_sf = norm2count(fit_sf, monitor)
            ax.plot(coil_plot, fit_nsf, color=COLOUR_NSF, label="FitNSF")
            ax.plot(coil_plot, fit_sf, color=COLOUR_SF, label="FitSF")
        ax2.plot(coil_plot, fit_pol, color=colour_ax2, label="FitPol")
        ax2.tick_params(axis="y", direction="in")
        ax2.set_ylabel(r"Polarisation", color=colour_ax2)
        ax2.tick_params(axis='y', color=colour_ax2, labelcolor=colour_ax2)
        ax.legend(loc=2)
        ax2.legend(loc=1)
        ax2.set_ylim(-1.0, 1.0)
        fig.savefig(filename, bbox_inches='tight')
        # plt.show()
        plt.close(fig)


def coils_fitting(meas, scan, pre_oi=None, pre_of=None, pre_ii=None, pre_if=None, shift_oi=None, shift_of=None,
                  shift_iif=None, p_init=None):
    print("Scan in process: {}".format(scan), pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_iif, p_init)
    if scan == OBJ_OI:
        nos_p = meas.nos_oi_nsf
        nos_m = meas.nos_oi_sf
    elif scan == OBJ_OF:
        nos_p = meas.nos_of_nsf
        nos_m = meas.nos_of_sf
    elif scan == OBJ_II:
        nos_p = meas.nos_ii_nsf
        nos_m = meas.nos_ii_sf
    elif scan == OBJ_IF:
        nos_p = meas.nos_if_nsf
        nos_m = meas.nos_if_sf
    elif scan == OBJ_FIT:
        nos_p = meas.nos_fit_nsf
        nos_m = meas.nos_fit_sf
        # print("scanfit", nos_p)
    elif scan == OBJ_CNT:
        nos_p = meas.nos_fit_nsf
        nos_m = meas.nos_fit_sf
        # print(nos_p, "\n", nos_m)
        if len(nos_p) != len(nos_m):
            raise RuntimeError("Different numbers of files with NSF and SF\nNSF:{}\nSF:{}".format(nos_p, nos_m))
    elif scan == OBJ_MAG:
        nos_p = meas.nos_mag_nsf
        nos_m = meas.nos_mag_sf
        print("File numbers NSF: {}\n File numbers SF: {}".format(nos_p, nos_m))
        # print(nos_p, "\n", nos_m)
        if len(nos_p) != len(nos_m):
            raise RuntimeError("Different numbers of files with NSF and SF\nNSF:{}\nSF:{}".format(nos_p, nos_m))
    else:
        raise ValueError("Invalid scan type.")

    counts_p = np.array([], dtype=int)
    counts_m = np.array([], dtype=int)
    scan_coils = np.empty((4, 0))
    scan_no_p = np.array([], dtype=int)
    scan_no_m = np.array([], dtype=int)
    monitor = np.array([], dtype=int)
    index_remove_p = []
    index_remove_m = []
    for no_nsf in nos_p:
        data_file = RawData(no_nsf)
        if data_file.relevant_scan is False:
            print("Scan No. {} is not complete".format(no_nsf))
            index_remove_p.append(no_nsf)
            continue
        data = data_file.scan_data
        if len(data.shape) == 1:
            data_4d = np.zeros((4, data.shape[0]))
            index = univ.positions.index(data_file.scan_posn)
            data_4d[index, :] = data
            scan_coils = np.append(scan_coils, data_4d, axis=1)
        else:
            scan_coils = np.append(scan_coils, data, axis=1)
        if scan == OBJ_CNT and no_nsf in meas.nos_crt_nsf:
            cnt_ntn = correct_count(no_nsf, meas.nos_ref_nsf)
        else:
            cnt_ntn = data_file.scan_count
        counts_p = np.append(counts_p, cnt_ntn)
        scan_no_p = np.append(scan_no_p, np.repeat(no_nsf, cnt_ntn.shape[0]))

        monitor = np.append(monitor, np.repeat(data_file.monitor_count, cnt_ntn.shape[0]))
    nos_p = [x for x in nos_p if x not in index_remove_p]

    for no_sf in nos_m:
        data_file = RawData(no_sf)
        if data_file.relevant_scan is False:
            print("Scan No. {} is not relevant".format(no_sf))
            index_remove_m.append(no_sf)
            continue
        if scan == OBJ_CNT and no_sf in meas.nos_crt_sf:
            cnt_ntn = correct_count(no_sf, meas.nos_ref_sf)
        else:
            cnt_ntn = data_file.scan_count
        counts_m = np.append(counts_m, cnt_ntn)
        scan_no_m = np.append(scan_no_m, np.repeat(no_sf, cnt_ntn.shape[0]))
    nos_m = [x for x in nos_m if x not in index_remove_m]

    pols = univ.count2pol(counts_p, counts_m)
    if scan == OBJ_OI:
        oi_params = FitParams(obj=scan)
        oi_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils, data_obj=pols)
        print(oi_params.pre_oi, oi_params.shift_oi, oi_params.p_init)
        return oi_params.pre_oi, oi_params.shift_oi, oi_params.p_init
    elif scan == OBJ_OF:
        of_params = FitParams(obj=scan, pre_oi=pre_oi, shift_oi=shift_oi, p_init=p_init)
        of_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils, data_obj=pols)
        return of_params.pre_oi, of_params.shift_oi, of_params.pre_of, of_params.shift_of, of_params.p_init
    elif scan == OBJ_II:
        ii_params = FitParams(obj=scan, pre_oi=pre_oi, shift_oi=shift_oi, pre_of=pre_of, shift_of=shift_of,
                              p_init=p_init)
        ii_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils, data_obj=pols)
        return ii_params.pre_oi, ii_params.shift_oi, ii_params.pre_of, ii_params.shift_of, ii_params.pre_ii, ii_params.shift_ii, ii_params.p_init
    elif scan == OBJ_IF:
        if_params = FitParams(obj=scan, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, shift_oi=shift_oi,
                              shift_of=shift_of, shift_ii=shift_iif, p_init=p_init)
        if_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils, data_obj=pols)
        return if_params.pre_oi, if_params.shift_oi, if_params.pre_of, if_params.shift_of, if_params.pre_ii, if_params.shift_ii, if_params.pre_if, if_params.shift_if, if_params.p_init
    elif scan == OBJ_FIT:
        # print(scan_no_p)
        fit_params = FitParams(obj=scan, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if, shift_oi=shift_oi,
                               shift_of=shift_of, shift_ii=shift_iif, shift_if=shift_if, p_init=p_init)
        fit_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils, data_obj=pols)
        # print(counts_p, counts_m)
        # print(scan_no_p)
        f_coils = open("CoilCurrents.dat", "w+")  #
        # print(len(scan_no_p), scan_coils.shape[1])
        for i in range(scan_coils.shape[1]):
            f_coils.write(
                "{}, {}, {:f}, {:f}, {:f}, {:f}, {:f}, {:.0f}, {:.0f}, {:.0f}\n".format(scan_no_p[i], scan_no_m[i],
                                                                                        scan_coils[0, i],
                                                                                        scan_coils[1, i],
                                                                                        scan_coils[2, i],
                                                                                        scan_coils[3, i], pols[i],
                                                                                        counts_p[i], counts_m[i],
                                                                                        monitor[i]))
        f_coils.close()
        return fit_params.pre_oi, fit_params.shift_oi, fit_params.pre_of, fit_params.shift_of, fit_params.pre_ii, fit_params.shift_ii, fit_params.pre_if, fit_params.shift_if, fit_params.p_init
    elif scan == OBJ_CNT:
        print("Nummers NSF: {}".format(nos_p))
        fit_params = FitParams(obj=scan, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if, shift_oi=shift_oi,
                               shift_of=shift_of, shift_ii=shift_iif, shift_if=shift_iif, p_init=p_init)
        nsf_params = FitParams(obj=scan, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if, shift_oi=shift_oi,
                               shift_of=shift_of, shift_ii=shift_iif, shift_if=shift_iif, p_init=p_init, sign=SIGN_NSF)
        nsf_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils,
                                data_obj=count2norm(counts_p, monitor))

        sf_params = FitParams(obj=scan, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if, shift_oi=shift_oi,
                              shift_of=shift_of, shift_ii=shift_iif, shift_if=shift_iif, p_init=p_init, sign=SIGN_SF)
        sf_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils,
                               data_obj=count2norm(counts_m, monitor))

        # plot_1d(meas, nos_p, nos_m, fit_params, nsf_params, sf_params)
        # fname = "PolMat_Nuc.dat"
        # f_nuc = open(fname, "w+")
        # f_nuc.write("pi, pf, pif, corrected by p_init = {:.3f}\n".format(fit_params.p_init))
        # for pi in AXES:
        #     for pf in AXES:
        #         pol_axes = (pi, pf)
        #         deltas = (measure.delta_i, measure.delta_f)
        #         prefactors = (fit_params.pre_oi, fit_params.pre_of, fit_params.pre_ii, fit_params.pre_if)
        #         shifts = (fit_params.shift_oi, fit_params.shift_of, fit_params.shift_ii, fit_params.shift_if)
        #         coils_mtx = polarisation2coil(pol_axes=pol_axes, deltas=deltas, prefactors=prefactors, shifts=shifts)
        #         nsf = coil2count(nsf_params.amp, SIGN_NSF, *coils_mtx, pre_oi=nsf_params.pre_oi,
        #                          pre_of=nsf_params.pre_of, pre_ii=nsf_params.pre_ii, pre_if=nsf_params.pre_if,
        #                          shift_oi=nsf_params.shift_oi, shift_of=nsf_params.shift_of,
        #                          shift_ii=nsf_params.shift_ii, shift_if=nsf_params.shift_if, delta_i=measure.delta_i,
        #                          delta_f=measure.delta_f, p_init=nsf_params.p_init)
        #         sf = coil2count(sf_params.amp, SIGN_SF, *coils_mtx, pre_oi=sf_params.pre_oi, pre_of=sf_params.pre_of,
        #                         pre_ii=sf_params.pre_ii, pre_if=sf_params.pre_if, shift_oi=sf_params.shift_oi,
        #                         shift_of=sf_params.shift_of, shift_ii=sf_params.shift_ii, shift_if=sf_params.shift_if,
        #                         delta_i=measure.delta_i, delta_f=measure.delta_f, p_init=sf_params.p_init)
        #         pol = univ.count2pol(nsf, sf)
        #         f_nuc.write("{:s}, {:s}, {:.3f}, {:.3f}\n".format(pi, pf, pol, pol / fit_params.p_init))
        # f_nuc.close()
    elif scan == OBJ_MAG:
        fit_params = FitParams(obj=scan, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if, shift_oi=shift_oi,
                               shift_of=shift_of, shift_ii=shift_iif, shift_if=shift_iif, p_init=p_init)
        fit_params.lmfit_loader(meas=meas, monitor=monitor, currents=scan_coils, data_obj=pols)
        f_coils = open("CoilCurrents_Mag.dat", "w+")  #
        # print(len(scan_no_p), scan_coils.shape[1])
        for i in range(scan_coils.shape[1]):
            f_coils.write(
                "{}, {}, {:f}, {:f}, {:f}, {:f}, {:f}, {:.0f}, {:.0f}, {:.0f}\n".format(scan_no_p[i], scan_no_m[i],
                                                                                        scan_coils[0, i],
                                                                                        scan_coils[1, i],
                                                                                        scan_coils[2, i],
                                                                                        scan_coils[3, i], pols[i],
                                                                                        counts_p[i], counts_m[i],
                                                                                        monitor[i]))
        f_coils.close()
        # print(fit_params.matrix)
        pol_mat = np.array(fit_params.pol_mat_flat).reshape((3, 3))
        pol_ext = np.array(fit_params.p_cr8)
        pix = np.array([1, 0, 0])
        piy = np.array([0, 1, 0])
        piz = np.array([0, 0, 1])
        pf1 = np.matmul(pol_mat, pix) + pol_ext
        pf2 = np.matmul(pol_mat, piy) + pol_ext
        pf3 = np.matmul(pol_mat, piz) + pol_ext
        print(pf1, pf2, pf3)
        # plot_1d(meas, nos_p, nos_m, fit_params)
        # coi = 2.05
        # cii = 3.0
        # ai = pre_oi * coi + shift_oi
        # bi = pre_ii + cii + shift_ii
        # pi_scatt = pol_rot_in(ai, bi, meas.delta_i, p_init)
        # print("Pi_scatt = {}".format(np.rad2deg(pi_scatt)))
        # pf_scatt = np.array([0, 0, np.sqrt(np.pi / 4.0)])
        # print(np.matmul(pf_scatt, np.transpose(pi_scatt)))


measure = Measurement(peak=PEAK_NUC)

pre_oi, shift_oi, p_init = coils_fitting(meas=measure, scan=OBJ_OI)
# c_oi = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_oi, shift=shift_oi)
# print("Outer coil upstream {:.3f} A".format(c_oi))

pre_oi, shift_oi, pre_of, shift_of, p_init = coils_fitting(meas=measure, scan=OBJ_OF, pre_oi=pre_oi, shift_oi=shift_oi,
                                                           p_init=p_init)
# c_of = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_of, shift=shift_of)
# print("Outer coil downstream {:.3f} A".format(c_of))
# print(pre_oi, pre_of, shift_oi, shift_of)

pre_oi, shift_oi, pre_of, shift_of, pre_ii, shift_iif, p_init = coils_fitting(meas=measure, scan=OBJ_II, pre_oi=pre_oi,
                                                                              pre_of=pre_of, shift_oi=shift_oi,
                                                                              shift_of=shift_of, p_init=p_init)
# c_ii = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_ii, shift=shift_ii)
pre_oi, shift_oi, pre_of, shift_of, pre_ii, shift_ii, pre_if, shift_if, p_init = coils_fitting(meas=measure,
                                                                                               scan=OBJ_IF,
                                                                                               pre_oi=pre_oi,
                                                                                               pre_of=pre_of,
                                                                                               pre_ii=pre_ii,
                                                                                               shift_oi=shift_oi,
                                                                                               shift_of=shift_of,
                                                                                               shift_iif=shift_iif,
                                                                                               p_init=p_init)
# c_if = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_if, shift=shift_if)
# print("Inner coil upstream {:.3f} A".format(c_ii))


pre_oi, shift_oi, pre_of, shift_of, pre_ii, shift_ii, pre_if, shift_if, p_init = coils_fitting(meas=measure,
                                                                                               scan=OBJ_FIT,
                                                                                               pre_oi=pre_oi,
                                                                                               pre_of=pre_of,
                                                                                               pre_ii=pre_ii,
                                                                                               pre_if=pre_if,
                                                                                               shift_oi=shift_oi,
                                                                                               shift_of=shift_of,
                                                                                               shift_iif=shift_ii,
                                                                                               p_init=p_init)

coils_fitting(meas=measure, scan=OBJ_CNT, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if,
              shift_oi=shift_oi, shift_of=shift_of, shift_iif=shift_ii, p_init=p_init)

measure = Measurement(peak=PEAK_MAG)
coils_fitting(meas=measure, scan=OBJ_MAG, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if,
              shift_oi=shift_oi, shift_of=shift_of, shift_iif=shift_ii, p_init=p_init)
