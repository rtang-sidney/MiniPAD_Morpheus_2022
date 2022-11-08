import numpy as np
from raw_data import RawData
import matplotlib.pyplot as plt
from lmfit import Model
import universal_params as univ
from matrix_angles import polarisation2angles

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

SCAN_OI = univ.oi_posn
SCAN_OF = univ.of_posn
SCAN_II = univ.ii_posn
SCAN_IF = univ.if_posn
SCAN_FIT = "FitAll"
SCAN_COUNT = "All"

FIT_POL = "Polarisation"
FIT_CNT = "Count"

VAR_C_OI = "c_oi"
VAR_C_OF = "c_of"
VAR_C_II = "c_ii"
VAR_C_IF = "c_if"

VAR_PRE_OI = 'pre_oi'
VAR_PRE_OF = 'pre_of'
VAR_PRE_II = 'pre_ii'
VAR_PRE_IF = 'pre_if'

VAR_SHIFT_OI = 'shift_oi'
VAR_SHIFT_OF = 'shift_of'
VAR_SHIFT_II = 'shift_ii'
VAR_SHIFT_IF = 'shift_if'

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


class MNSI_NUC:
    def __init__(self):
        self.scan_oi_p = [3173]  # 2223,3173
        self.scan_oi_m = [3174]  # 2224,3174
        self.scan_of_p = [3175]
        self.scan_of_m = [3176]
        self.scan_ii_p = [3179]
        self.scan_ii_m = [3180]
        self.scan_if_p = [3181, 3183]
        self.scan_if_m = [3182, 3184]
        # self.scan_fit_p = [3173, 3175, 3179, 3181, 3183]
        # self.scan_fit_m = [3174, 3176, 3180, 3182, 3184]
        self.scan_all_p = list(np.concatenate((np.arange(3173, 3216 + 1, step=2), np.arange(3218, 3229 + 1, step=2),
                                               np.arange(3230, 3257 + 1, step=4),
                                               np.arange(3231, 3257 + 1, step=4))))
        self.scan_all_m = list(np.concatenate((np.arange(3174, 3216 + 1, step=2), np.arange(3219, 3229 + 1, step=2),
                                               np.arange(3232, 3257 + 1, step=4),
                                               np.arange(3233, 3257 + 1, step=4))))
        self.scan_fit_p = self.scan_all_p
        self.scan_fit_m = self.scan_all_m
        self.theta = np.deg2rad(99.0 / 2.0)
        self.delta_i = np.pi / 2.0 - self.theta
        self.delta_f = np.pi / 2.0 + self.theta

        self.frequency_oi = 0.736418
        self.frequency_of = 0.737330
        self.frequency_ii = 1.424906
        self.frequency_if = 1.031450
        self.shift_oi = -0.007004
        self.shift_of = -0.107698
        self.shift_ii = -0.389777
        self.shift_if = -0.377453


class MNSI_MAG:
    def __init__(self):
        self.scan_all_p = list(np.concatenate((np.arange(3259, 3286 + 1, step=4), np.arange(3261, 3286 + 1, step=4),
                                               np.arange(3287, 3328 + 1, step=2))))
        self.scan_all_m = list(np.concatenate((np.arange(3260, 3286 + 1, step=4), np.arange(3262, 3286 + 1, step=4),
                                               np.arange(3288, 3328 + 1, step=2))))
        self.theta = np.deg2rad(97.12 / 2.0)
        self.delta_i = np.pi / 2.0 - self.theta
        self.delta_f = np.pi / 2.0 + self.theta


def coil2pol(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, delta_i,
             delta_f, p_init):
    alpha_i = pre_oi * c_oi + shift_oi
    alpha_f = pre_of * c_of + shift_of
    beta_i = pre_ii * c_ii + shift_ii
    beta_f = pre_if * c_if + shift_if
    pol = np.cos(alpha_i) * np.cos(alpha_f) - np.sin(alpha_i) * np.sin(alpha_f) * np.cos(
        beta_i + beta_f + delta_i - delta_f)
    pol *= p_init
    return pol


def coil2count(amp, pm, c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii,
               shift_if, delta_i, delta_f, p_init):
    # gives the neutron count providing the coil parameters
    pol = coil2pol(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
                   delta_i, delta_f, p_init)
    return amp * (pol * pm + 1)


def cosine_simple(current, scale, pre, phase, y0):
    return scale * np.cos(pre * current - phase) + y0


def phase2current(precession_phase, prefactor, shift):
    # if abs(precession_phase) < 1e-4:
    #     return 0
    current = (precession_phase - shift) / prefactor
    return current


def lmfit_loader(fmodel, meas, scan, monitor, data_oi, data_of, data_ii, data_if, data_obj, pre_oi=0.75, pre_of=-0.75,
                 pre_ii=1.4, pre_if=1.0, shift_oi=0.0, shift_of=0.0, shift_ii=0.0, shift_if=0.0, p_init=0.8, sign=None):
    fmodel.set_param_hint(VAR_DELTA_I, value=meas.delta_i, vary=False)
    fmodel.set_param_hint(VAR_DELTA_F, value=meas.delta_f, vary=False)
    fmodel.set_param_hint(VAR_P_INIT, value=p_init, min=0.7, max=1.0)
    # data_obj is either the polarisation or count
    if scan == SCAN_FIT:
        fmodel.set_param_hint(VAR_PRE_OI, value=pre_oi, min=pre_oi - abs(pre_oi) * 0.5, max=pre_oi + abs(pre_oi) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_OI, value=shift_oi, min=shift_oi - SHIFT_LIM, max=shift_oi + SHIFT_LIM)
        fmodel.set_param_hint(VAR_PRE_OF, value=pre_of, min=pre_of - abs(pre_of) * 0.5, max=pre_of + abs(pre_of) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_OF, value=shift_of, min=shift_of - SHIFT_LIM, max=shift_of + SHIFT_LIM)
        fmodel.set_param_hint(VAR_PRE_II, value=pre_ii, min=pre_ii - abs(pre_ii) * 0.5, max=pre_ii + abs(pre_ii) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_II, value=shift_ii, min=shift_ii - SHIFT_LIM, max=shift_ii + SHIFT_LIM)
        fmodel.set_param_hint(VAR_PRE_IF, value=pre_if, min=pre_if - abs(pre_if) * 0.5, max=pre_if + abs(pre_if) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_IF, value=shift_if, min=shift_if - SHIFT_LIM, max=shift_if + SHIFT_LIM)
    elif scan == SCAN_COUNT:
        if sign is None:
            raise RuntimeError("The sign has to be given for fitting neutron counts")
        print(data_obj)
        amp_init = (np.max(data_obj) - np.min(data_obj)) / 2.0
        fmodel.set_param_hint(VAR_AMP, value=amp_init, min=amp_init * 0.5, max=amp_init * 1.5)
        fmodel.set_param_hint(VAR_SIGN, value=sign, vary=False)
        fmodel.set_param_hint(VAR_PRE_OI, value=pre_oi, min=pre_oi - abs(pre_oi) * 0.5, max=pre_oi + abs(pre_oi) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_OI, value=shift_oi, min=shift_oi - SHIFT_LIM, max=shift_oi + SHIFT_LIM)
        fmodel.set_param_hint(VAR_PRE_OF, value=pre_of, min=pre_of - abs(pre_of) * 0.5, max=pre_of + abs(pre_of) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_OF, value=shift_of, min=shift_of - SHIFT_LIM, max=shift_of + SHIFT_LIM)
        fmodel.set_param_hint(VAR_PRE_II, value=pre_ii, min=pre_ii - abs(pre_ii) * 0.5, max=pre_ii + abs(pre_ii) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_II, value=shift_ii, min=shift_ii - SHIFT_LIM, max=shift_ii + SHIFT_LIM)
        fmodel.set_param_hint(VAR_PRE_IF, value=pre_if, min=pre_if - abs(pre_if) * 0.5, max=pre_if + abs(pre_if) * 0.5)
        fmodel.set_param_hint(VAR_SHIFT_IF, value=shift_if, min=shift_if - SHIFT_LIM, max=shift_if + SHIFT_LIM)
    else:
        if scan == SCAN_OI:
            fmodel.set_param_hint(VAR_PRE_OI, value=pre_oi, min=pre_oi - abs(pre_oi) * 0.5,
                                  max=pre_oi + abs(pre_oi) * 0.5)
            fmodel.set_param_hint(VAR_SHIFT_OI, value=shift_oi, min=shift_oi - SHIFT_LIM, max=shift_oi + SHIFT_LIM)
        else:
            fmodel.set_param_hint(VAR_C_OI, value=np.round(np.mean(data_oi), 1), vary=False)
            fmodel.set_param_hint(VAR_PRE_OI, value=pre_oi, vary=False)
            fmodel.set_param_hint(VAR_SHIFT_OI, value=shift_oi, vary=False)
        if scan == SCAN_OF:
            # print(fmodel.independent_vars)
            fmodel.set_param_hint(VAR_PRE_OF, value=pre_of, min=pre_of - abs(pre_of) * 0.5,
                                  max=pre_of + abs(pre_of) * 0.5)
            fmodel.set_param_hint(VAR_SHIFT_OF, value=shift_of, min=shift_of - SHIFT_LIM, max=shift_of + SHIFT_LIM)
        else:
            fmodel.set_param_hint(VAR_C_OF, value=np.round(np.mean(data_of), 1), vary=False)
            fmodel.set_param_hint(VAR_PRE_OF, value=pre_of, vary=False)
            fmodel.set_param_hint(VAR_SHIFT_OF, value=shift_of, vary=False)
        if scan == SCAN_II:
            fmodel.set_param_hint(VAR_PRE_II, value=pre_ii, min=pre_ii - abs(pre_ii) * 0.5,
                                  max=pre_ii + abs(pre_ii) * 0.5)
            fmodel.set_param_hint(VAR_SHIFT_II, value=shift_ii, min=shift_ii - SHIFT_LIM, max=shift_ii + SHIFT_LIM)
        else:
            fmodel.set_param_hint(VAR_C_II, value=np.round(np.mean(data_ii), 1), vary=False)
            fmodel.set_param_hint(VAR_PRE_II, value=pre_ii, vary=False)
            fmodel.set_param_hint(VAR_SHIFT_II, value=shift_ii, vary=False)
        if scan == SCAN_IF:
            fmodel.set_param_hint(VAR_PRE_IF, value=pre_if, min=pre_if - abs(pre_if) * 0.5,
                                  max=pre_if + abs(pre_if) * 0.5)
            fmodel.set_param_hint(VAR_SHIFT_IF, value=shift_if, min=shift_if - SHIFT_LIM,
                                  max=shift_if + SHIFT_LIM)
        else:
            fmodel.set_param_hint(VAR_C_IF, value=np.round(np.mean(data_if), 1), vary=False)
            fmodel.set_param_hint(VAR_PRE_IF, value=pre_if, vary=False)
            fmodel.set_param_hint(VAR_SHIFT_IF, value=shift_if, vary=False)

    params = fmodel.make_params()
    # print(data_oi, data_of, data_ii, data_if, data_obj)
    # print(fmodel.independent_vars)
    print(params)
    if scan == SCAN_OI:
        result = fmodel.fit(data_obj, params, c_oi=data_oi)
        print(result.fit_report())
        return result.params[VAR_PRE_OI].value, result.params[VAR_SHIFT_OI].value, result.params[VAR_P_INIT].value
    elif scan == SCAN_OF:
        result = fmodel.fit(data_obj, params, c_of=data_of)
        print(result.fit_report())
        return result.params[VAR_PRE_OI].value, result.params[VAR_PRE_OF].value, result.params[VAR_SHIFT_OI].value, \
               result.params[VAR_SHIFT_OF].value, result.params[VAR_P_INIT].value
    elif scan == SCAN_II:
        result = fmodel.fit(data_obj, params, c_ii=data_ii)
        print(result.fit_report())
        return result.params[VAR_PRE_OI].value, result.params[VAR_PRE_OF].value, result.params[VAR_PRE_II].value, \
               result.params[VAR_SHIFT_II].value, result.params[VAR_SHIFT_OI].value, result.params[VAR_SHIFT_OF].value, \
               result.params[VAR_P_INIT].value
    elif scan == SCAN_IF:
        result = fmodel.fit(data_obj, params, c_if=data_if)
        print(result.fit_report())
        return result.params[VAR_PRE_OI].value, result.params[VAR_PRE_OF].value, result.params[VAR_PRE_II].value, \
               result.params[VAR_PRE_IF].value, result.params[VAR_SHIFT_II].value, result.params[VAR_SHIFT_OI].value, \
               result.params[VAR_SHIFT_OF].value, result.params[VAR_SHIFT_IF].value, result.params[VAR_P_INIT].value
    elif scan == SCAN_FIT:
        result = fmodel.fit(data_obj, params, c_oi=data_oi, c_of=data_of, c_ii=data_ii, c_if=data_if, weights=monitor)
        print(result.fit_report())
        return result.params[VAR_PRE_OI].value, result.params[VAR_PRE_OF].value, result.params[VAR_PRE_II].value, \
               result.params[VAR_PRE_IF].value, result.params[VAR_SHIFT_II].value, result.params[VAR_SHIFT_OI].value, \
               result.params[VAR_SHIFT_OF].value, result.params[VAR_SHIFT_IF].value, result.params[VAR_P_INIT].value
    else:
        result = fmodel.fit(data_obj, params, c_oi=data_oi, c_of=data_of, c_ii=data_ii, c_if=data_if, weights=monitor)
        print(result.fit_report())
        return result.params[VAR_AMP].value, result.params[VAR_PRE_OI].value, result.params[VAR_PRE_OF].value, \
               result.params[VAR_PRE_II].value, result.params[VAR_PRE_IF].value, result.params[VAR_SHIFT_OI].value, \
               result.params[VAR_SHIFT_OF].value, result.params[VAR_SHIFT_II].value, result.params[VAR_SHIFT_IF].value, \
               result.params[VAR_P_INIT].value


def current_adjust(coil, period):
    if coil < 4 - period:
        coil += period
    elif coil > period:
        coil -= period
    else:
        pass
    return coil


def plot_1d2(meas: MNSI_NUC, numbers_p, numbers_m, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii,
             shift_if, p_init, amp_nsf, pre_oi_nsf, pre_of_nsf, pre_ii_nsf, pre_if_nsf, shift_oi_nsf, shift_of_nsf,
             shift_ii_nsf, shift_if_nsf, p_init_nsf, amp_sf, pre_oi_sf, pre_of_sf, pre_ii_sf, pre_if_sf, shift_oi_sf,
             shift_of_sf, shift_ii_sf, shift_if_sf, p_init_sf):
    for numor, scan_number_p in enumerate(numbers_p):
        data_file_p = RawData(scan_number_p)
        coils = data_file_p.scan_data
        if coils.ndim == 1:
            coil_4d = np.zeros((4, coils.shape[0]))
            index = univ.positions.index(data_file_p.scan_posn)
            coil_4d[index] = coils
        else:
            coil_4d = coils
        counts_p = data_file_p.scan_count
        scan_number_m = numbers_m[numor]
        data_file_m = RawData(scan_number_m)
        counts_m = data_file_m.scan_count
        pols = univ.count2pol(counts_p, counts_m)

        c_oi, c_of, c_ii, c_if = coil_4d[0], coil_4d[1], coil_4d[2], coil_4d[3]
        coil_plot = np.linspace(-3, 3, 61)
        if data_file_p.scan_posn == SCAN_OI:
            c_oi = coil_plot
            c_of = np.round(np.mean(c_of), 1)
            c_ii = np.round(np.mean(c_ii), 1)
            c_if = np.round(np.mean(c_if), 1)
            plt_ttl = "{}: {:.1f} A, {}: {:.1f} A, {}: {:.1f} A".format(SCAN_OF, c_of, SCAN_II, c_ii, SCAN_IF, c_if)
        elif data_file_p.scan_posn == SCAN_OF:
            c_oi = np.round(np.mean(c_oi), 1)
            c_of = coil_plot
            c_ii = np.round(np.mean(c_ii), 1)
            c_if = np.round(np.mean(c_if), 1)
            plt_ttl = "{}: {:.1f} A, {}: {:.1f} A, {}: {:.1f} A".format(SCAN_OI, c_oi, SCAN_II, c_ii, SCAN_IF, c_if)
        elif data_file_p.scan_posn == SCAN_II:
            c_oi = np.round(np.mean(c_oi), 1)
            c_of = np.round(np.mean(c_of), 1)
            c_ii = coil_plot
            c_if = np.round(np.mean(c_if), 1)
            plt_ttl = "{}: {:.1f} A, {}: {:.1f} A, {}: {:.1f} A".format(SCAN_OF, c_of, SCAN_OI, c_oi, SCAN_IF, c_if)
        elif data_file_p.scan_posn == SCAN_IF:
            c_oi = np.round(np.mean(c_oi), 1)
            c_of = np.round(np.mean(c_of), 1)
            c_ii = np.round(np.mean(c_ii), 1)
            c_if = coil_plot
            plt_ttl = "{}: {:.1f} A, {}: {:.1f} A, {}: {:.1f} A".format(SCAN_OF, c_of, SCAN_II, c_ii, SCAN_II, c_ii)
        else:
            raise ValueError("Invalid scan name given.")

        fit_pol = coil2pol(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii,
                           shift_if, meas.delta_i, meas.delta_f, p_init)
        fit_nsf = coil2count(amp_nsf, SIGN_NSF, c_oi, c_of, c_ii, c_if, pre_oi_nsf, pre_of_nsf, pre_ii_nsf, pre_if_nsf,
                             shift_oi_nsf, shift_of_nsf, shift_ii_nsf, shift_if_nsf, meas.delta_i, meas.delta_f,
                             p_init_nsf)
        fit_sf = coil2count(amp_sf, SIGN_SF, c_oi, c_of, c_ii, c_if, pre_oi_sf, pre_of_sf, pre_ii_sf, pre_if_sf,
                            shift_oi_sf, shift_of_sf, shift_ii_sf, shift_if_sf, meas.delta_i, meas.delta_f, p_init_sf)
        filename = "Scan{:d}_{:d}.png".format(scan_number_p, scan_number_m)
        filename = "/".join([FOLDER_1D_MNSI, filename])
        fig, ax = plt.subplots(figsize=(15, 10))
        sort_ind = np.argsort(data_file_p.scan_x)
        scan_x = data_file_p.scan_x[sort_ind]
        counts_p = counts_p[sort_ind]
        counts_m = counts_m[sort_ind]
        pols = pols[sort_ind]
        ax.errorbar(scan_x, counts_p, yerr=np.sqrt(counts_p), color=COLOUR_NSF, fmt="o", label="Meas NSF")
        ax.errorbar(scan_x, counts_m, yerr=np.sqrt(counts_m), color=COLOUR_SF, fmt="o", label="Meas SF")
        ax.plot(coil_plot, fit_nsf, color=COLOUR_NSF, label="Fit NSF")
        ax.plot(coil_plot, fit_sf, color=COLOUR_SF, label="Fit SF")
        ax.set_xlabel("{} (A)".format(data_file_p.scan_posn))
        ax.set_ylabel("Counts")
        ax.set_title(plt_ttl)
        ax.tick_params(axis="both", direction="in")
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        colour_ax2 = "green"
        ax2.plot(scan_x, pols, "o", color=colour_ax2, label="Pol Meas")
        ax2.plot(coil_plot, fit_pol, color=colour_ax2, label="Pol Fit")
        ax2.tick_params(axis="y", direction="in")
        ax2.set_ylabel(r"Polarisation", color=colour_ax2)
        ax2.tick_params(axis='y', color=colour_ax2, labelcolor=colour_ax2)
        ax.legend(loc=2)
        ax2.legend(loc=1)
        ax2.set_ylim(-1, 1)
        fig.savefig(filename, bbox_inches='tight')
        # plt.show()
        plt.close(fig)


def plot_4d(meas, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init):
    filename = "OuterCoils.png"
    fig, ax = plt.subplots(figsize=(15, 10))  # figsize=(10, 5)
    c_oi = np.linspace(-3, 3, 61)
    c_of = np.linspace(-3, 3, 61)
    c_oi, c_of = np.meshgrid(c_oi, c_of)
    c_ii = phase2current(precession_phase=0.0, prefactor=pre_ii, shift=shift_ii)
    c_if = phase2current(precession_phase=0.0, prefactor=pre_if, shift=shift_if)
    print("II = {:.1f},IF = {:.1f}".format(c_ii, c_if))
    pols = coil2pol(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
                    meas.delta_i, meas.delta_f, p_init)
    cnt = ax.contourf(c_oi, c_of, pols)
    ax.set_xlabel(r"$I$ (A): {:s}".format(SCAN_OI))
    ax.set_ylabel(r"$I$ (A): {:s}".format(SCAN_OF))
    cbar = fig.colorbar(cnt)
    ax.tick_params(axis="both", direction="in")
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

    filename = "InnerCoils.png"
    fig, ax = plt.subplots(figsize=(15, 10))  # figsize=(10, 5)
    c_oi = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_oi, shift=shift_oi)
    c_of = phase2current(precession_phase=-np.pi / 2.0, prefactor=pre_of, shift=shift_of)
    print("OI = {:.1f}, OF = {:.1f}".format(c_oi, c_of))
    c_ii = np.linspace(-3, 3, 61)
    c_if = c_ii
    c_ii, c_if = np.meshgrid(c_ii, c_if)
    pols = coil2pol(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
                    meas.delta_i, meas.delta_f, p_init)
    cnt = ax.contourf(c_ii, c_if, pols)
    ax.set_xlabel(r"$I$ (A): {:s}".format(SCAN_II))
    ax.set_ylabel(r"$I$ (A): {:s}".format(SCAN_IF))
    cbar = fig.colorbar(cnt)
    ax.tick_params(axis="both", direction="in")
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def coils_fitting(meas, scan, pre_oi=None, pre_of=None, pre_ii=None, pre_if=None, shift_oi=None, shift_of=None,
                  shift_ii=None, shift_if=None, p_init=None):
    print("Scan in process: {}".format(scan))
    if scan == SCAN_OI:
        numbers_p = meas.scan_oi_p
        numbers_m = meas.scan_oi_m
    elif scan == SCAN_OF:
        numbers_p = meas.scan_of_p
        numbers_m = meas.scan_of_m
    elif scan == SCAN_II:
        numbers_p = meas.scan_ii_p
        numbers_m = meas.scan_ii_m
    elif scan == SCAN_IF:
        numbers_p = meas.scan_if_p
        numbers_m = meas.scan_if_m
    elif scan == SCAN_FIT:
        numbers_p = meas.scan_fit_p
        numbers_m = meas.scan_fit_m
    elif scan == SCAN_COUNT:
        numbers_p = meas.scan_all_p
        numbers_m = meas.scan_all_m
        # print(numbers_p, "\n", numbers_m)
        if len(numbers_p) != len(numbers_m):
            raise RuntimeError("Different numbers of files with NSF and SF\nNSF:{}\nSF:{}".format(numbers_p, numbers_m))
    else:
        raise ValueError("Invalid scan type.")

    counts_p = np.array([], dtype=int)
    counts_m = np.array([], dtype=int)
    scan_coils = np.empty((4, 0))
    scan_no_p = np.array([], dtype=int)
    scan_no_m = np.array([], dtype=int)
    scan_monitor = np.array([], dtype=int)
    for scan_number in numbers_p:
        data_file = RawData(scan_number)
        # print(scan_number, data_file.relevant_scan)
        if data_file.relevant_scan is False:
            print("Scan No. {} is not complete".format(scan_number))
            numbers_p.remove(scan_number)
            continue
        # if scan == SCAN_ALL:
        #     data = data_file.scan_data
        #     if len(data.shape) == 1:
        #         data_4d = np.zeros((4, data.shape[0]))
        #         index = univ.positions.index(data_file.scan_posn)
        #         data_4d[index, :] = data
        #         scan_coils = np.append(scan_coils, data_4d, axis=1)
        #     else:
        #         scan_coils = np.append(scan_coils, data, axis=1)
        # else:
        #     scan_coil = np.append(scan_coil, data_file.scan_x)
        data = data_file.scan_data
        if len(data.shape) == 1:
            data_4d = np.zeros((4, data.shape[0]))
            index = univ.positions.index(data_file.scan_posn)
            data_4d[index, :] = data
            scan_coils = np.append(scan_coils, data_4d, axis=1)
        else:
            scan_coils = np.append(scan_coils, data, axis=1)
        counts_p = np.append(counts_p, data_file.scan_count)
        scan_no_p = np.append(scan_no_p, np.repeat(scan_number, data_file.scan_count.shape[0]))
        scan_monitor = np.append(scan_monitor, np.repeat(data_file.monitor_count, data_file.scan_count.shape[0]))
    for scan_number in numbers_m:
        data_file = RawData(scan_number)
        # print(scan_number, data_file.relevant_scan)
        if data_file.relevant_scan is False:
            print("Scan No. {} is not relevant".format(scan_number))
            numbers_m.remove(scan_number)
            continue
        counts_m = np.append(counts_m, data_file.scan_count)
        scan_no_m = np.append(scan_no_m, np.repeat(scan_number, data_file.scan_count.shape[0]))
    # print(counts_p.shape[0], counts_m.shape[0])
    pols = univ.count2pol(counts_p, counts_m)
    if scan == SCAN_OI:
        # print(scan_coils,pols)VAR_C_OI
        print(VAR_C_OI)
        fmodel_oi = Model(coil2pol, independent_vars=[VAR_C_OI])
        print(fmodel_oi)
        pre_oi, shift_oi, p_init = lmfit_loader(fmodel_oi, meas, scan, scan_monitor, scan_coils[0], scan_coils[1],
                                                scan_coils[2], scan_coils[3], pols)
        # plot_1d2(meas, scan, numbers_p, numbers_m, scan_coil, counts_p, counts_m, pols, pre, shift, p_init)
        return pre_oi, shift_oi, p_init
    elif scan == SCAN_OF:
        fmodel_of = Model(coil2pol, independent_vars=[VAR_C_OF])
        pre_oi, pre_of, shift_oi, shift_of, p_init = lmfit_loader(fmodel_of, meas, scan, scan_monitor, scan_coils[0],
                                                                  scan_coils[1],
                                                                  scan_coils[2], scan_coils[3], pols, pre_oi=pre_oi,
                                                                  shift_oi=shift_oi, p_init=p_init)
        # plot_1d2(meas, scan, numbers_p, numbers_m, scan_coil, counts_p, counts_m, pols, pre, shift, p_init)
        return pre_oi, pre_of, shift_oi, shift_of, p_init
    elif scan == SCAN_II:
        fmodel_ii = Model(coil2pol, independent_vars=[VAR_C_II])
        pre_oi, pre_of, pre_ii, shift_oi, shift_of, shift_ii, p_init = lmfit_loader(fmodel_ii, meas, scan, scan_monitor,
                                                                                    scan_coils[0], scan_coils[1],
                                                                                    scan_coils[2], scan_coils[3], pols,
                                                                                    pre_oi=pre_oi, pre_of=pre_of,
                                                                                    shift_oi=shift_oi,
                                                                                    shift_of=shift_of,
                                                                                    p_init=p_init)
        # plot_1d2(meas, scan, numbers_p, numbers_m, scan_coil, counts_p, counts_m, pols, pre, shift, p_init)
        return pre_oi, pre_of, pre_ii, shift_oi, shift_of, shift_ii, p_init
    elif scan == SCAN_IF:
        fmodel_if = Model(coil2pol, independent_vars=[VAR_C_IF])
        pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init = lmfit_loader(fmodel_if, meas,
                                                                                                      scan,
                                                                                                      scan_monitor,
                                                                                                      scan_coils[0],
                                                                                                      scan_coils[1],
                                                                                                      scan_coils[2],
                                                                                                      scan_coils[3],
                                                                                                      pols,
                                                                                                      pre_oi=pre_oi,
                                                                                                      pre_of=pre_of,
                                                                                                      pre_ii=pre_ii,
                                                                                                      shift_oi=shift_oi,
                                                                                                      shift_of=shift_of,
                                                                                                      shift_ii=shift_ii,
                                                                                                      p_init=p_init)
        # plot_1d2(meas, scan, numbers_p, numbers_m, scan_coil, counts_p, counts_m, pols, pre, shift, p_init)
        return pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init
    elif scan == SCAN_FIT:
        fmodel_fit = Model(coil2pol, independent_vars=[VAR_C_OI, VAR_C_OF, VAR_C_II, VAR_C_IF])
        pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init = lmfit_loader(fmodel_fit, meas,
                                                                                                      scan,
                                                                                                      scan_monitor,
                                                                                                      scan_coils[0],
                                                                                                      scan_coils[1],
                                                                                                      scan_coils[2],
                                                                                                      scan_coils[3],
                                                                                                      pols,
                                                                                                      pre_oi=pre_oi,
                                                                                                      pre_of=pre_of,
                                                                                                      pre_ii=pre_ii,
                                                                                                      shift_oi=shift_oi,
                                                                                                      shift_of=shift_of,
                                                                                                      shift_ii=shift_ii,
                                                                                                      p_init=p_init)
        # print(counts_p, counts_m)
        f_coils = open("CoilCurrents.dat", "w+")  #
        print(len(scan_no_p), scan_coils.shape[1])
        for i in range(scan_coils.shape[1]):
            f_coils.write(
                "{}, {}, {}, {:f}, {:f}, {:f}, {:f}, {:f}, {:.0f}, {:.0f}\n".format(scan_no_p[i], scan_no_m[i],
                                                                                    scan_monitor[i],
                                                                                    scan_coils[0, i],
                                                                                    scan_coils[1, i],
                                                                                    scan_coils[2, i],
                                                                                    scan_coils[3, i], pols[i],
                                                                                    counts_p[i], counts_m[i]))
        f_coils.close()
        return pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init
    elif scan == SCAN_COUNT:
        fmodel_nsf = Model(coil2count, independent_vars=[VAR_C_OI, VAR_C_OF, VAR_C_II, VAR_C_IF])
        amp_p, pre_oi_p, pre_of_p, pre_ii_p, pre_if_p, shift_oi_p, shift_of_p, shift_ii_p, shift_if_p, p_init_p = lmfit_loader(
            fmodel_nsf, meas, scan, scan_monitor, scan_coils[0], scan_coils[1], scan_coils[2], scan_coils[3],
            data_obj=counts_p, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if,
            shift_oi=shift_oi, shift_of=shift_of, shift_ii=shift_ii, shift_if=shift_if, p_init=p_init, sign=SIGN_NSF)
        fmodel_sf = Model(coil2count, independent_vars=[VAR_C_OI, VAR_C_OF, VAR_C_II, VAR_C_IF])
        amp_m, pre_oi_m, pre_of_m, pre_ii_m, pre_if_m, shift_oi_m, shift_of_m, shift_ii_m, shift_if_m, p_init_m = lmfit_loader(
            fmodel_sf, meas, scan, scan_monitor, scan_coils[0], scan_coils[1], scan_coils[2], scan_coils[3],
            data_obj=counts_m, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if,
            shift_oi=shift_oi, shift_of=shift_of, shift_ii=shift_ii, shift_if=shift_if, p_init=p_init, sign=SIGN_SF)
        plot_1d2(meas, numbers_p, numbers_m, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
                 p_init, amp_p, pre_oi_p, pre_of_p, pre_ii_p, pre_if_p, shift_oi_p, shift_of_p, shift_ii_p, shift_if_p,
                 p_init_p, amp_m, pre_oi_m, pre_of_m, pre_ii_m, pre_if_m, shift_oi_m, shift_of_m, shift_ii_m,
                 shift_if_m, p_init_m)
        f_nuc = open("PolMat_Nuc.dat", "w+")
        f_nuc.write("pi, pf, pif, coil_o1 (°), coil_o2 (°), coil_i1 (°), coil_i2 (°),\n")
        for pi in AXES:
            for pf in AXES:
                alpha_i, alpha_f, beta_i, beta_f = polarisation2angles(pi, pf, measure.delta_i, measure.delta_f)
                # print(pi, pf, np.rad2deg(alpha_i), np.rad2deg(alpha_f), np.rad2deg(beta_i), np.rad2deg(beta_f))
                c_oi_mtx = phase2current(precession_phase=alpha_i, prefactor=pre_oi, shift=shift_oi)
                c_of_mtx = phase2current(precession_phase=alpha_f, prefactor=pre_of, shift=shift_of)
                c_ii_mtx = phase2current(precession_phase=beta_i, prefactor=pre_ii, shift=shift_ii)
                c_if_mtx = phase2current(precession_phase=beta_f, prefactor=pre_if, shift=shift_if)
                alpha_i = pre_oi * c_oi_mtx + shift_oi
                alpha_f = pre_of * c_of_mtx + shift_of
                beta_i = pre_ii * c_ii_mtx + shift_ii
                beta_f = pre_if * c_if_mtx + shift_if
                # print(pi, pf, np.rad2deg(alpha_i), np.rad2deg(alpha_f), np.rad2deg(beta_i), np.rad2deg(beta_f))
                nsf = coil2count(amp_p, SIGN_NSF, c_oi=c_oi_mtx, c_of=c_of_mtx, c_ii=c_ii_mtx, c_if=c_if_mtx,
                                 pre_oi=pre_oi_p, pre_of=pre_of_p, pre_ii=pre_ii_p, pre_if=pre_if_p,
                                 shift_oi=shift_oi_p, shift_of=shift_of_p, shift_ii=shift_ii_p, shift_if=shift_if_p,
                                 delta_i=measure.delta_i, delta_f=measure.delta_f, p_init=p_init_p)
                sf = coil2count(amp_m, SIGN_SF, c_oi=c_oi_mtx, c_of=c_of_mtx, c_ii=c_ii_mtx, c_if=c_if_mtx,
                                pre_oi=pre_oi_m, pre_of=pre_of_m, pre_ii=pre_ii_m, pre_if=pre_if_m, shift_oi=shift_oi_m,
                                shift_of=shift_of_m, shift_ii=shift_ii_m, shift_if=shift_if_m, delta_i=measure.delta_i,
                                delta_f=measure.delta_f, p_init=p_init_m)
                pol = univ.count2pol(nsf, sf)
                f_nuc.write(
                    "{:s}, {:s}, {:f}, {:.1f}°, {:.1f}°, {:.1f}°, {:.1f}°\n".format(pi, pf, pol, np.rad2deg(alpha_i),
                                                                                    np.rad2deg(alpha_f),
                                                                                    np.rad2deg(beta_i),
                                                                                    np.rad2deg(beta_f)))
        f_nuc.close()


measure = MNSI_NUC()

f = open(univ.fname_parameters, "w+")
f.write("Pre-factor outer incoming, pre-factor outer final, shift outer incoming, shift outer final\n")
pre_oi, shift_oi, p_init = coils_fitting(meas=measure, scan=SCAN_OI)
c_oi = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_oi, shift=shift_oi)
print("Outer coil upstream {:.3f} A".format(c_oi))

pre_oi, pre_of, shift_oi, shift_of, p_init = coils_fitting(meas=measure, scan=SCAN_OF, pre_oi=pre_oi, shift_oi=shift_oi,
                                                           p_init=p_init)
c_of = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_of, shift=shift_of)
print("Outer coil downstream {:.3f} A".format(c_of))
print(pre_oi, pre_of, shift_oi, shift_of)

pre_oi, pre_of, pre_ii, shift_oi, shift_of, shift_ii, p_init = coils_fitting(meas=measure, scan=SCAN_II, pre_oi=pre_oi,
                                                                             pre_of=pre_of, shift_oi=shift_oi,
                                                                             shift_of=shift_of, p_init=p_init)
c_ii = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_ii, shift=shift_ii)
pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init = coils_fitting(meas=measure,
                                                                                               scan=SCAN_IF,
                                                                                               pre_oi=pre_oi,
                                                                                               pre_of=pre_of,
                                                                                               pre_ii=pre_ii,
                                                                                               shift_oi=shift_oi,
                                                                                               shift_of=shift_of,
                                                                                               shift_ii=shift_ii,
                                                                                               p_init=p_init)
c_if = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_if, shift=shift_if)
print("Inner coil upstream {:.3f} A".format(c_ii))
f.write("{:f}, {:f}, {:f}, {:f}\n".format(pre_oi, pre_of, shift_oi, shift_of))
f.write("{:f}, {:f}, {:f}, {:f}\n".format(pre_ii, pre_if, shift_ii, shift_if))
f.close()

pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init = coils_fitting(meas=measure,
                                                                                               scan=SCAN_FIT,
                                                                                               pre_oi=pre_oi,
                                                                                               pre_of=pre_of,
                                                                                               pre_ii=pre_ii,
                                                                                               pre_if=pre_if,
                                                                                               shift_oi=shift_oi,
                                                                                               shift_of=shift_of,
                                                                                               shift_ii=shift_ii,
                                                                                               shift_if=shift_if,
                                                                                               p_init=p_init)

coils_fitting(meas=measure, scan=SCAN_COUNT, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii, pre_if=pre_if,
              shift_oi=shift_oi, shift_of=shift_of, shift_ii=shift_ii, shift_if=shift_if, p_init=p_init)
# f_nuc = open("PolMat_Nuc.dat", "w+")
# f_nuc.write("pi, pf, pif, coil_o1 (°), coil_o2 (°), coil_i1 (°), coil_i2 (°),\n")
# for pi in AXES:
#     for pf in AXES:
#         alpha_i, alpha_f, beta_i, beta_f = polarisation2angles(pi, pf, measure.delta_i, measure.delta_f)
#         print(pi, pf, np.rad2deg(alpha_i), np.rad2deg(alpha_f), np.rad2deg(beta_i), np.rad2deg(beta_f))
#         c_oi = phase2current(precession_phase=alpha_i, prefactor=pre_oi, shift=shift_oi)
#         c_of = phase2current(precession_phase=alpha_f, prefactor=pre_of, shift=shift_of)
#         c_ii = phase2current(precession_phase=beta_i, prefactor=pre_ii, shift=shift_ii)
#         c_if = phase2current(precession_phase=beta_f, prefactor=pre_if, shift=shift_if)
#         alpha_i = pre_oi * c_oi + shift_oi
#         alpha_f = pre_of * c_of + shift_of
#         beta_i = pre_ii * c_ii + shift_ii
#         beta_f = pre_if * c_if + shift_if
#         print(pi, pf, np.rad2deg(alpha_i), np.rad2deg(alpha_f), np.rad2deg(beta_i), np.rad2deg(beta_f))
#         pol = coil_system(c_oi=c_oi, c_of=c_of, c_ii=c_ii, c_if=c_if, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii,
#                           pre_if=pre_if, shift_oi=shift_oi, shift_of=shift_of, shift_ii=shift_ii, shift_if=shift_if,
#                           delta_i=measure.delta_i, delta_f=measure.delta_f, p_init=p_init)
#         f_nuc.write("{:s}, {:s}, {:f}, {:.1f}°, {:.1f}°, {:.1f}°, {:.1f}°\n".format(pi, pf, pol, np.rad2deg(alpha_i),
#                                                                                     np.rad2deg(alpha_f),
#                                                                                     np.rad2deg(beta_i),
#                                                                                     np.rad2deg(beta_f)))
# f_nuc.close()
