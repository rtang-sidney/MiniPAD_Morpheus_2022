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
SAMPLE_MNSI = "MnSi"
# SAMPLES = [SAMPLE_PG, SAMPLE_MNSI]

FOLDER_1D_MNSI = "Scan_1D_MnSi"

FOLDER_SCAN_2D = "Scan_2D"

SCAN_OI = univ.oi_posn
SCAN_OF = univ.of_posn
SCAN_II = univ.ii_posn
SCAN_IF = univ.if_posn
SCAN_ALL = "All"

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

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

SHIFT_LIMIT = 0.05 * np.pi


class Parameters:
    def __init__(self, sample_name):
        self.sample = sample_name
        if self.sample == SAMPLE_MNSI:
            self.scan_oi_p = [3173]  # 2223,3173
            self.scan_oi_m = [3174]  # 2224,3174
            self.scan_of_p = [3175]
            self.scan_of_m = [3176]
            self.scan_ii_p = [3179]
            self.scan_ii_m = [3180]
            self.scan_if_p = [3181, 3183]
            self.scan_if_m = [3182, 3184]
            self.d_spacing = univ.d_mnsi011
            # self.scan_all_p = [3173, 3175, 3179, 3181, 3183,3185]
            # self.scan_all_m = [3174, 3174, 3180, 3182, 3184,3186]
            self.scan_all_p = list(np.concatenate((np.arange(3173, 3216 + 1, step=2), np.arange(3218, 3229 + 1, step=2),
                                                   np.arange(3230, 3257 + 1, step=4),
                                                   np.arange(3231, 3257 + 1, step=4))))
            self.scan_all_m = list(np.concatenate((np.arange(3174, 3216 + 1, step=2), np.arange(3219, 3229 + 1, step=2),
                                                   np.arange(3232, 3257 + 1, step=4),
                                                   np.arange(3233, 3257 + 1, step=4))))
            # self.scan_all = list(np.append(np.arange(3173, 3216 + 1), np.arange(3218, 3257 + 1)))
        else:
            raise ValueError("Invalid sample name given. {}".format(sample_name))

        self.theta = np.deg2rad(99.0 / 2.0)
        self.delta_i = np.pi / 2.0 - self.theta
        self.delta_f = np.pi / 2.0 + self.theta

        self.period_oi = 8.0
        self.period_of = 6.0
        self.period_ii = 2.3 * 2.0
        self.period_if = 7.0
        # self.frequency_oi = 2 * np.pi / self.period_oi
        # self.frequency_of = 2 * np.pi / self.period_of
        # self.frequency_ii = 2 * np.pi / self.period_ii
        # self.frequency_if = 2 * np.pi / self.period_if
        self.frequency_oi = 0.736418
        self.frequency_of = 0.737330
        self.frequency_ii = 1.424906
        self.frequency_if = 1.031450
        self.shift_oi = -0.007004
        self.shift_of = -0.107698
        self.shift_ii = -0.389777
        self.shift_if = -0.377453


def coil_system(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, delta_i,
                delta_f, p_init):
    alpha_i = pre_oi * c_oi + shift_oi
    alpha_f = pre_of * c_of + shift_of
    beta_i = pre_ii * c_ii + shift_ii
    beta_f = pre_if * c_if + shift_if
    pol = np.cos(alpha_i) * np.cos(alpha_f) - np.sin(alpha_i) * np.sin(alpha_f) * np.cos(
        beta_i + beta_f + delta_i - delta_f)
    pol *= p_init
    return pol


def cosine_simple(current, scale, pre, phase, y0):
    return scale * np.cos(pre * current - phase) + y0


def phase2current(precession_phase, prefactor, shift):
    # if abs(precession_phase) < 1e-4:
    #     return 0
    current = (precession_phase - shift) / prefactor
    return current


def lmfit_2d(meas, scan, data_c, data_pol):
    print(data_pol)
    c_oi = 0
    c_of = 0
    c_ii = 0
    c_if = 0
    if scan == SCAN_OI:
        fmodel = Model(coil_system, independent_vars=[VAR_C_OI])
        fmodel.set_param_hint(VAR_C_OF, value=0, vary=False)
        fmodel.set_param_hint(VAR_C_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_C_IF, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_OI, value=meas.frequency_oi, min=meas.frequency_oi * 0.5,
                              max=meas.frequency_oi * 1.5)
        fmodel.set_param_hint(VAR_PRE_OF, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_IF, value=0, vary=False)
        c_oi = data_c
        fmodel.set_param_hint(VAR_SHIFT_OI, value=0, min=-SHIFT_LIMIT * 10, max=SHIFT_LIMIT * 10)
        fmodel.set_param_hint(VAR_SHIFT_OF, value=0, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_IF, value=0, vary=False)
    elif scan == SCAN_OF:
        fmodel = Model(coil_system, independent_vars=[VAR_C_OF])
        fmodel.set_param_hint(VAR_C_OI, value=0, vary=False)
        fmodel.set_param_hint(VAR_C_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_C_IF, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_OI, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_OF, value=-meas.frequency_of, max=-meas.frequency_of * 0.5,
                              min=-meas.frequency_of * 1.5)
        fmodel.set_param_hint(VAR_PRE_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_IF, value=0, vary=False)
        c_of = data_c
        fmodel.set_param_hint(VAR_SHIFT_OI, value=0, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_OF, value=0, min=-SHIFT_LIMIT * 10, max=SHIFT_LIMIT * 10)
        fmodel.set_param_hint(VAR_SHIFT_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_IF, value=0, vary=False)
    elif scan == SCAN_II:
        fmodel = Model(coil_system, independent_vars=[VAR_C_II])
        fmodel.set_param_hint(VAR_C_OI, value=2.05, vary=False)
        fmodel.set_param_hint(VAR_C_OF, value=1.75, vary=False)
        fmodel.set_param_hint(VAR_C_IF, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_OI, value=2.0 * np.pi / (4 * 2.05), vary=False)
        fmodel.set_param_hint(VAR_PRE_OF, value=-2.0 * np.pi / (4 * 1.75), vary=False)
        # print("Now it is ii", meas.frequency_ii)
        fmodel.set_param_hint(VAR_PRE_II, value=meas.frequency_ii, min=meas.frequency_ii * 0.5,
                              max=meas.frequency_ii * 1.5)
        fmodel.set_param_hint(VAR_PRE_IF, value=0, vary=False)
        c_ii = data_c
        fmodel.set_param_hint(VAR_SHIFT_OI, value=meas.shift_oi, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_OF, value=meas.shift_of, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_II, value=0, min=-SHIFT_LIMIT * 20, max=SHIFT_LIMIT * 20)
        fmodel.set_param_hint(VAR_SHIFT_IF, value=0, vary=False)
    else:
        fmodel = Model(coil_system, independent_vars=[VAR_C_IF])
        fmodel.set_param_hint(VAR_C_OI, value=2.05, vary=False)
        fmodel.set_param_hint(VAR_C_OF, value=1.75, vary=False)
        fmodel.set_param_hint(VAR_C_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_OI, value=2.0 * np.pi / (4 * 2.05), vary=False)
        fmodel.set_param_hint(VAR_PRE_OF, value=-2.0 * np.pi / (4 * 1.75), vary=False)
        fmodel.set_param_hint(VAR_PRE_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_PRE_IF, value=meas.frequency_if, min=meas.frequency_if * 0.5,
                              max=meas.frequency_if * 1.5)
        c_if = data_c
        fmodel.set_param_hint(VAR_SHIFT_OI, value=meas.shift_oi, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_OF, value=meas.shift_of, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_II, value=0, vary=False)
        fmodel.set_param_hint(VAR_SHIFT_IF, value=0, min=-SHIFT_LIMIT * 20, max=SHIFT_LIMIT * 20)
    fmodel.set_param_hint(VAR_DELTA_I, value=meas.delta_i, vary=False)
    fmodel.set_param_hint(VAR_DELTA_F, value=meas.delta_f, vary=False)
    fmodel.set_param_hint(VAR_P_INIT, value=0.8, min=0.7, max=1.0)

    params = fmodel.make_params()
    if scan == SCAN_OI:
        result = fmodel.fit(data_pol, params, c_oi=c_oi)
        print(result.fit_report())
        return result.params[VAR_PRE_OI].value, result.params[VAR_SHIFT_OI].value, result.params[VAR_P_INIT].value
    elif scan == SCAN_OF:
        result = fmodel.fit(data_pol, params, c_of=c_of)
        print(result.fit_report())
        return result.params[VAR_PRE_OF].value, result.params[VAR_SHIFT_OF].value, result.params[VAR_P_INIT].value
    elif scan == SCAN_II:
        # print("ii", params)
        result = fmodel.fit(data_pol, params, c_ii=c_ii)
        print(result.fit_report())
        return result.params[VAR_PRE_II].value, result.params[VAR_SHIFT_II].value, result.params[VAR_P_INIT].value
    else:
        result = fmodel.fit(data_pol, params, c_if=c_if)
        print(result.fit_report())
        return result.params[VAR_PRE_IF].value, result.params[VAR_SHIFT_IF].value, result.params[VAR_P_INIT].value
    # return result.params[VAR_PRE_OI].value, result.params[].value, result.params[VAR_PRE_II].value, \
    #        result.params[VAR_PRE_IF].value, result.params[VAR_SHIFT_OI].value, result.params[VAR_PRE_OF].value, \
    #        result.params[VAR_PRE_II].value, result.params[VAR_SHIFT_IF].value, result.params[VAR_P_INIT].value


def lm_fit_4d(meas, monitor, data_oi, data_of, data_ii, data_if, data_pol):
    fmodel = Model(coil_system, independent_vars=[VAR_C_OI, VAR_C_OF, VAR_C_II, VAR_C_IF])
    fmodel.set_param_hint(VAR_PRE_OI, value=meas.frequency_oi, min=meas.frequency_oi * 0.8,
                          max=meas.frequency_oi * 1.2)
    fmodel.set_param_hint(VAR_PRE_OF, value=-meas.frequency_of, max=-meas.frequency_of * 0.8,
                          min=-meas.frequency_of * 1.2)
    fmodel.set_param_hint(VAR_PRE_II, value=meas.frequency_ii, min=meas.frequency_ii * 0.8,
                          max=meas.frequency_ii * 1.2)
    fmodel.set_param_hint(VAR_PRE_IF, value=meas.frequency_if, min=meas.frequency_if * 0.8,
                          max=meas.frequency_if * 1.2)
    fmodel.set_param_hint(VAR_SHIFT_OI, value=meas.shift_oi, min=meas.shift_oi - SHIFT_LIMIT,
                          max=meas.shift_oi + SHIFT_LIMIT)
    fmodel.set_param_hint(VAR_SHIFT_OF, value=meas.shift_of, min=meas.shift_of - SHIFT_LIMIT,
                          max=meas.shift_of + SHIFT_LIMIT)
    fmodel.set_param_hint(VAR_SHIFT_II, value=meas.shift_ii, min=meas.shift_ii - SHIFT_LIMIT,
                          max=meas.shift_ii + SHIFT_LIMIT)
    fmodel.set_param_hint(VAR_SHIFT_IF, value=meas.shift_if, min=meas.shift_if - SHIFT_LIMIT,
                          max=meas.shift_if + SHIFT_LIMIT)
    fmodel.set_param_hint(VAR_DELTA_I, value=meas.delta_i, vary=False)
    fmodel.set_param_hint(VAR_DELTA_F, value=meas.delta_f, vary=False)
    fmodel.set_param_hint(VAR_P_INIT, value=0.8, min=0.6, max=1.0)
    params = fmodel.make_params()
    result = fmodel.fit(data_pol, params, c_oi=data_oi, c_of=data_of, c_ii=data_ii, c_if=data_if,
                        weights=monitor)  # / np.sum(monitor)
    # errors = result.eval_uncertainty()
    print(result.fit_report())
    return result.params[VAR_PRE_OI].value, result.params[VAR_PRE_OF].value, result.params[VAR_PRE_II].value, \
           result.params[VAR_PRE_IF].value, result.params[VAR_SHIFT_OI].value, result.params[VAR_SHIFT_OF].value, \
           result.params[VAR_SHIFT_II].value, result.params[VAR_SHIFT_IF].value, result.params[VAR_P_INIT].value


def current_adjust(coil, period):
    if coil < 4 - period:
        coil += period
    elif coil > period:
        coil -= period
    else:
        pass
    return coil


def plot_1d(meas, scan, numbers_p, numbers_m, scan_coil, counts_p, counts_m, pols, pre, shift, p_init):
    scan_numbers_p = numbers_p
    scan_numbers_m = numbers_m

    for numor, scan_number_p in enumerate(scan_numbers_p):
        data_file_p = RawData(scan_number_p)
        scan_number_m = scan_numbers_m[numor]
        coil_plot = np.linspace(-3, 3, 61)
        if scan == SCAN_OI:
            fit_pol = coil_system(coil_plot, 0, 0, 0, pre, 0, 0, 0, shift, 0, 0, 0, meas.delta_i, meas.delta_f, p_init)
        elif scan == SCAN_OF:
            fit_pol = coil_system(0, coil_plot, 0, 0, 0, pre, 0, 0, 0, shift, 0, 0, meas.delta_i, meas.delta_f, p_init)
        elif scan == SCAN_II:
            fit_pol = coil_system(2.05, 1.75, coil_plot, 0, meas.frequency_oi, -meas.frequency_of, pre, 0, 0, 0, shift,
                                  0, meas.delta_i, meas.delta_f, p_init)
            # print(coil_plot, fit_pol, pre, shift)
        elif scan == SCAN_IF:
            # print(scan_coil, scan_counts_p, scan_number_m)
            fit_pol = coil_system(2.05, 1.75, 0, coil_plot, meas.frequency_oi, -meas.frequency_of, 0, pre, 0, 0, 0,
                                  shift, meas.delta_i, meas.delta_f, p_init)
        else:
            raise ValueError("Invalid scan name given.")
        print("Coil: {}, Pol: {}".format(coil_plot, fit_pol))
        filename = "Scan{:d}_{:d}.png".format(scan_number_p, scan_number_m)
        filename = "/".join([FOLDER_1D_MNSI, filename])
        fig, ax = plt.subplots(figsize=(15, 10))  # figsize=(10, 5)
        # ax.errorbar(scan_coil, scan_pol, yerr=scan_err)  # , fmt="o"
        if scan == SCAN_IF:
            sort_ind = np.argsort(scan_coil)
            scan_coil = scan_coil[sort_ind]
            counts_p = counts_p[sort_ind]
            counts_m = counts_m[sort_ind]
            pols = pols[sort_ind]
        ax.errorbar(scan_coil, counts_p, yerr=np.sqrt(counts_p), label="Meas NSF")  # , fmt="o"
        ax.errorbar(scan_coil, counts_m, yerr=np.sqrt(counts_m), label="Meas SF")  # , fmt="o"
        ax.set_xlabel("{} (A)".format(data_file_p.scan_posn))
        ax.set_ylabel("Counts")
        text_pairing_coil = "{}: {:.1f} A ".format(data_file_p.pair_posn,
                                                   data_file_p.coils_currents[data_file_p.pair_posn])
        ax.set_title(text_pairing_coil)
        ax.tick_params(axis="both", direction="in")
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        colour_ax2 = "green"
        ax2.plot(scan_coil, pols, "o", color=colour_ax2, label="Pol Meas")
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


def plot_1d2(meas: Parameters, numbers_p, numbers_m, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii,
             shift_if, p_init):
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

        fit_pol = coil_system(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii,
                              shift_if, meas.delta_i, meas.delta_f, p_init)
        filename = "Scan{:d}_{:d}.png".format(scan_number_p, scan_number_m)
        filename = "/".join([FOLDER_1D_MNSI, filename])
        fig, ax = plt.subplots(figsize=(15, 10))
        sort_ind = np.argsort(data_file_p.scan_x)
        scan_x = data_file_p.scan_x[sort_ind]
        counts_p = counts_p[sort_ind]
        counts_m = counts_m[sort_ind]
        pols = pols[sort_ind]
        ax.errorbar(scan_x, counts_p, yerr=np.sqrt(counts_p), label="Meas NSF")  # , fmt="o"
        ax.errorbar(scan_x, counts_m, yerr=np.sqrt(counts_m), label="Meas SF")  # , fmt="o"
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
    pols = coil_system(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
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
    pols = coil_system(c_oi, c_of, c_ii, c_if, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
                       meas.delta_i, meas.delta_f, p_init)
    cnt = ax.contourf(c_ii, c_if, pols)
    ax.set_xlabel(r"$I$ (A): {:s}".format(SCAN_II))
    ax.set_ylabel(r"$I$ (A): {:s}".format(SCAN_IF))
    cbar = fig.colorbar(cnt)
    ax.tick_params(axis="both", direction="in")
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def coils_fitting(meas, scan):
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
    elif scan == SCAN_ALL:
        numbers_p = meas.scan_all_p
        numbers_m = meas.scan_all_m
        print(numbers_p, "\n", numbers_m)
        if len(numbers_p) != len(numbers_m):
            raise RuntimeError("Different numbers of files with NSF and SF\nNSF:{}\nSF:{}".format(numbers_p, numbers_m))
    else:
        raise ValueError("Invalid scan type.")

    counts_p = np.array([], dtype=int)
    counts_m = np.array([], dtype=int)
    scan_coil = np.array([])
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
        if scan == SCAN_ALL:
            data = data_file.scan_data
            if len(data.shape) == 1:
                data_4d = np.zeros((4, data.shape[0]))
                index = univ.positions.index(data_file.scan_posn)
                data_4d[index, :] = data
                scan_coils = np.append(scan_coils, data_4d, axis=1)
            else:
                scan_coils = np.append(scan_coils, data, axis=1)
        else:
            scan_coil = np.append(scan_coil, data_file.scan_x)
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
    # scan_coil = np.linspace(-2.9, 2.9, num=59)
    if scan == SCAN_ALL:
        # print(scan_coils, "\n", counts_p, "\n", counts_m)
        pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init = lm_fit_4d(meas, scan_monitor,
                                                                                                   scan_coils[0, :],
                                                                                                   scan_coils[1, :],
                                                                                                   scan_coils[2, :],
                                                                                                   scan_coils[3, :],
                                                                                                   pols)
        # print(scan_coil, "\n", pols)
        # print(scan_coils[0, 1], scan_coils[1, 1], scan_coils[2, 1], scan_coils[3, 1], pols[1])
        f = open("CoilCurrents.dat", "w+")  #
        print(len(scan_no_p), scan_coils.shape[1])
        for i in range(scan_coils.shape[1]):
            f.write("{}, {}, {}, {:f}, {:f}, {:f}, {:f}, {:f}\n".format(scan_no_p[i], scan_no_m[i], scan_monitor[i],
                                                                        scan_coils[0, i], scan_coils[1, i],
                                                                        scan_coils[2, i], scan_coils[3, i], pols[i]))
        f.close()

        print(pre_oi, shift_oi)
        plot_4d(meas, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init)
        plot_1d2(meas, numbers_p, numbers_m, pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if,
                 p_init)
        return pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init
    else:
        print(scan_coil, "\n", counts_p, "\n", counts_m)
        pre, shift, p_init = lmfit_2d(meas=meas, scan=scan, data_c=scan_coil, data_pol=pols)
        plot_1d(meas, scan, numbers_p, numbers_m, scan_coil, counts_p, counts_m, pols, pre, shift, p_init)
        return pre, shift


sample = SAMPLE_MNSI
measure = Parameters(sample_name=sample)

f = open(univ.fname_parameters, "w+")
f.write("Pre-factor outer incoming, pre-factor outer final, shift outer incoming, shift outer final\n")
pre_oi, shift_oi = coils_fitting(meas=measure, scan=SCAN_OI)
c_oi = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_oi, shift=shift_oi)
print("Outer coil upstream {:.3f} A".format(c_oi))

pre_of, shift_of = coils_fitting(meas=measure, scan=SCAN_OF)
c_of = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_of, shift=shift_of)
print("Outer coil downstream {:.3f} A".format(c_of))
print(pre_oi, pre_of, shift_oi, shift_of)
f.write("{:f}, {:f}, {:f}, {:f}\n".format(pre_oi, pre_of, shift_oi, shift_of))

pre_ii, shift_ii = coils_fitting(meas=measure, scan=SCAN_II)
c_ii = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_ii, shift=shift_ii)
pre_if, shift_if = coils_fitting(meas=measure, scan=SCAN_IF)
c_if = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_if, shift=shift_if)
print("Inner coil upstream {:.3f} A".format(c_ii))
f.write("{:f}, {:f}, {:f}, {:f}\n".format(pre_ii, pre_if, shift_ii, shift_if))
f.close()

pre_oi, pre_of, pre_ii, pre_if, shift_oi, shift_of, shift_ii, shift_if, p_init = coils_fitting(meas=measure,
                                                                                               scan=SCAN_ALL)

f = open("PolMat_Nuc.dat", "w+")
f.write("pi, pf, pif, coil_o1 (°), coil_o2 (°), coil_i1 (°), coil_i2 (°),\n")
for pi in AXES:
    for pf in AXES:
        alpha_i, alpha_f, beta_i, beta_f = polarisation2angles(pi, pf, measure.delta_i, measure.delta_f)
        print(pi, pf, np.rad2deg(alpha_i), np.rad2deg(alpha_f), np.rad2deg(beta_i), np.rad2deg(beta_f))
        c_oi = phase2current(precession_phase=alpha_i, prefactor=pre_oi, shift=shift_oi)
        c_of = phase2current(precession_phase=alpha_f, prefactor=pre_of, shift=shift_of)
        c_ii = phase2current(precession_phase=beta_i, prefactor=pre_ii, shift=shift_ii)
        c_if = phase2current(precession_phase=beta_f, prefactor=pre_if, shift=shift_if)
        alpha_i = pre_oi * c_oi + shift_oi
        alpha_f = pre_of * c_of + shift_of
        beta_i = pre_ii * c_ii + shift_ii
        beta_f = pre_if * c_if + shift_if
        print(pi, pf, np.rad2deg(alpha_i), np.rad2deg(alpha_f), np.rad2deg(beta_i), np.rad2deg(beta_f))
        pol = coil_system(c_oi=c_oi, c_of=c_of, c_ii=c_ii, c_if=c_if, pre_oi=pre_oi, pre_of=pre_of, pre_ii=pre_ii,
                          pre_if=pre_if, shift_oi=shift_oi, shift_of=shift_of, shift_ii=shift_ii, shift_if=shift_if,
                          delta_i=measure.delta_i, delta_f=measure.delta_f, p_init=p_init)
        f.write("{:s}, {:s}, {:f}, {:.1f}°, {:.1f}°, {:.1f}°, {:.1f}°\n".format(pi, pf, pol, np.rad2deg(alpha_i),
                                                                                np.rad2deg(alpha_f), np.rad2deg(beta_i),
                                                                                np.rad2deg(beta_f)))
f.close()
