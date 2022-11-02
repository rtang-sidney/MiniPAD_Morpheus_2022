import numpy as np
from raw_data import RawData
import matplotlib.pyplot as plt
from lmfit import Model
import universal_params as univ

# import sys

plt.rcParams.update({'font.size': 18})
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Arial']
# np.set_printoptions(threshold=sys.maxsize)

SAMPLE_PG = "PG"
SAMPLE_MNSI = "MnSi"
SAMPLES = [SAMPLE_PG, SAMPLE_MNSI]

FOLDER_1D_PG = "Scan_1D_PG"
FOLDER_1D_MNSI = "Scan_1D_MnSi"
FOLDERS_SCAN_1D = [FOLDER_1D_PG, FOLDER_1D_MNSI]

FOLDER_SCAN_2D = "Scan_2D"

SCAN_OUTER = "Outer"
SCAN_INNER = "Inner"

VAR_C_I = "c_i"
VAR_C_F = "c_f"
VAR_PRE_I = 'pre_i'
VAR_PRE_F = 'pre_f'
VAR_SHIFT_I = 'shift_i'
VAR_SHIFT_F = 'shift_f'
VAR_DELTA_I = 'delta_i'
VAR_DELTA_F = 'delta_f'
VAR_P_INIT = 'p_init'


class Parameters:
    def __init__(self, sample_name):
        self.sample = sample_name
        if self.sample == SAMPLE_PG:
            self.scan_outer_p = [2808, 2809]
            self.scan_outer_m = [2808, 2809]
            self.scan_inner_p = list(range(2814, 2854 + 1))
            self.scan_inner_m = list(range(2814, 2854 + 1))
            self.d_spacing = univ.d_pg002
        elif self.sample == SAMPLE_MNSI:
            self.scan_outer_p = [2223]
            self.scan_outer_m = [2224]
            self.scan_inner_p = list(range(3079, 3086 + 1))
            self.scan_inner_m = list(range(2814, 2854 + 1))
            self.d_spacing = univ.d_mnsi011
        else:
            raise ValueError("Invalid sample name given. {}".format(sample_name))
        self.theta = np.arcsin(univ.wavelength / (2 * self.d_spacing))
        self.delta_i = self.theta
        self.delta_f = np.pi - self.theta

        self.period_i = 6.8
        self.period_f = 1.498 * 4
        self.frequency_i = 2 * np.pi / self.period_i
        self.frequency_f = 2 * np.pi / self.period_f


def coil_outer(c_i, c_f, pre_i, pre_f, shift_i, shift_f, delta_i, delta_f, p_init):
    alpha_i = pre_i * c_i + shift_i
    alpha_f = pre_f * c_f + shift_f
    pol = np.cos(alpha_i) * np.cos(alpha_f) - np.sin(alpha_i) * np.sin(alpha_f) * np.cos(delta_i - delta_f)
    pol *= p_init
    return pol


def coil_inner(c_i, c_f, pre_i, pre_f, shift_i, shift_f, delta_i, delta_f, p_init):
    beta_i = pre_i * c_i + shift_i
    beta_f = pre_f * c_f + shift_f
    pol = -np.cos(beta_i + beta_f + delta_i - delta_f)
    pol *= p_init
    return pol


def phase2current(precession_phase, prefactor, shift):
    if abs(precession_phase) < 1e-4:
        return 0
    current = (precession_phase - shift) / prefactor
    return current


def lmfit_2d(meas, scan, data_i, data_f, data_pol):
    if scan == SCAN_OUTER:
        fmodel = Model(coil_outer, independent_vars=[VAR_C_I, VAR_C_F])
    else:
        fmodel = Model(coil_inner, independent_vars=[VAR_C_I, VAR_C_F])
    fmodel.set_param_hint(VAR_PRE_I, value=meas.frequency_oi, min=meas.frequency_oi * 0.9,
                          max=meas.frequency_oi * 1.1)
    fmodel.set_param_hint(VAR_PRE_F, value=meas.frequency_of, min=meas.frequency_of * 0.9,
                          max=meas.frequency_of * 1.1)
    fmodel.set_param_hint(VAR_SHIFT_I, value=0.1 * np.pi, min=-0.3 * np.pi, max=0.3 * np.pi)
    fmodel.set_param_hint(VAR_SHIFT_F, value=0.1 * np.pi, min=-0.3 * np.pi, max=0.3 * np.pi)
    fmodel.set_param_hint(VAR_DELTA_I, value=meas.delta_i, vary=False)
    fmodel.set_param_hint(VAR_DELTA_F, value=meas.delta_f, vary=False)
    fmodel.set_param_hint(VAR_P_INIT, value=0.7, min=0.5, max=1.0)

    params = fmodel.make_params()
    result = fmodel.fit(data_pol, params, c_i=data_i, c_f=data_f)
    print(result.fit_report())
    return result.params[VAR_PRE_I].value, result.params[VAR_PRE_F].value, result.params[VAR_SHIFT_I].value, \
           result.params[VAR_SHIFT_F].value, result.params[VAR_P_INIT].value


def current_adjust(coil, period):
    if coil < 4 - period:
        coil += period
    elif coil > period:
        coil -= period
    else:
        pass
    return coil


def plot_1d(meas, scan, numbers_p, numbers_m, pre_i, pre_f, shift_i, shift_f, p_init):
    scan_numbers_p = numbers_p
    scan_numbers_m = numbers_m

    for numor, scan_number_p in enumerate(scan_numbers_p):
        data_file_p = RawData(scan_number_p)
        scan_number_m = scan_numbers_m[numor]
        c_i = 0
        c_f = 0
        coils = [c_i, c_f]
        if data_file_p.scan_coil in [univ.oi_posn, univ.ii_posn]:
            coils[0] = np.linspace(-3, 3, 61)
            coils[1] = data_file_p.coils_currents[data_file_p.pair_coil]
            coil_plot = coils[0]
        elif data_file_p.scan_coil in [univ.of_posn, univ.if_posn]:
            coils[1] = np.linspace(-3, 3, 61)
            coils[0] = data_file_p.coils_currents[data_file_p.pair_coil]
            coil_plot = coils[1]
        else:
            raise ValueError("Invalid scan coil {}".format(data_file_p.scan_coil))
        c_i, c_f = coils
        scan_coil = data_file_p.scan_x
        scan_counts_p = data_file_p.scan_count
        data_file_m = RawData(scan_number_m)
        scan_counts_m = data_file_m.scan_count
        scan_pol = np.array(
            list(map(lambda k: univ.count2pol(scan_counts_p[k], scan_counts_m[k]), range(scan_counts_p.shape[0]))))
        scan_err = np.array(
            list(map(lambda k: univ.err_count2pol(scan_counts_p[k], scan_counts_m[k]), range(scan_counts_p.shape[0]))))

        if scan == SCAN_OUTER:
            fit_pol = coil_outer(c_i, c_f, pre_i, pre_f, shift_i, shift_f, meas.delta_i, meas.delta_f, p_init)
        elif scan == SCAN_INNER:
            fit_pol = coil_inner(c_i, c_f, pre_i, pre_f, shift_i, shift_f, meas.delta_i, meas.delta_f, p_init)
        else:
            raise ValueError("Invalid scan name given.")
        filename = "Scan{:d}_{:d}.png".format(scan_number_p, scan_number_m)
        filename = "/".join([FOLDERS_SCAN_1D[1], filename])
        fig, ax = plt.subplots(figsize=(4.5, 3))  # figsize=(10, 5)
        ax.errorbar(scan_coil, scan_pol, yerr=scan_err)  # , fmt="o"
        ax.plot(coil_plot, fit_pol)
        ax.set_xlabel("{} (A)".format(data_file_p.scan_coil))
        ax.set_ylabel("Counts")
        ax.set_ylim(-1, 1)
        text_pairing_coil = "{}: {:.1f} A ".format(data_file_p.pair_coil,
                                                   data_file_p.coils_currents[data_file_p.pair_coil])
        ax.set_title(text_pairing_coil)
        ax.tick_params(axis="both", direction="in")
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)


def coils_fitting(meas, scan):
    c_i = np.array([])
    c_f = np.array([])
    coils = [c_i, c_f]  # it has to have the sequence as the channel list in univ
    if scan == SCAN_OUTER:
        numbers_p = meas.scan_oi_p
        numbers_m = meas.scan_oi_m
    elif scan == SCAN_INNER:
        numbers_p = meas.scan_if_p
        numbers_m = meas.scan_if_m
    else:
        raise ValueError("Invalid scan type.")

    counts_p = np.array([])
    counts_m = np.array([])
    for scan_number in numbers_p:
        data_file = RawData(scan_number)
        if data_file.relevant_scan is False:
            print("Scan No. {} is not complete".format(scan_number))
            numbers_p.remove(scan_number)
            continue
        if data_file.scan_coil in [univ.oi_posn, univ.ii_posn]:
            coils[0] = np.append(coils[0], data_file.scan_x)
        elif data_file.scan_coil in [univ.of_posn, univ.if_posn]:
            coils[1] = np.append(coils[1],
                                 np.repeat(data_file.coils_currents[data_file.pair_coil], data_file.scan_x.shape[0]))
        else:
            continue
        counts_p = np.append(counts_p, data_file.scan_count)
        print(data_file.scan_count)
    for scan_number in numbers_m:
        data_file = RawData(scan_number)
        if data_file.relevant_scan is False:
            print("Scan No. {} is not complete".format(scan_number))
            numbers_m.remove(scan_number)
            continue
        counts_m = np.append(counts_m, data_file.scan_count)
    pols = univ.count2pol(counts_p, counts_m)

    # print(scan_number, data_file.scan_count)

    c_i, c_f = coils
    print(c_i,c_f)
    # print(c_oi, "\n", c_ii, "\n", c_if, "\n", c_of, "\n", counts)
    pre_i, pre_f, shift_i, shift_f, p_init = lmfit_2d(meas=meas, scan=scan, data_i=c_i, data_f=c_f,
                                                      data_pol=pols)

    plot_1d(meas, scan, numbers_p, numbers_m, pre_i, pre_f, shift_i, shift_f, p_init)
    return pre_i, pre_f, shift_i, shift_f


sample = SAMPLE_MNSI
measure = Parameters(sample_name=sample)

f = open(univ.fname_parameters, "w+")
f.write("Pre-factor outer incoming, pre-factor outer final, shift outer incoming, shift outer final\n")
pre_oi, pre_of, shift_oi, shift_of = coils_fitting(meas=measure, scan=SCAN_OUTER)
c_oi = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_oi, shift=shift_oi)
c_of = phase2current(precession_phase=np.pi / 2.0, prefactor=pre_of, shift=shift_of)
print("Outer coil upstream {:.3f} A, outer coil downsteam {:.3f} A".format(c_oi, c_of))
f.write("{.f}, {.f}, {.f}, {.f}\n".format(pre_oi, pre_of, shift_oi, shift_of))

# pre_ii, pre_if, shift_ii, shift_if = coils_fitting(meas=measure, scan=SCAN_INNER)
# f.write("{.f}, {.f}, {.f}, {.f}".format(pre_ii, pre_if, shift_ii, shift_if))
