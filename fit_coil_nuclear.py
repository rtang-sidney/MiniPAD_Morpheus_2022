import numpy as np
from raw_data import RawData
import matplotlib.pyplot as plt
from lmfit import Model
import universal_params as univ
from matrix_angles import polarisation2angles

# import sys
plt.rcParams.update({'font.size': 18})
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['font.sans-serif'] = ['Arial']
# np.set_printoptions(threshold=sys.maxsize)

SAMPLE_PG = "PG"
SAMPLE_MNSI = "MnSi"
SAMPLES = [SAMPLE_PG, SAMPLE_MNSI]

# SCAN_0_0 = [3054, 3055]
# SCAN_90_90 = range(3071, 3092 + 1)
# SCANS = SCAN_0_0 + list(SCAN_90_90)  # + list(SCAN_0_90) + list(SCAN_90_0)
#
# SCAN_PG_0_0 = [2808, 2809]
# SCAN_PG_90_90 = range(2814, 2854 + 1)
# SCAN_PG_0_90 = range(2855, 2895 + 1)
# SCAN_PG_90_0 = range(2896, 2936 + 1)
# SCANS_PG = SCAN_PG_0_0 + list(SCAN_PG_90_90) + list(SCAN_PG_0_90) + list(SCAN_PG_90_0)
#
# PERIOD_MNSI_OI = 1.2275 * 4
# PERIOD_MNSI_OF = 1.386 * 4
# FREQUENCY_MNSI_OI = 2 * np.pi / PERIOD_MNSI_OI
# FREQUENCY_MNSI_OF = 2 * np.pi / PERIOD_MNSI_OF
#
# PERIOD_PG_OI = 1.2737 * 4
# PERIOD_PG_OF = 1.498 * 4
# FREQUENCY_PG_OI = 2 * np.pi / PERIOD_PG_OI
# FREQUENCY_PG_OF = 2 * np.pi / PERIOD_PG_OF

# D_SPACING_PG = 3.55e-10  # PG 002
# # D_SPACING = 3.55e-10  # PG 002
# d_spacing = 4.556e-10  # MnSi
WAVELENGTH = 4.905921e-10

AXIS_X = "x"
AXIS_Y = "y"
AXIS_Z = "z"
AXES = [AXIS_X, AXIS_Y, AXIS_Z]

FILE_MATRIX_PG = "PolarisationMatrix_PG.dat"
FILE_MATRIX_MNSI = "PolarisationMatrix_MnSi.dat"
FILES_MATRIX = [FILE_MATRIX_PG, FILE_MATRIX_MNSI]

FOLDER_1D_PG = "Scan_1D_PG"
FOLDER_1D_MNSI = "Scan_1D_MnSi"
FOLDERS_SCAN_1D = [FOLDER_1D_PG, FOLDER_1D_MNSI]

FOLDER_SCAN_2D = "Scan_2D"


class Parameters:
    def __init__(self, sample_name):
        self.sample = sample_name
        if self.sample == SAMPLE_PG:
            self.scan_0_0 = [2808, 2809]
            self.scan_90_90 = list(range(2814, 2854 + 1))
            self.scan_0_90 = list(range(2855, 2895 + 1))
            self.scan_90_0 = list(range(2896, 2936 + 1))
            self.scans = self.scan_0_0 + self.scan_90_90 + self.scan_0_90 + self.scan_90_0

            self.period_oi = 1.2737 * 4
            self.period_of = 1.498 * 4
            self.d_spacing = 3.55e-10
        elif self.sample == SAMPLE_MNSI:
            self.scan_0_0 = [3054, 3055, 3087, 3088, 3089]
            self.scan_90_0 = list(range(3071, 3078 + 1))
            self.scan_90_90 = list(range(3079, 3086 + 1))
            self.scan_0_90 = list(range(3090, 3092 + 1))
            self.scans = self.scan_0_0 + self.scan_90_90 + self.scan_0_90 + self.scan_90_0

            self.period_oi = 1.2275 * 4
            self.period_of = 1.386 * 4
            self.d_spacing = 4.556e-10
        else:
            raise ValueError("Invalid sample name given. {}".format(sample_name))

        self.frequency_oi = 2 * np.pi / self.period_oi
        self.frequency_of = 2 * np.pi / self.period_of


def coil_4d(c_oi, c_ii, c_if, c_of, pre_oi, pre_ii, pre_if, pre_of,
            shift_oi, shift_of, shift_i, delta_i, delta_f, scale, shift_y):
    pol = pol_4d(c_oi, c_ii, c_if, c_of, pre_oi, pre_ii, pre_if, pre_of,
                 shift_oi, shift_of, shift_i, delta_i, delta_f, scale)
    return shift_y * (1 + pol)


def pol_4d(c_oi, c_ii, c_if, c_of, pre_oi, pre_ii, pre_if, pre_of, shift_oi,
           shift_of, shift_i, delta_i, delta_f, scale):
    alpha_i = pre_oi * c_oi
    alpha_f = pre_of * c_of
    beta_i = pre_ii * c_ii
    beta_f = pre_if * c_if
    p = np.cos(alpha_i + shift_oi) * np.cos(alpha_f + shift_of) - np.sin(alpha_i + shift_oi) * np.sin(
        alpha_f + shift_of) * np.cos(beta_i + beta_f + delta_i - delta_f + shift_i * 2)
    return scale * p


def phase2current(precession_phase, prefactor, shift):
    if abs(precession_phase) < 1e-4:
        return 0
    current = (precession_phase - shift) / prefactor
    return current


def counts2polarisation(count, shift_y):
    return count / shift_y - 1


def lm_fit_4d(meas, fmodel, data_oi, data_ii, data_if, data_of, data_count, delta_i, delta_f):
    # delta_i and delta_f are the initial and final scattering angles
    fmodel.set_param_hint('pre_oi', value=meas.frequency_oi, min=meas.frequency_oi * 0.9,
                          max=meas.frequency_oi * 1.1)
    fmodel.set_param_hint('pre_ii', value=np.pi / 2.5, min=np.pi / 3.5, max=np.pi / 1.5)
    if meas.sample == SAMPLE_PG:
        fmodel.set_param_hint('pre_if', value=-np.pi / 2.5, min=-np.pi / 2, max=-np.pi / 3.5)
    else:
        fmodel.set_param_hint('pre_if', value=np.pi / 2.5, min=np.pi / 3.5, max=np.pi / 1.5)
    fmodel.set_param_hint('pre_of', value=meas.frequency_of, min=meas.frequency_of * 0.9,
                          max=meas.frequency_of * 1.1)
    fmodel.set_param_hint('shift_oi', value=0.1 * np.pi, min=-0.3 * np.pi, max=0.3 * np.pi)
    fmodel.set_param_hint('shift_of', value=0.1 * np.pi, min=-0.3 * np.pi, max=0.3 * np.pi)
    fmodel.set_param_hint('shift_i', value=0.1 * np.pi, min=-0.3 * np.pi, max=0.3 * np.pi)
    fmodel.set_param_hint('delta_i', value=delta_i, vary=False)
    fmodel.set_param_hint('delta_f', value=delta_f, vary=False)
    fmodel.set_param_hint('scale', value=0.7, min=0.5, max=1.0)
    fmodel.set_param_hint('shift_y', value=(np.max(data_count) + np.min(data_count)) / 2.0, min=np.min(data_count),
                          max=np.max(data_count))
    params = fmodel.make_params()
    result = fmodel.fit(data_count, params, c_oi=data_oi, c_ii=data_ii, c_if=data_if, c_of=data_of)
    errors = result.eval_uncertainty()
    print(result.fit_report())
    return result.params["pre_oi"].value, result.params["pre_ii"].value, result.params["pre_if"].value, result.params[
        "pre_of"].value, result.params["shift_oi"].value, result.params["shift_of"].value, result.params[
               "shift_i"].value, result.params["scale"].value, result.params["shift_y"].value, errors


def current_adjust(coil, period):
    if coil < 4 - period:
        coil += period
    elif coil > period:
        coil -= period
    else:
        pass
    return coil


def coils_fitting(meas):
    fmodel = Model(coil_4d, independent_vars=["c_oi", "c_ii", "c_if", "c_of"])

    c_oi = np.array([])
    c_ii = np.array([])
    c_if = np.array([])
    c_of = np.array([])
    counts = np.array([])
    coils = [c_oi, c_ii, c_if, c_of]  # it has to have the sequence as the channel list in univ

    counts_min = np.array([])
    counts_max = np.array([])

    for scan_number in meas.scans:
        data_file = RawData(scan_number)
        if data_file.relevant_scan is False:
            print("Scan No. {} is not complete".format(scan_number))
            continue
        if data_file.scan_name in univ.channels:
            for i in range(len(coils)):
                if univ.channels[i] == data_file.scan_name:
                    coils[i] = np.append(coils[i], data_file.scan_x)
                else:
                    coils[i] = np.append(coils[i], np.repeat(data_file.COILS_CURRENTS[univ.channels[i]],
                                                             data_file.scan_x.shape[0]))
        else:
            continue
        counts = np.append(counts, data_file.scan_count)
        counts_min = np.append(counts_min, np.min(counts))
        counts_max = np.append(counts_max, np.max(counts))

        # print(scan_number, data_file.scan_count)

    c_oi, c_ii, c_if, c_of = coils
    # print(c_oi, "\n", c_ii, "\n", c_if, "\n", c_of, "\n", counts)
    pre_oi, pre_ii, pre_if, pre_of, shift_oi, shift_of, shift_i, scale, shift_y, errors = lm_fit_4d(
        meas,
        fmodel,
        c_oi,
        c_ii,
        c_if,
        c_of,
        counts, delta_i, delta_f)
    err_relative = errors / counts

    return pre_oi, pre_ii, pre_if, pre_of, shift_oi, shift_of, shift_i, scale, shift_y, np.min(
        counts_min), np.max(counts_max), errors, err_relative


def phase2element(pre_oi, pre_ii, pre_if, pre_of, shift_oi, shift_of, shift_i,
                  alpha_i, beta_i, alpha_f,
                  beta_f):
    period_oi = abs(2 * np.pi / pre_oi)
    period_ii = abs(2 * np.pi / pre_ii)
    period_if = abs(2 * np.pi / pre_if)
    period_of = abs(2 * np.pi / pre_of)

    c_oi_element = phase2current(precession_phase=alpha_i, prefactor=pre_oi, shift=shift_oi)
    c_ii_element = phase2current(precession_phase=beta_i, prefactor=pre_ii, shift=shift_i)
    c_if_element = phase2current(precession_phase=beta_f, prefactor=pre_if, shift=shift_i)
    c_of_element = phase2current(precession_phase=alpha_f, prefactor=pre_of, shift=shift_of)

    c_oi_element = current_adjust(c_oi_element, period_oi)
    c_ii_element = current_adjust(c_ii_element, period_ii)
    c_if_element = current_adjust(c_if_element, period_if)
    c_of_element = current_adjust(c_of_element, period_of)

    # if pi != AXIS_Z and pf != AXIS_Z:
    pol = pol_4d(c_oi_element, c_ii_element, c_if_element, c_of_element, pre_oi, pre_ii,
                 pre_if, pre_of, shift_oi, shift_of, shift_i, delta_i, delta_f, scale)
    return pol, c_oi_element, c_ii_element, c_if_element, c_of_element


# def plot_inner_coils(pre_oi, pre_ii, pre_if, pre_of, shift_oi, shift_of, shift_i):
#     phase_oi, phase_of = np.pi / 2.0, np.pi / 2.0
#     c_oi = phase2current(precession_phase=phase_oi, prefactor=pre_oi, shift=shift_oi)
#     c_of = phase2current(precession_phase=phase_of, prefactor=pre_of, shift=shift_of)
#     c_ii_1d = np.linspace(0, 4, 41)
#     c_if_1d = np.linspace(0, 4, 41)
#     c_ii_2d, c_if_2d = np.meshgrid(c_ii_1d, c_if_1d)
#     pol_2d = pol_4d(c_oi, c_ii_2d, c_if_2d, c_of, pre_oi, pre_ii, pre_if, pre_of,
#                     shift_oi, shift_of, shift_i, DELTA_I, DELTA_F, scale)
#
#     filename1 = "PG_InnerCoils_2DFitted_Global_MnSi.png"
#     fig1, ax1 = plt.subplots(figsize=(10, 5))
#     cnt1 = ax1.contourf(c_ii_1d, c_if_1d, pol_2d)
#     ax1.set_xlabel(r"$I$ (A): {:s}".format(univ.ii_position), fontname='sans-serif')
#     ax1.set_ylabel(r"$I$ (A): {:s}".format(univ.if_position), fontname="Arial")
#     cbar1 = fig1.colorbar(cnt1)
#     ax1.tick_params(axis="both", direction="in")
#     fig1.savefig(filename1, bbox_inches='tight')
#     plt.close(fig1)


def plot_1d_scans(measure, index, pre_oi, pre_ii, pre_if, pre_of, shift_oi,
                  shift_of, shift_i, scale, shift_y, count_min, count_max, errors):
    t_start = 0
    for scan_number in measure.scans:
        data_file = RawData(scan_number)
        c_oi = 0
        c_ii = 0
        c_if = 0
        c_of = 0
        coils = [c_oi, c_ii, c_if, c_of]
        coil_plot = None
        if data_file.complete_scan is False:
            print("Scan No. {} is not complete".format(scan_number))
            continue
        for i in range(len(coils)):
            if univ.channels[i] == data_file.scan_name:
                coils[i] = np.linspace(-2, 4, 61)
                coil_plot = np.linspace(-2, 4, 61)
            elif data_file.scan_name in univ.channels:
                coils[i] = data_file.COILS_CURRENTS[univ.channels[i]]
            else:
                raise ValueError("Invalid scan name {}".format(data_file.scan_name))
        c_oi, c_ii, c_if, c_of = coils
        scan_coil = data_file.scan_x
        scan_counts = data_file.scan_count
        fit_counts = coil_4d(c_oi, c_ii, c_if, c_of, pre_oi, pre_ii, pre_if,
                             pre_of, shift_oi, shift_of, shift_i, delta_i, delta_f, scale, shift_y)
        t_end = t_start + scan_coil.shape[0]

        filename = "Scan{:d}.png".format(scan_number)
        filename = "/".join([FOLDERS_SCAN_1D[index], filename])
        fig, ax = plt.subplots(figsize=(4.5, 3))  # figsize=(10, 5)
        ax.errorbar(scan_coil, scan_counts, yerr=errors[t_start:t_end])  # , fmt="o"
        ax.plot(coil_plot, fit_counts)
        ax.set_xlabel("{} (A)".format(data_file.POSITIONS[data_file.CHANNELS.index(data_file.scan_name)]))
        ax.set_ylabel("Counts")
        ax.set_ylim(count_min * 0.9, count_max * 1.1)
        text_pairing_coil = "{}: {:.1f} A ".format(data_file.POSITIONS[data_file.CHANNELS.index(data_file.pair_coil)],
                                                   data_file.COILS_CURRENTS[data_file.pair_coil])
        ax.set_title(text_pairing_coil)
        ax.tick_params(axis="both", direction="in")
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        t_start = t_end


for index, sample in enumerate(SAMPLES):
    measure = Parameters(sample_name=sample)
    theta = np.arcsin(WAVELENGTH / (2 * measure.d_spacing))
    delta_i = theta
    delta_f = np.pi - theta
    f = open(FILES_MATRIX[index], "w+")
    f.write(
        "pi, pf, pif, coil_o1 (°), coil_o1 (A), coil_i1 (°), coil_i1 (A), coil_i2 (°), coil_i2 (A), coil_o2 (°), coil_o2 (A)\n")
    pre_oi, pre_ii, pre_if, pre_of, shift_oi, shift_of, shift_i, scale, shift_y, count_min, count_max, errors, err_relative = coils_fitting(
        measure)
    plot_1d_scans(measure, index, pre_oi, pre_ii, pre_if, pre_of, shift_oi,
                  shift_of, shift_i, scale, shift_y, count_min, count_max, errors)
    print(err_relative)

    for pi in AXES:
        for pf in AXES:
            alpha_i, beta_i, alpha_f, beta_f = polarisation2angles(pi, pf, 2 * theta)
            pol, c_oi, c_ii, c_if, c_of = phase2element(pre_oi, pre_ii, pre_if, pre_of, shift_oi, shift_of, shift_i,
                                                        alpha_i, beta_i, alpha_f, beta_f)
            f.write(
                "{:s}, {:s}, {:.3f} ({:.0e}), {:.3f}°, {:.3f} A, {:.3f}°, {:.3f} A, {:.3f}°, {:.3f} A, {:.3f}°, {:.3f} A\n".format(
                    pi, pf, pol, pol * np.mean(err_relative), np.rad2deg(alpha_i), c_oi, np.rad2deg(beta_i), c_ii,
                    np.rad2deg(beta_f), c_if,
                    np.rad2deg(alpha_f), c_of))
    f.close()

    # plot_inner_coils(pre_oi, pre_ii, pre_if, pre_of, shift_oi, shift_of, shift_i)
