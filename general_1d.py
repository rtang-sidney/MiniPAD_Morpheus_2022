import numpy as np
from raw_data import RawData
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import universal_params as univ

plt.rcParams.update({'font.size': 18})

# SCAN_1D = range(2181, 2204 + 1)
# MAP_ANFANG = 2185

# SCAN_1D = range(2288, 2333 + 1)
# SCAN_1D = [2808, 2809]
# SCAN_1D = range(2814, 2855 + 1)
# SCAN_0_0 = [2808, 2809]
# SCAN_90_90 = range(2814, 2854 + 1)
# SCAN_0_90 = range(2855, 2895 + 1)
# SCAN_90_0 = range(2896, 2936 + 1)
SCAN_1D = range(1613, 3328 + 1)  # + list(SCAN_90_90) + list(SCAN_0_90) + list(SCAN_90_0)


# SCAN_1D = range(2288, 2288 + 1)


def sine_function(x, a, b, c, d):
    return a * np.cos(b * x + c) + d


def sine_fitting(xdata, ydata, pairing_coil=None):
    if pairing_coil:
        guess = [(np.max(ydata) - np.min(ydata)) / 2.0, np.pi * 2.0 / 6.0, pairing_coil,
                 (np.max(ydata) + np.min(ydata)) / 2.0]
        bound = ([(np.max(ydata) - np.min(ydata)) / 3.0, np.pi * 2.0 / 7.0, pairing_coil - 2, np.min(ydata)],
                 [np.max(ydata) - np.min(ydata), np.pi * 2.0 / 3.5, pairing_coil + 2, np.max(ydata)])
    else:
        guess = [(np.max(ydata) - np.min(ydata)) / 2.0, np.pi * 2.0 / 6.0, 0, (np.max(ydata) + np.min(ydata)) / 2.0]
        bound = ([(np.max(ydata) - np.min(ydata)) / 3.0, np.pi * 2.0 / 7.0, -6, np.min(ydata)],
                 [np.max(ydata) - np.min(ydata), np.pi * 2.0 / 4.0, 6, np.max(ydata)])
    try:
        popt, pcov = curve_fit(sine_function, xdata, ydata, p0=guess, bounds=bound)
    except ValueError:
        print("xdata:", xdata, "ydata:", ydata)
        raise KeyboardInterrupt
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def counts2polarisation(fit_count, sine_params):
    # print(count_data)
    if isinstance(sine_params, np.ndarray) is False:
        raise TypeError("Wrong type of z data are given")
    if sine_params.ndim != 1:
        raise TypeError("Data of wrong dimension are given")
    # intensity_total = 2 * sine_params[-1]
    pol_max = sine_params[0] / sine_params[-1]
    polarisation = (fit_count - sine_params[-1]) / sine_params[0] * pol_max
    return polarisation


current_pairing_coil = 0.0
for scan_1d in SCAN_1D:
    print("File: {}".format(scan_1d))
    filename = "General_1D/Scan{:d}.png".format(scan_1d)
    data_file = RawData(scan_1d)
    if data_file.relevant_scan:
        print("relevant scan: {}".format(data_file.relevant_scan))
    else:
        continue
    print(data_file.step_size, data_file.scan_x, data_file.scan_count)
    sine_params = sine_fitting(data_file.scan_x, data_file.scan_count)[0]
    print(2 * np.pi / sine_params[1] / 4.0)

    fig, ax = plt.subplots()
    ax.plot(data_file.scan_x, data_file.scan_count, "o")
    ax.tick_params(axis="both", direction="in")
    ax.set_xlabel("{} (A)".format(data_file.scan_posn))
    ax.set_ylabel("Counts")

    plot_x = np.linspace(-3, 3, num=100)
    plot_y = sine_function(plot_x, *sine_params)
    fit_pol = counts2polarisation(plot_y, sine_params)
    ax.plot(plot_x, plot_y)
    ax.set_ylim(np.min(plot_y), np.max(plot_y))
    print(sine_params[2], np.pi / sine_params[1], np.pi / (2.0 * sine_params[1]) - sine_params[2])
    print(sine_params[0] + sine_params[-1], -sine_params[0] + sine_params[-1])
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    colour_ax2 = "green"
    ax2.plot(plot_x, fit_pol * 100, color=colour_ax2)
    ax2.set_ylim(np.min(fit_pol * 100), np.max(fit_pol * 100))

    ax2.tick_params(axis="y", direction="in")
    ax2.set_ylabel(r"Polarisation (%)", color=colour_ax2)
    ax2.tick_params(axis='y', labelcolor=colour_ax2)

    # plt.show()
    plt.savefig(filename, bbox_inches='tight')
