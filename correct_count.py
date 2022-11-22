import numpy as np
from raw_data import RawData
import universal_params as univ

correct_nsf = [3173, 3175]
correct_sf = [3174, 3176]

reference_nsf = list(range(3185, 3216 + 1, 2)) + list(range(3218, 3223 + 1, 2))
reference_sf = list(range(3186, 3216 + 1, 2)) + list(range(3219, 3223 + 1, 2))


def count2norm(count, monitor):
    return count / monitor * 1e6


def norm2count(count_normed, monitor):
    return count_normed * 1e-6 * monitor


def data_loader(number_list):
    scan_coils = np.empty((4, 0))
    counts = np.array([], dtype=int)
    monitors = np.array([], dtype=int)
    for number in number_list:
        data_file = RawData(number)
        data = data_file.scan_data
        if len(data.shape) == 1:
            data_4d = np.zeros((4, data.shape[0]))
            index = univ.positions.index(data_file.scan_posn)
            data_4d[index, :] = data
            scan_coils = np.append(scan_coils, data_4d, axis=1)
        else:
            scan_coils = np.append(scan_coils, data, axis=1)
        counts = np.append(counts, data_file.scan_count)
        monitors = np.append(monitors, np.repeat(data_file.monitor_count, data_file.scan_count.shape[0]))
    return scan_coils, counts, monitors


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
    scan_coils_ref, counts_ref, monitors_ref = data_loader(correct_ref)

    correct_factor = get_correction_factor(counts=counts, counts_ref=counts_ref, coils_scan=scan_coils,
                                           coils_ref=scan_coils_ref, monitors=monitors, monitors_ref=monitors_ref)
    counts_corrected = np.round(norm2count(counts * correct_factor, monitors))
    print(counts, "\n", counts_corrected)


print("NSF")
correct_count(correct_obj=correct_nsf, correct_ref=reference_nsf)
print("SF")
correct_count(correct_obj=correct_sf, correct_ref=reference_sf)
