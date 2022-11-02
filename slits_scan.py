import numpy as np
import matplotlib.pyplot as plt

files = range(1968, 2133 + 1)
# fplus = 1968
# fminus = 1969
# files = [fplus, fminus]

FILE_PREFIX = 'morpheus2022n'
FILE_FORMAT = '.dat'
FOLDER = "Morpheus"


def read_data(file_index):
    fname = FOLDER + "/" + FILE_PREFIX + "{:0>6d}".format(file_index) + FILE_FORMAT
    data = np.loadtxt(fname=fname, skiprows=55, max_rows=20)
    print(data[:, 1:5])
    return data[:, 1:5]


fig, ax = plt.subplots()
counts_mm = np.empty(20)
counts_pm = np.empty(20)
data_width = np.empty(20)
fm = 1968
fp = fm + 1

while fm < files[-2]:  # files[-2]:
    data_m = read_data(fm)
    while data_m.shape[0] < 20:
        fm += 1
        data_m = read_data(fm)

    data_width = data_m[:, 0]
    counts_mm = data_m[:, -1]

    fp = fm + 1
    data_p = read_data(fp)
    while data_p.shape[0] < 20:
        fp += 1
    counts_pm = data_p[:, -1]
    flip = counts_mm / counts_pm
    ax.plot(data_width, flip)
    print("fm {}, fp {}".format(fm, fp))

    fm = fp + 1

plt.show()
