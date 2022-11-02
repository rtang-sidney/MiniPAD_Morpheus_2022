import numpy as np
import matplotlib.pyplot as plt

FOLDER = "../MorpheusData/"
FILE_PREFIX = 'morpheus2022n'
FILE_FORMAT = '.dat'
f1 = 2227
f2 = 3175
fname1 = FOLDER + FILE_PREFIX + "{:0>6d}".format(f1) + FILE_FORMAT
fname2 = FOLDER + FILE_PREFIX + "{:0>6d}".format(f2) + FILE_FORMAT

coil = np.linspace(-2.9, 2.9, 59)
data1 = np.loadtxt(fname1, skiprows=55, max_rows=59)[:,2]
data2 = np.loadtxt(fname2, skiprows=55, max_rows=59)[:,2]
plt.subplots()
plt.plot(coil, data1 / np.max(data1))
plt.plot(coil, data2 / np.max(data2))
plt.show()
