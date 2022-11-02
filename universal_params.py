import numpy as np

wavelength = 4.905921e-10
d_pg002 = 3.55e-10
d_mnsi011 = 4.556e-10

oi_posn = 'upstream, outer'
ii_posn = 'upstream, inner'
if_posn = 'downstream, inner'
of_posn = 'downstream, outer'
positions = [oi_posn, ii_posn, if_posn, of_posn]

# coils_positions = dict(zip(channels, positions))
fname_parameters = "Coil_Parameters.dat"


def count2pol(count_plus, count_minus):
    return (count_plus - count_minus) / (count_plus + count_minus)


def err_count2pol(count_plus, count_minus):
    return 2.0 * np.sqrt(count_plus * count_minus / (count_plus + count_minus) ** 3)
