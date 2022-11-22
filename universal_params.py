import numpy as np

wavelength = 4.905921e-10
d_pg002 = 3.55e-10
d_mnsi011 = 4.556e-10

oi_posn = 'OuterIncoming'
of_posn = 'OuterOutgoing'
ii_posn = 'InnerIncoming'
if_posn = 'InnerOutgoing'
positions = [oi_posn, of_posn, ii_posn, if_posn]

# coils_positions = dict(zip(channels, positions))
fname_parameters = "Coil_Parameters.dat"


def count2pol(count_nsf, count_sf):
    return (count_nsf - count_sf) / (count_nsf + count_sf)


def err_count2pol(count_plus, count_minus):
    return 2.0 * np.sqrt(count_plus * count_minus / (count_plus + count_minus) ** 3)
