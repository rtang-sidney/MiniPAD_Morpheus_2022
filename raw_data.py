import numpy as np
import re
import universal_params as univ


class RawData:
    PATTERN_POINTS = re.compile(r"([0-9]*)\sPoints")
    PATTERN_VARIABLE = re.compile(r"se_ps[1-4]_out[1-2]?")
    PATTERN_STEP = re.compile(r"[+-]?[0-9*]\.[0-9]*e?[+-]?[0-9]*")
    # PATTERN_CURRENTS = re.compile(r"(se_ps[1-4]_out[1-2]?)\s*=\s*([1-9]*.*[1-9]*)")
    PATTERN_MONITOR = re.compile(r"scan\s\S*\s\S*\s\S*\s\S*\s([0-9]*)")  # tti[1-2]_out[1-2]\s
    PATTERN_TEMP = re.compile(r"TEMP\s*=\s*([1-9]*.*[1-9]*)")

    DATA_HEADER = "**************************** DATA ******************************************\n"
    SCANNING_VARIABLE_KEY = "Scanning Variables"
    SCAN_COMMAND = "Last Scan Command"
    SCAN_NP = "NP"

    FOLDER = "../MorpheusData/"
    FILE_PREFIX = 'morpheus2022n'
    FILE_FORMAT = '.dat'

    PARAMETER_SUFFIX = '_value'
    DATA_START = 'Scan data'
    COUNT_TIME = 'det_preset'
    SCAN_INFO = 'info'
    TEMP = "TEMP"

    PLOT_Y_LABEL = 'Normalised counts'
    MONITOR_REFERENCE = int(6e6)

    TTI2_OUT1 = "se_ps4_out1"
    TTI2_OUT2 = "se_ps4_out2"
    TTI_TOL_ZERO = 1e-2  # tolerance for the tti current

    OI_CH = 'se_ps1_out'
    OF_CH = 'se_ps2_out'
    II_CH = 'se_ps3_out'
    IF_POS_CH = TTI2_OUT1
    IF_NEG_CH = TTI2_OUT2
    IF_CHS = [IF_POS_CH, IF_NEG_CH]
    CHANNELS = [OI_CH, OF_CH, II_CH, IF_POS_CH, IF_NEG_CH]
    CHANNELS_POSITIONS = dict(zip(CHANNELS, univ.positions + [univ.positions[-1]]))

    def __init__(self, file_index):
        print("In process: {}".format(file_index))
        self.relevant_scan = True

        # initialise current of all 4 coils (5 channels)
        self.current_oi = 0.0
        self.current_of = 0.0
        self.current_ii = 0.0
        self.current_if = 0.0
        self.currents = [self.current_oi, self.current_of, self.current_ii, self.current_if]
        # dictionary as {coil_position: current}
        self.coils_currents = dict(zip(univ.positions, self.currents))

        self.filename = self.FOLDER + self.FILE_PREFIX + "{:0>6d}".format(file_index) + self.FILE_FORMAT
        self.header_number = None
        self._number_points = 0
        self._scan_channel = None
        self.scan_posn = None
        self.step_size = 0
        self.monitor_count = 0
        self.temperature = 0

        self._scanned_ind = None
        self._number_variables = 0

        try:

            self._get_info()
            self._get_data()
            self._get_object()

        except FileNotFoundError as fnfe:
            print(fnfe)
            self.relevant_scan = False
        except ValueError as ve:
            print(ve)
            self.relevant_scan = False
        except TypeError as te:
            print(te)
            self.relevant_scan = False

    def _get_info(self):
        file_opened = open(self.filename, 'r')
        lines = file_opened.readlines()
        self.header_number = lines.index(self.DATA_HEADER)
        self.header_number += 3 + 1  # there are still three lines after the "data" line
        for line in lines:
            if re.search(self.PATTERN_POINTS, line):
                self._number_points = int(re.search(self.PATTERN_POINTS, line).groups()[0])
                if self._number_points == 0:
                    self.relevant_scan = False
                    raise ValueError("No number of points found")
            # elif re.search(self.PATTERN_CURRENTS, line):
            #     channel, current = re.search(self.PATTERN_CURRENTS, line).groups()
            #     if channel in self.CHANNELS:
            #         if channel == self.IF_POS_CH:
            #             self.if_pos = float(current)
            #         elif channel == self.IF_NEG_CH:
            #             self.if_neg = float(current)
            #         else:
            #             self.coils_currents[univ.positions[self.CHANNELS.index(channel)]] = float(current)
            # the variables have to be searched directly in the line next to data instead of the line with #
            # "scanning variables"
            elif self.SCAN_NP in line:
                if re.search(self.PATTERN_VARIABLE, line):
                    self.variables_all = np.array(re.findall(pattern=self.PATTERN_VARIABLE, string=line))
                    self._number_variables = self.variables_all.shape[0]
                else:
                    raise ValueError("No relevant variables found")
            else:
                pass

    def _get_data(self):
        if self._number_variables == 0:
            raise ValueError("No (relevant) scanning variables found.")
        data = np.loadtxt(self.filename, skiprows=self.header_number, max_rows=self._number_points)
        self.scan_count = data[:, self._number_variables + 1]
        if self._number_variables == 1:
            self.scan_data = data[:, 1]
            # print(self.scan_data.shape)
            self.scan_x = self.scan_data
            self.step_size = np.round(np.mean(np.diff(self.scan_x[1:])), 1)
            self._scan_channel = self.variables_all[0]
        else:
            self.scan_data = data[:, 1:self._number_variables + 1].transpose()

            # the steps were often recorded with mistakes. check and correct it is necessary
            steps = np.mean(np.diff(self.scan_data[:, 1:], axis=1), axis=1)
            self._scanned_ind = np.abs(steps) > 9e-2
            if np.count_nonzero(self._scanned_ind) != 1 and self._number_variables > 0:
                raise RuntimeError(
                    "Failed to find scan object, scanned index {}, No. variables {}".format(self._scanned_ind,
                                                                                            self._number_variables))
            # steps[self.scanned_ind] would be an array with size of 1
            self._scanned_ind = np.nonzero(self._scanned_ind)[0][0]
            self.step_size = round(steps[self._scanned_ind], 1)
            self._scan_channel = self.variables_all[self._scanned_ind]
            # print(self._scanned_ind, self._scan_channel)
            self.scan_x = self.scan_data[self._scanned_ind, :]
            # Somehow the data can be with in a 2D array with the shape of (1, n), where n is the real length
        if self.scan_x.ndim == 2 and self.scan_x.shape[0] == 1:
            self.scan_x = self.scan_x[0]
        if self.scan_x.ndim == 2 and self.scan_x.shape[1] == 1:
            self.scan_x = self.scan_x.transpose()
        self.scan_x[0] = round(self.scan_x[0], 1)
        self.scan_x = np.linspace(self.scan_x[0], self.scan_x[0] + self.step_size * (self._number_points - 1),
                                  num=self._number_points)

        # print(self.scan_x.shape, self.scan_count.shape, self.step_size)
        if self.scan_x.shape != self.scan_count.shape:
            raise RuntimeError(
                "xdata and ydata have different dimensions: {}, {}".format(self.scan_x.shape,
                                                                           self.scan_count.shape))
        if abs(self.step_size) < 9e-2:
            raise RuntimeError("Still no finite step size found.")
        # print(self.scan_pos, self._scan_channel)
        if self._scan_channel == self.TTI2_OUT2:  # self.scan_coil == univ.if_posn and
            self.scan_x = -self.scan_x
            # self._scan_channel = self.TTI2_OUT1
            self._scanned_ind = -2

        # after the correction of scan_x, the whole data set has to be updated
        if self._number_variables == 1:
            self.scan_data = self.scan_x
            # self.coils_currents[self.CHANNELS_POSITIONS[self._scan_channel]] = self.scan_x
        else:
            self.scan_data[self._scanned_ind, :] = self.scan_x
            self.scan_data = np.delete(self.scan_data, -1, axis=0)  # the channel tti2_out2 does not matter any more
            # for posn in univ.positions:
            #     if posn == self.CHANNELS_POSITIONS[self._scan_channel]:
            #         self.coils_currents[posn] = self.scan_x
            #     else:
            #         self.coils_currents[posn] = np.mean(self.scan_data[univ.positions.index(posn)])

    def _get_object(self):
        if self._scan_channel in self.IF_CHS:
            self.scan_posn = univ.if_posn
            self.pair_posn = univ.ii_posn
        elif self._scan_channel == self.OI_CH:
            self.scan_posn = univ.oi_posn
            self.pair_posn = univ.of_posn
        elif self._scan_channel == self.OF_CH:
            self.scan_posn = univ.of_posn
            self.pair_posn = univ.oi_posn
        elif self._scan_channel == self.II_CH:
            self.scan_posn = univ.ii_posn
            self.pair_posn = univ.if_posn
        elif self.relevant_scan:
            raise RuntimeError("Relevant scan but no channel found: {}".format(self._scan_channel))
