import numpy as np
import re
import universal_params as univ


class RawData:
    PATTERN_POINTS = re.compile(r"([0-9]*)\sPoints")
    PATTERN_VARIABLE = re.compile(r"se_ps[1-4]_out[1-2]?")
    PATTERN_STEP = re.compile(r"[+-]?[0-9*]\.[0-9]*e?[+-]?[0-9]*")
    PATTERN_CURRENTS = re.compile(r"(se_ps[1-4]_out[1-2]?)\s*=\s*([1-9]*.*[1-9]*)")
    PATTERN_MONITOR = re.compile(r"scan\s\S*\s\S*\s\S*\s\S*\s([0-9]*)")  # tti[1-2]_out[1-2]\s
    PATTERN_TEMP = re.compile(r"TEMP\s*=\s*([1-9]*.*[1-9]*)")

    DATA_HEADER = "**************************** DATA ******************************************\n"
    SCANNING_VARIABLE_KEY = "Scanning Variables"
    SCAN_COMMAND = "Last Scan Command"
    SCAN_NP = "NP"

    FOLDER = "../MorpheusData/"
    FILE_PREFIX = 'morpheus2022n'
    FILE_FORMAT = '.dat'

    # POWER_SUPPLY_PREFIX = "tti"

    PARAMETER_SUFFIX = '_value'
    DATA_START = 'Scan data'
    COUNT_TIME = 'det_preset'
    SCAN_INFO = 'info'
    TEMP = "TEMP"

    PLOT_Y_LABEL = 'Normalised counts'
    MONITOR_REFERENCE = int(6e6)

    TTI2_OUT1 = "se_ps4_out1"
    TTI2_OUT2 = "se_ps4_out2"
    # tti2 = [tti2_out1, tti2_out2]
    TTI_TOL_ZERO = 1e-2  # tolerance for the tti current

    OI_CH = 'se_ps1_out'
    OF_CH = 'se_ps2_out'
    II_CH = 'se_ps3_out'
    IF_POS_CH = TTI2_OUT1
    IF_NEG_CH = TTI2_OUT2
    IF_CHS = [IF_POS_CH, IF_NEG_CH]
    CHANNELS = [OI_CH, OF_CH, II_CH, IF_POS_CH, IF_NEG_CH]

    def __init__(self, file_index):
        print("In process: {}".format(file_index))
        # two criterium to check if the scan is useful
        self.relevant_scan = True

        # initialise current of all 4 coils (5 channels)
        self.current_oi = 0.0
        self.current_ii = 0.0
        self.current_if = 0.0
        self.current_of = 0.0
        self.currents = [self.current_ii, self.current_of, self.current_if, self.current_oi]

        # dictionary as {coil_position: current}
        self.coils_currents = dict(zip(univ.positions, self.currents))

        self.filename = self.FOLDER + self.FILE_PREFIX + "{:0>6d}".format(file_index) + self.FILE_FORMAT
        self.header_number = None
        self.number_points = 0
        self.scan_channel = None
        self.scan_coil = None
        self.step_size = 0
        self.monitor_count = 0
        self.temperature = 0
        self.if_pos = 0.0
        self.if_neg = 0.0
        self.scanned_ind = None
        self._number_variables = 0

        try:
            file_opened = open(self.filename, 'r')
            lines = file_opened.readlines()
            self.header_number = lines.index(self.DATA_HEADER)
            self.header_number += 3 + 1  # there are still three lines after the "data" line

            self._get_info(lines)
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

    def _get_info(self, lines):
        for line in lines:
            if re.search(self.PATTERN_POINTS, line):
                self.number_points = int(re.search(self.PATTERN_POINTS, line).groups()[0])
                if self.number_points == 0:
                    self.relevant_scan = False
                    raise ValueError("No number of points found")
            elif self.SCANNING_VARIABLE_KEY in line:
                if re.search(self.PATTERN_VARIABLE, line):
                    self.variables_all = np.array(re.findall(pattern=self.PATTERN_VARIABLE, string=line))
                    steps_all = re.findall(pattern=self.PATTERN_STEP, string=line)
                    steps_all = np.array(list(map(float, steps_all)))
                    self._number_variables = self.variables_all.shape[0]
                    if self.variables_all.shape[0] != steps_all.shape[0]:
                        raise ValueError("Coils are scanned together with another device")
                else:
                    raise ValueError("No scan variables found")

                # Try to find the scan step
                # If no stepsize is recorded by NICOS, it gives stepsize=0
                if self._number_variables == 1:
                    self.scanned_ind = 0
                    # self.scan_channel, self.step_size = self.variables_all[0], steps_all[0]
                    self.scan_channel = self.variables_all[0]
                else:
                    self.scanned_ind = np.abs(steps_all) > 1e-3
                    if np.count_nonzero(self.scanned_ind) == 1:
                        # print(self.variables_all, steps_all, self.scanned_ind)
                        # self.scan_channel, self.step_size = self.variables_all[self.scanned_ind], steps_all[
                        #     self.scanned_ind][0]
                        self.scan_channel = self.variables_all[self.scanned_ind]
                    elif np.count_nonzero(self.scanned_ind) == 0:
                        # self.step_size = 0
                        pass
                    else:
                        raise ValueError(
                            "More than one scan object found:\nChannels{},\nSteps{},\nindices{}".format(
                                self.variables_all, steps_all, self.scanned_ind))
            elif re.search(self.PATTERN_CURRENTS, line):
                channel, current = re.search(self.PATTERN_CURRENTS, line).groups()
                if channel in self.CHANNELS:
                    if channel == self.IF_POS_CH:
                        self.if_pos = float(current)
                    elif channel == self.IF_NEG_CH:
                        self.if_neg = float(current)
                    else:
                        self.coils_currents[univ.positions[self.CHANNELS.index(channel)]] = float(current)
            elif self.SCAN_NP in line:
                if re.search(self.PATTERN_VARIABLE, line):
                    self.variables_all = np.array(re.findall(pattern=self.PATTERN_VARIABLE, string=line))
                    self._number_variables = self.variables_all.shape[0]
                    if self.variables_all.shape[0] != self._number_variables:
                        self._number_variables = self.variables_all.shape[0]
                        self.step_size = 0
            else:
                pass

    def _get_data(self):
        if self._number_variables == 0:
            raise ValueError("No (relevant) scanning variables found.")
        data = np.loadtxt(self.filename, skiprows=self.header_number, max_rows=self.number_points)
        self.scan_count = data[:, self._number_variables + 1]
        if self._number_variables == 1:
            self.scan_data = data[:, self._number_variables]
            print(self.scan_data.shape)
            # if self.scan_data.shape == 1:
            #     self.scan_data = self.scan_data.transpose()[0]
            # else:
            #     raise RuntimeError("Wrong format of scan data: {}".format(self.scan_data))
            # print(self.scan_data)
            self.scan_x = self.scan_data
            self.step_size = np.round(np.mean(np.diff(self.scan_x)), 1)
        else:
            self.scan_data = data[:, 1:self._number_variables + 1].transpose()

            # the steps were often recorded with mistakes. check and correct it is necessary
            steps = np.mean(np.diff(self.scan_data, axis=1), axis=1)
            self.scanned_ind = np.abs(steps) > 1e-3
            # steps[self.scanned_ind] would be an array with size of 1
            self.step_size = round(steps[self.scanned_ind][0], 1)
            self.scan_channel = self.variables_all[self.scanned_ind][0]

            # print("Scan variable: {}, scan steps: {}, scan index: {}".format(self.scan_variables, steps,
            #                                                                  self.scanned_ind))
            self.scan_x = self.scan_data[self.scanned_ind, :]
            # Somehow the data can be with in a 2D array with the shape of (1, n), where n is the real length
        if self.scan_x.shape[0] == 1:
            self.scan_x = self.scan_x[0]
        if self.scan_x.ndim == 2 and self.scan_x.shape[1] == 1:
            self.scan_x = self.scan_x.transpose()
        self.scan_x[0] = round(self.scan_x[0], 1)
        self.scan_x = np.linspace(self.scan_x[0], self.scan_x[0] + self.step_size * (self.number_points - 1),
                                  num=self.number_points)

        # print(self.scan_x.shape, self.scan_count.shape, self.step_size)
        if self.scan_x.shape != self.scan_count.shape:
            raise RuntimeError(
                "xdata and ydata have different dimensions: {}, {}".format(self.scan_x.shape, self.scan_count.shape))
        if abs(self.step_size) > 1e-3:
            # self.scan_x = np.linspace(self.scan_x[0], self.scan_x[0] + self.step_size * (self.number_points - 1),
            #                           num=self.number_points)
            pass
        else:
            raise RuntimeError("Still no finite step size found.")
        print(self.scan_coil, self.scan_channel)
        if self.scan_channel == self.TTI2_OUT2:  # self.scan_coil == univ.if_posn and
            self.scan_x = -self.scan_x
            self.scan_channel = self.TTI2_OUT1
            self.scanned_ind = np.array([False, False, False, True, False])
        # if self.scan_x.shape[0] < 5:
        #     self.relevant_scan = False
        if self._number_variables == 1:
            self.scan_data = self.scan_x
        else:
            self.scan_data[self.scanned_ind, :] = self.scan_x
            self.scan_data = np.delete(self.scan_data, -1, axis=0)

    def _get_object(self):
        if self.scan_channel in self.IF_CHS:
            self.scan_coil = univ.if_posn
            self.pair_coil = univ.ii_posn
            if abs(self.if_pos) < self.TTI_TOL_ZERO:
                self.coils_currents[univ.if_posn] = self.if_neg
            elif abs(self.if_neg) < self.TTI_TOL_ZERO:
                self.coils_currents[univ.if_posn] = self.if_pos
            else:
                raise ValueError(
                    "No correct value for of coil found in: [{:.f}, {:.f}]".format(self.if_pos, self.if_neg))
        elif self.scan_channel == self.OI_CH:
            self.scan_coil = univ.oi_posn
            self.pair_coil = univ.of_posn
        elif self.scan_channel == self.OF_CH:
            self.scan_coil = univ.of_posn
            self.pair_coil = univ.oi_posn
        elif self.scan_channel == self.II_CH:
            self.scan_coil = univ.ii_posn
            self.pair_coil = univ.if_posn
        elif self.relevant_scan:
            raise RuntimeError("Relevant scan but no channel found")
