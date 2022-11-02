import re

PATTERN_VARIABLE = re.compile(r"se_ps[1-4]_out[1-2]?")
PATTERN_STEP = re.compile(r"[+-]?[0-9*]\.[0-9]*e?[+-]?[0-9]*")


def variable_parser(line):
    variables_all = re.findall(pattern=PATTERN_VARIABLE, string=line)
    steps_all = re.findall(pattern=PATTERN_STEP, string=line)
    steps_all = list(map(float, steps_all))
    print(variables_all)
    print(steps_all)


line = "Scanning Variables: se_ps1_out,se_ps2_out,se_ps3_out,se_ps4_out1,se_ps4_out2, Steps: 1.000000000139778e-06, 8.999999999925734e-06, 8.000000000230045e-06, 0.0, 0.0"
variable_parser(line)

line2 = "se_ps1_out"
variable_parser(line2)

print(re.match(PATTERN_VARIABLE, line))
print(re.match(PATTERN_VARIABLE, line2))
