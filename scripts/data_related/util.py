import numpy as np


def convert_to_int(str):

    try:
        int_val = int(str)
        return int_val
    except ValueError:
        return np.nan


def convert_to_float(str):
    try:
        float_val = float(str)
        return float_val
    except ValueError:
        return np.nan


def convert_to_boolean(str):
    try:
        int_val = int(str)
        bool_val = bool(int_val)
        return bool_val
    except ValueError:
        return np.nan