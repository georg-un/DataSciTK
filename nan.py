import numpy as np
import pandas as pd

from _helper import _check_input_array
import check_data as cd
from utils import match_by_pattern
from type_ops import contains_types
from type_ops import match_by_type
from type_ops import get_contained_types


def contains_nan(input_array):
    """
    Check if array contains NaN values. Works for arrays of strings as well.

    :param input_array:     1-dimensional numpy array
    :return:                True or False

    """

    # Check if input is valid
    _check_input_array(input_array)

    # Check string values for NaN
    if contains_types(input_array, 'str', exclusively=False, verbose=False):
        string_values = input_array[match_by_type(input_array, 'str')]
        string_values = string_values[match_by_pattern(string_values, ['nan', 'NaN', 'NAN', 'N/A'])]
        if string_values.size > 0:
            return True

    # Check non-string values for NaN
    if not contains_types(input_array, 'str', exclusively=True, verbose=False):
        non_string_values = input_array[[type(element) is not str for element in input_array]]
        try:
            if any(pd.isnull(non_string_values)):
                return True
        except TypeError:
            types = get_contained_types(input_array, unique=True, as_string=True)
            print("Warning: Input array contains types which cannot be checked for NaN. Found types are: {0}.".format(
                types
            ))
            return True  # better save than sorry..

    # If no matches have been found
    return False


def recode_binary(input_array):
    """
    Replaces the NaN's in the input array by a 1 and all other values by a 0.

    :param input_array:     1-dimensional numpy array
    :return:                1-dimensional numpy array with a 1 for every NaN and a 0 for every other value

    """

    # Check if input is valid
    _check_input_array(input_array)

    if not cd.is_type_homogeneous(input_array, verbose=False):
        raise TypeError("Input array contains multiple types ({0}). Please use only type homogeneous types.".format(
            cd.get_contained_types(input_array, unique=True, as_string=True)
        ))

    if not contains_nan(input_array):
        raise TypeError("Input array does not contain any NaN. All result values would be zero.")

    # Find NaN's
    if isinstance(input_array[0], str):
        binary_array = match_by_pattern(input_array, ['nan', 'NaN', 'NAN', 'N/A'])
    else:
        binary_array = [np.isnan(value) for value in input_array]

    # convert from True and False to 1 and 0
    binary_array = [[int(value) for value in binary_array]]

    return binary_array

