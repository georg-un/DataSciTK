import numpy as np

from _helper import _check_numpy_array_1d
from _helper import _check_numpy_array_pandas_series_1d
from type_ops import is_type_homogeneous
from type_ops import get_contained_types


def match_by_pattern(input_array, patterns):
    """
    Find all occurrences in the input array which are defined in patterns. Returns a list of True and False values.

    :param input_array:         1-dimensional numpy array or pandas Series.
    :param patterns:            Single value or list of values. Must be of the same type as the values in the input.

    :return:                    1-dimensional numpy array containing True and False values.

    """

    # Make patterns to a list if it is not already one
    if type(patterns) is not list:
        patterns = [patterns]

    # Check if inputs are valid
    _check_numpy_array_pandas_series_1d(input_array)

    if not is_type_homogeneous(input_array, verbose=False):
        raise TypeError("Input array contains multiple types ({0}) but must be type homogeneous.".format(
            get_contained_types(input_array, unique=True, as_string=True)
        ))

    # Get matches
    matched_array = [value in patterns for value in input_array]

    return matched_array


def contains_negatives_or_zero(input_array):
    """
    Checks if the input contains only values larger than 0. Returns True or False.

    :param input_array:         1-dimensional numpy array or pandas Series of data-type float or int.
    :return:                    True if input contains only values larger than zero. False otherwise.
    """

    # Check if input is valid
    _check_numpy_array_pandas_series_1d(input_array)

    if not is_type_homogeneous(input_array, verbose=False):
        raise TypeError("Input must be type homogeneous but contains multiple types: {0}.".format(
            get_contained_types(input_array, unique=True, as_string=True)
        ))

    if type(input_array[0]) is not int and type(input_array[0]) is not float and type(input_array[0]) is not np.float:
        if type(input_array[0]) is not np.float64 and type(input_array[0]) is not np.int64:
            raise TypeError("Input must contain integer or float values but contains type {0}.".format(
                type(input_array[0])
            ))

    # Count number of negatives
    zero_or_smaller = input_array[input_array <= 0]

    # Check number of negatives and return respective result
    if zero_or_smaller.size == 0:
        return False
    elif zero_or_smaller.size > 0:
        return True
    else:
        raise IOError("Number of values which are zero or smaller ({0}) was neither 0 nor a number larger than 0.".format(
            zero_or_smaller
        ))
