import numpy as np
import scipy

from _input_checks import check_numpy_array_1d


def find_and_replace(data, pattern, replacement):
    """
    If a value in the data matches at least one pattern exactly, replace it with the specified replacement.


    :param data:            1-dimensional numpy array. Contains the input values.
    :param pattern:         Single value or list. Must be the same type as the values in 'data'.
    :param replacement:     Single value. Must be the same type as the values in 'data'.

    :return:                1-dimensional numpy array
    """

    # Transform pattern parameter to list
    if type(pattern) is not list:
        pattern = [pattern]

    # Check if inputs are valid
    check_numpy_array_1d(data, 'data')

    for element in pattern:
        if not isinstance(data[0], type(element)):
            raise TypeError("Type of pattern {0} ({1}) must match type of values in 'data' ({2})".format(
                element,
                type(element),
                type(data[0])))

    if not isinstance(data[0], type(replacement)):
        print("Warning: Type of replacement ({0}) does not match type of the values in 'data' ({1}). ".format(
            type(replacement),
            type(data[0])))

    # Replace values
    data[np.in1d(data, pattern)] = replacement

    return data


