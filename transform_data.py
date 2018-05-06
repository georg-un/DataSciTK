import numpy as np
from check_data import _check_input_array


def find_and_replace(input_array, pattern, replacement):
    """
    Take a 1 dimensional numpy array as input. If a value of the array matches at least one pattern exactly,
    replace it with the specified replacement.


    :param input_array:     1-dimensional numpy array. Contains the input values.
    :param pattern:         Single value or list. Must be the same type as the values in the input array.
    :param replacement:     Single value. Must be the same type as the values in the input array.

    :return:                1-dimensional numpy array
    """

    # Transform pattern parameter to list
    if type(pattern) is not list:
        pattern = [pattern]

    # Check if inputs are valid
    _check_input_array(input_array)

    for element in pattern:
        if not isinstance(input_array[0], type(element)):
            raise TypeError('Type of pattern {0} ({1}) must match type of the input array values ({2})'.format(
                element,
                type(element),
                type(input_array[0])))

    if not isinstance(input_array[0], type(replacement)):
        print('Warning: Type of replacement ({0}) does not match type of the input array values ({1}). '.format(
            type(replacement),
            type(input_array[0])))

    # Replace values
    input_array[np.in1d(input_array, pattern)] = replacement

    return input_array
