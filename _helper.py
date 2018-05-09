from numpy import ndarray


def _check_input_array(input_array):
    """
    Check if the input array is a 1-dimensional numpy array.

    """

    if type(input_array) is not ndarray:
        raise TypeError("Parameter 'input_array' must be a numpy array")
    elif input_array.ndim != 1:
        raise TypeError("Parameter 'input_array' must be 1 dimensional")


def type_as_string(obj):
    """
    Takes an object and returns its type as string.

    :param obj:     Any object
    :return:        Type of the object as a string value.

    """

    return str(type(obj))[8:-2]
