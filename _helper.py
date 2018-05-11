from numpy import ndarray
from pandas import DataFrame
from pandas import Series


def _check_numpy_1d(input_array):
    """
    Check if the input array is a 1-dimensional numpy array.

    """

    if type(input_array) is not ndarray:
        raise TypeError("Parameter 'input_array' must be a numpy array")
    elif input_array.ndim != 1:
        raise TypeError("Parameter 'input_array' must be 1 dimensional")


def _check_numpy_pandas_1d(input_array):
    """
    Check if the input is a 1-dimensional numpy array, a pandas DataFrame with shape (n, 1)
    or a 1-dimensional pandas Series.

    """
    if type(input_array) is not ndarray and type(input_array) is not DataFrame and type(input_array) is not Series:
        raise TypeError("'input_array' must be either a numpy ndarray, a pandas DataFrame or a pandas Series.")
    elif type(input_array) is ndarray and input_array.ndim != 1:
        raise TypeError("'input_array' must be a 1 dimensional numpy array or a pandas DataFrame with shape (m, 1) or a 1 dimensional pandas Series.")
    elif type(input_array) is DataFrame and input_array.shape[1] != 1:
        raise TypeError("'input_array' must be a 1 dimensional numpy array or a pandas DataFrame with shape (m, 1) or a 1 dimensional pandas Series.")
    elif type(input_array) is Series and input_array.ndim != 1:
        raise TypeError("'input_array' must be a 1 dimensional numpy array or a pandas DataFrame with shape (m, 1) or a 1 dimensional pandas Series.")


def type_as_string(obj):
    """
    Takes an object and returns its type as string.

    :param obj:     Any object
    :return:        Type of the object as a string value.

    """

    return str(type(obj))[8:-2]