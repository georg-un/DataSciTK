from numpy import ndarray
from numpy import int64, float32, float64
from pandas import DataFrame
from pandas import Series


def _check_numpy_array_1d(input_array):
    """
    Check if the input array is a 1-dimensional numpy array.

    """

    if type(input_array) is not ndarray:
        raise TypeError("Parameter 'input_array' must be a numpy array")
    elif input_array.ndim != 1:
        raise TypeError("Parameter 'input_array' must be 1 dimensional")


def _check_pandas_dataframe_1d(data, name):
    """
    Check if the input is a 1-dimensional pandas DataFrame.

    """

    if type(data) is not DataFrame:
        raise TypeError("Object for parameter '{0}' must be of type pd.DataFrame but is currently of the type {1}.".format(
            str(name),
            type(data)
        ))
    elif data.shape[1] != 1:
        raise TypeError("Data frame for parameter '{0}' does not have the right shape ({1}). It should have a shape of (n, 1).".format(
            str(name),
            data.shape
        ))


def _check_pandas_dataframe_nd(data):
    """
    Check if input is a pandas DataFrame.

    """
    if type(data) is not DataFrame:
        raise TypeError("Argument for 'data' must be a pandas DataFrame.")


def _check_numpy_array_pandas_series_1d(input_array):
    """
    Check if the input is a 1-dimensional numpy array or pandas Series.

    """
    if type(input_array) is not ndarray and type(input_array) is not Series:
        raise TypeError("Input for 'input_array' must be either a numpy ndarray or a pandas Series.")
    elif input_array.ndim != 1:
        raise TypeError("Input for 'input_array' must be 1-dimensional.")


def _check_numpy_array_pandas_dataframe_series_1d(input_array):
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


def _check_string(data, name):
    """
    Check if input is a string.

    """
    if type(data) is not str:
        raise TypeError("Argument for parameter '{0}' must be of type string but is currently of type {1}.".format(
            str(name),
            type(data)
        ))


def _check_list_of_strings(data, name):
    """
    Check if input is a list of strings.

    """
    if type(data) is not list:
        raise TypeError("Argument for parameter '{0}' must be a list of strings.".format(
            str(name)
        ))
    else:
        for element in data:
            if type(element) is not str:
                raise TypeError("At least one value in the list for the parameter '{0}' is not of type string: {1}.".format(
                    str(name),
                    element
                ))


def _check_integer(data, name):
    """
    Check if input is an integer.

    """
    if type(data) is not int and type(data) is not int64:
        raise TypeError("Argument for parameter '{0}' must be of type integer but is currently of type {1}.".format(
            str(name),
            type(data)
        ))


def _check_boolean(data, name):
    """
    Check if input is boolean.

    """
    if type(data) is not bool:
        raise TypeError("Argument for parameter '{0}' must be of type boolean but is currently of type {1}.".format(
            str(name),
            type(data)
        ))


def _check_float(data, name):
    """
    Check if input is of type float.

    """
    if type(data) is not float and type(data) is not float64 and type(data) is not float32:
        raise TypeError("Argument for parameter {0} must be of type float but is currently of type {1].".format(
            str(name),
            type(data)
        ))









def type_as_string(obj):
    """
    Takes an object and returns its type as string.

    :param obj:     Any object
    :return:        Type of the object as a string value.

    """

    return str(type(obj))[8:-2]