from numpy import ndarray
from numpy import int16, int32, int64, float16, float32, float64
from pandas import DataFrame
from pandas import Series


def check_numpy_array_1d(data, name):
    """
    Check if the input array is a 1-dimensional numpy array.

    """

    if type(data) is not ndarray:
        raise TypeError("Parameter '{0}' must be a numpy array".format(
            str(name)
        ))
    elif data.ndim != 1:
        raise TypeError("Parameter '{0}' must be 1 dimensional".format(
            str(name)
        ))


def check_pandas_dataframe_1d(data, name):
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


def check_pandas_dataframe_nd(data, name):
    """
    Check if input is a pandas DataFrame.

    """
    if type(data) is not DataFrame:
        raise TypeError("Argument for '{0}' must be a pandas DataFrame.".format(
            str(name)
        ))


def check_numpy_array_pandas_series_1d(data, name):
    """
    Check if the input is a 1-dimensional numpy array or pandas Series.

    """
    if type(data) is not ndarray and type(data) is not Series:
        raise TypeError("Input for '{0}' must be either a numpy ndarray or a pandas Series.".format(
            str(name)
        ))
    elif data.ndim != 1:
        raise TypeError("Input for '{0}' must be 1-dimensional.".format(
            str(name)
        ))


def check_numpy_array_pandas_dataframe_series_1d(data, name):
    """
    Check if the input is a 1-dimensional numpy array, a pandas DataFrame with shape (n, 1)
    or a 1-dimensional pandas Series.

    """
    if type(data) is not ndarray and type(data) is not DataFrame and type(data) is not Series:
        raise TypeError("'{0}' must be either a numpy ndarray, a pandas DataFrame or a pandas Series.".format(
            str(name)
        ))
    elif type(data) is ndarray and data.ndim != 1:
        raise TypeError("'{0}' must be a 1 dimensional numpy array or a pandas DataFrame with shape (m, 1) or a 1 dimensional pandas Series.".format(
            str(name)
        ))
    elif type(data) is DataFrame and data.shape[1] != 1:
        raise TypeError("'{0}' must be a 1 dimensional numpy array or a pandas DataFrame with shape (m, 1) or a 1 dimensional pandas Series.".format(
            str(name)
        ))
    elif type(data) is Series and data.ndim != 1:
        raise TypeError("'{0}' must be a 1 dimensional numpy array or a pandas DataFrame with shape (m, 1) or a 1 dimensional pandas Series.".format(
            str(name)
        ))


def check_string(data, name):
    """
    Check if input is a string.

    """
    if type(data) is not str:
        raise TypeError("Argument for parameter '{0}' must be of type string but is currently of type {1}.".format(
            str(name),
            type(data)
        ))


def check_list_of_strings(data, name):
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


def check_list_numpy_array(data, name):
    """
    Check if input is a list.

    """
    if type(data) is not list and not isinstance(data, ndarray):
        raise TypeError("Argument for parameter '{0}' must be of type list but is of type {1}.".format(
            str(name),
            str(type(data))
        ))


def check_boolean(data, name):
    """
    Check if input is boolean.

    """
    if type(data) is not bool:
        raise TypeError("Argument for parameter '{0}' must be of type boolean but is currently of type {1}.".format(
            str(name),
            type(data)
        ))


def check_integer(data, name):
    """
    Check if input is an integer.

    """
    if type(data) is not int and type(data) is not int64:
        raise TypeError("Argument for parameter '{0}' must be of type integer but is currently of type {1}.".format(
            str(name),
            type(data)
        ))


def is_integer(data):
    """
    Check if input is of type integer.

    """
    if type(data) is int or type(data) is int16 or type(data) is int32 or type(data) is int64:
        return True
    else:
        return False


def check_float(data, name):
    """
    Check if input is of type float.

    """
    if not is_float(data):
        raise TypeError("Argument for parameter {0} must be of type float but is currently of type {1].".format(
            str(name),
            type(data)
        ))


def is_float(data):
    """
    Check if input is of type float.

    """
    if type(data) is float or type(data) is float16 or type(data) is float32 or type(data) is float64:
        return True
    else:
        return False


def is_numeric(data):
    """
    Check if input is either of type int or of type float.

    """
    if is_integer(data) or is_float(data):
        return True
    else:
        return False


def check_numeric(data, name):
    """
    Check if input is either of type in or of type float.

    """
    if not is_numeric(data):
        raise TypeError("Argument of parameter '{0}' must be numeric (integer or floar) but is of type {1}.".format(
            str(name),
            str(type(data))
        ))


def check_larger(data, name, min_value):
    """
    Check if input is larger than min_value.

    """
    if data < min_value:
        raise TypeError("Argument for parameter '{0}' must be {1} or larger.".format(
            str(name),
            str(min_value)
        ))

