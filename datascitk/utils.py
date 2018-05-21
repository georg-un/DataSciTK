# Copyright (C) 2018 Georg Unterholzner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ==============================================================================
""" Utilities module """

import numpy as np

from _input_checks import check_numpy_array_1d
from _input_checks import check_numpy_array_pandas_series_1d
from _input_checks import check_pandas_dataframe_nd
from type_ops import is_type_homogeneous
from type_ops import get_contained_types


def match_by_pattern(data, patterns):
    """
    Find all occurrences in the data which are defined in patterns. Returns a list of True and False values.

    :param data:                1-dimensional numpy array or pandas Series.
    :param patterns:            Single value or list of values. Must be of the same type as the values in the data.

    :return:                    1-dimensional numpy array containing True and False values.

    """

    # Make patterns to a list if it is not already one
    if type(patterns) is not list:
        patterns = [patterns]

    # Check if inputs are valid
    check_numpy_array_pandas_series_1d(data, 'data')

    if not is_type_homogeneous(data, verbose=False):
        raise TypeError("Argument for parameter 'data' contains multiple types ({0}) but must be type homogeneous.".format(
            get_contained_types(data, unique=True, as_string=True)
        ))

    # Get matches
    matched_array = [value in patterns for value in data]

    return matched_array


def contains_negatives_or_zero(data):
    """
    Checks if the data contains only values larger than 0. Returns True or False.

    :param data:                1-dimensional numpy array or pandas Series of data-type float or int.
    :return:                    True if the data contains only values larger than zero. False otherwise.
    """

    # Check if input is valid
    check_numpy_array_pandas_series_1d(data, 'data')

    if not is_type_homogeneous(data, verbose=False):
        raise TypeError("Argument for 'data' must be type homogeneous but contains multiple types: {0}.".format(
            get_contained_types(data, unique=True, as_string=True)
        ))

    if type(data[0]) is not int and type(data[0]) is not float and type(data[0]) is not np.float:
        if type(data[0]) is not np.float64 and type(data[0]) is not np.int64:
            raise TypeError("Argument for 'data' must contain integer or float values but contains type {0}.".format(
                type(data[0])
            ))

    # Count number of negatives
    zero_or_smaller = data[data <= 0]

    # Check number of negatives and return respective result
    if zero_or_smaller.size == 0:
        return False
    elif zero_or_smaller.size > 0:
        return True
    else:
        raise IOError("Number of values which are zero or smaller ({0}) was neither 0 nor a number larger than 0.".format(
            zero_or_smaller
        ))


def get_columns_larger_zero(data):
    # TODO: description
    check_pandas_dataframe_nd(data, 'data')

    # Initialize result list
    columns_larger_zero = []

    for column in data:
        if not contains_negatives_or_zero(data[column]):
            columns_larger_zero.append(column)

    return columns_larger_zero


