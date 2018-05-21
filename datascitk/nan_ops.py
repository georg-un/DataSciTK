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
""" NaN operations module """

import numpy as np
import pandas as pd

from _input_checks import check_numpy_array_1d
from utils import match_by_pattern
from type_ops import contains_types
from type_ops import match_by_type
from type_ops import get_contained_types
from type_ops import is_type_homogeneous


def count_nan(data):
    """
    Count number of NaN's in the data. Works also for an array of strings.

    :param data:                1-dimensional numpy array
    :return:                    Integer. Number of NaN's found.

    """

    # Check if input is valid
    check_numpy_array_1d(data, 'data')

    nans = 0

    # Check string values for NaN's
    if contains_types(data, 'str', exclusively=False, verbose=False):
        string_values = data[match_by_type(data, 'str')]
        string_values = string_values[match_by_pattern(string_values, ['nan', 'NaN', 'NAN', 'N/A'])]
        nans += string_values.size

    # Check non-string values for NaN's
    if not contains_types(data, 'str', exclusively=True, verbose=False):
        non_string_values = data[[type(element) is not str for element in data]]
        try:
            nans += np.sum(pd.isnull(non_string_values))
        except TypeError:
            types = get_contained_types(data, unique=True, as_string=True)
            raise TypeError("Argument for 'data' contains types which cannot be checked for NaN. Found types are: {0}.".format(
                types
            ))

    return nans


def contains_nan(data):
    """
    Check if data contains NaN values. Works for arrays of strings as well.

    :param data:            1-dimensional numpy array
    :return:                True or False

    """

    # Get number of NaN's
    nans = count_nan(data)

    # Return True if NaN's have been found
    if nans is None:
        raise IOError("Did not get the number of NaN's. Please check function count_nan.")
    elif nans > 0:
        return True
    elif nans == 0:
        return False
    else:
        raise IOError("Number of NaN's was not zero or larger: {0}".format(nans))


def recode_nan_binary(data):
    """
    Replaces the NaN's in the data by a 1 and all other values by a 0.

    :param data:            1-dimensional numpy array
    :return:                1-dimensional numpy array with a 1 for every NaN and a 0 for every other value

    """

    # Check if input is valid
    check_numpy_array_1d(data, 'data')

    if not is_type_homogeneous(data, verbose=False):
        raise TypeError("Argument for 'data' contains multiple types ({0}). Please use only type homogeneous types.".format(
            get_contained_types(data, unique=True, as_string=True)
        ))

    if not contains_nan(data):
        raise TypeError("Argument for 'data' does not contain any NaN. All result values would be zero.")

    # Find NaN's
    if isinstance(data[0], str):
        binary_array = match_by_pattern(data, ['nan', 'NaN', 'NAN', 'N/A'])
    else:
        binary_array = [np.isnan(value) for value in data]

    # convert from True and False to 1 and 0
    binary_array = np.asarray([int(value) for value in binary_array])

    return binary_array
