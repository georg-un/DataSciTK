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
""" Type operations module"""

import numpy as np

from _input_checks import check_numpy_array_pandas_series_1d
from _input_checks import check_boolean
from _input_checks import check_list_of_strings

from _helper import type_as_string


def get_contained_types(data, unique=True, as_string=True):
    """
    Gets all types in the input array

    :param data:            1-dimensional numpy array or pandas Series
    :param unique:          True or False. If true types are returned uniquely as strings.
    :param as_string:       True or False. Types are either returned as string or as type.

    :return:                1-dimensional numpy array containing either strings or types.

    """

    # Check if inputs are valid
    check_numpy_array_pandas_series_1d(data, 'data')

    check_boolean(unique, 'unique')

    check_boolean(as_string, 'as_string')

    if unique and not as_string:
        raise TypeError("Parameter 'as_string' cannot be False as long as parameter 'unique' is True.")

    # Create a list with all the types in the input array
    if as_string:
        types_found = [str(type(element))[8:-2] for element in data]
    else:
        types_found = [type(element) for element in data]

    if unique:
        types_found = np.unique(types_found)

    return types_found


def contains_types(data, types, exclusively=False, verbose=True):
    """
    Check if the input array contains certain types. If exclusively is set to True, check if the input array
    contains ONLY the specified types.

    :param data:                1-dimensional numpy array or pandas Series
    :param types:               string or list of strings. Specifies the types (e.g. ['str', 'int']
    :param exclusively:         True or False. If set to True, check if ONLY the specified types are present
    :param verbose:             True or False. Set to true for verbose output.

    :return:                    True or False

    """

    # Make sure parameter 'types' is a list
    if type(types) is not list:
        types = [types]

    # Check if inputs are valid
    check_numpy_array_pandas_series_1d(data, 'data')

    check_list_of_strings(types, 'types')

    check_boolean(exclusively, 'exclusively')

    check_boolean(verbose, 'verbose')

    # Get types in input array
    contained_types = get_contained_types(data, unique=True, as_string=True)

    # Check if all types can be found
    types_found = [element in contained_types for element in types]
    if all(types_found):
        result = True
    else:
        if verbose:
            for index, found in enumerate(types_found):
                if not found:
                    print("Type '{0}' has not been found.".format(types[index]))
        result = False

    # Check if additional types are present if exclusively is set to True
    if exclusively:
        additional_types = [element in types for element in contained_types]
        additional_types = np.invert(additional_types)
        if any(additional_types):
            if verbose:
                for index, additional_type in enumerate(additional_types):
                    if additional_type:
                        print("Additional type '{0}' has been found.".format(contained_types[index]))
            result = False

    return result


def is_type_homogeneous(input_array, verbose=True):
    """
    Check if all values of the input array have the same type.

    :param input_array:     1-dimensional numpy array or pandas Series
    :param verbose:         True for verbose output (default)

    :return:                True or False. True if all values have the same type.

    """
    check_numpy_array_pandas_series_1d(input_array, 'input_array')

    # Check if input is valid
    check_boolean(verbose, 'verbose')

    # Get types in input array
    types_found = get_contained_types(input_array, unique=True, as_string=True)

    # Check for number of types
    if types_found.size == 1:
        if verbose:
            print('Input array contains the following type: {0}.'.format(types_found))
        return True
    elif types_found.size > 1:
        if verbose:
            print('Input array contains the following types {0}.:'.format(types_found))
        return False
    else:
        raise IOError('No types have been found.')


def match_by_type(input_array, match_types):
    """
    Searches for all values in the input string which match one of the given types. Returns a array in the same length
    as the input array containing True and False values for the indexes where matches have been found or not.

    :param input_array:     1-dimensional numpy array or pandas Series
    :param match_types:     String or list of strings. Defines the types which should be matched, e.g. ['int', 'float']

    :return:                1-dimensional numpy array containing True and False values for the indexes of the matches

    """

    # Check if inputs are valid
    check_numpy_array_pandas_series_1d(input_array, 'input_array')

    if type(match_types) is not list:
        match_types = [match_types]

    for match_type in match_types:
        if type(match_type) is not str:
            raise TypeError("Argument for 'match_type' must be a string or list of strings, but is of type {0}.".format(
                type(match_type)
            ))

    # Get matches
    matched_array = [type_as_string(value) in match_types for value in input_array]

    return matched_array
