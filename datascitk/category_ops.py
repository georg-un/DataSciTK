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
""" Category operations module """

import numpy as np
import pandas as pd

from _input_checks import check_numpy_array_1d
from _input_checks import check_numpy_array_pandas_series_1d
from _input_checks import check_pandas_dataframe_nd
from _input_checks import check_boolean
from _input_checks import check_list_numpy_array
from _input_checks import is_numeric
from _input_checks import is_float
from _input_checks import is_integer

from type_ops import is_type_homogeneous
from type_ops import get_contained_types
from type_ops import type_as_string


def convert_to_binary_columns(data, columns, delete_category_cols=True):

    # TODO checks & description

    result_data = data.copy()

    for column in columns:
        # Get all contained categories
        categories = get_contained_categories(np.array(data[[column]]).ravel())
        for number, category in enumerate(categories):
            if number != 0:
                # Recode each single category binary
                other_categories = categories[categories != category]
                result_data[str(column) + '_' + str(category)] = recode_binary_by_categories(data=np.array(data[column]).ravel(),
                                                                                             to_0=other_categories, to_1=[category],
                                                                                             verbose=False)

    if delete_category_cols:
        result_data = result_data.drop(columns=columns)

    return result_data


def recode_binary_by_categories(data, to_0, to_1, verbose=False):
    """
    Recode a numpy array or pandas Series to 0 and 1 according to two lists of categories.

    :param data:                1-dimensional numpy array or pandas Series
    :param to_0:                List. Categories which should be coded to 0.
    :param to_1:                List. Categories which should be coded to 1.
    :param verbose:             True or False. If true a warning is printed, if some category is not found in data.

    :return:                    1-dimensional numpy array containing only 1 and 0.

    """

    # Check if inputs are valid
    check_numpy_array_pandas_series_1d(data, 'data')

    check_boolean(verbose, 'verbose')

    check_list_numpy_array(to_0, 'to_0')

    check_list_numpy_array(to_1, 'to_1')

    # Get contained categories
    contained_categories = get_contained_categories(data)

    # Check if all categories are defined in to_0 and to_1
    for category in contained_categories:
        if category not in to_0 and category not in to_1:
            raise TypeError("Argument for 'data' contains the category '{0}' which is neither defined in to_0 or to_1.".format(
                category
            ))

    # Check if categories are defined for both lists
    for category in to_0:
        if category in to_1:
            raise TypeError("Category '{0}' is defined in to_0 and to_1. A category must not be contained in both lists.".format(
                category
            ))

    # Print warning if one of the defined categories has not been found in data
    if verbose:
        for category in to_0:
            if category not in contained_categories:
                print("Info: Category '{0}' from to_0 has not been found in data.".format(category))
        for category in to_1:
            if category not in contained_categories:
                print("Info: Category '{0}' from to_1 has not been found in data.".format(category))

    # Copy data to binary array
    binary_array = data.copy()

    # Loop over array and recode to 0 and 1
    for index, value in enumerate(data):
        if value in to_0:
            binary_array[index] = 0
        elif value in to_1:
            binary_array[index] = 1
        else:
            raise IOError("Value '{0}' was neither in to_0 nor in to_1.".format(value))

    return binary_array


def get_contained_categories(data):
    """
    Get an array of all contained categories in a numpy array or pandas Series.

    :param data:                1-dimensional numpy array or pandas Series
    :return:                    1-dimensional numpy array containing all unique categories of the input

    """

    # Check if input is valid
    check_numpy_array_pandas_series_1d(data, 'data')

    # Get all unique values (= categories) and return them
    categories = pd.unique(data)
    return categories


def contains_category(data, categories, exclusively=False, verbose=True):
    """
    Check if all specified categories are present in the data. If exclusively is set to True (default), check
    if ONLY the specified categories are present in the data and return False if additional categories are found.

    :param data:            1-dimensional numpy array.
    :param categories:      Single value or list. Must be of the same type as the values in the data.
    :param exclusively:     True or False. Checks if the data contains exclusively the specified categories.
    :param verbose:         True or False. Prints additional information to console if True.

    :return:                True or False.

    """

    # Transform categories parameter to list if it is not already one
    if type(categories) is not list:
        categories = [categories]

    # Check if inputs are valid
    #check_numpy_array_1d(data, 'data')  # TODO: pandas series or df possible as well?
    check_numpy_array_pandas_series_1d(data, 'data')

    for category in categories:
        if is_float(data[0]) and not is_float(category):
            raise TypeError("Type of category '{0}' ({1}) must match type of the values for 'data' ({2}).".format(
                category,
                type_as_string(category),
                type_as_string(data[0])
            ))
        if is_integer(data[0]) and not is_integer(category):
            raise TypeError("Type of category '{0}' ({1}) must match type of the values for 'data' ({2}).".format(
                category,
                type_as_string(category),
                type_as_string(data[0])
            ))
        elif not is_float(data[0]) and not is_integer(data[0]) and not isinstance(data[0], type(category)):
            raise TypeError("Type of category '{0}' ({1}) must match type of the values for 'data' ({2}).".format(
                category,
                type_as_string(category),
                type_as_string(data[0])
            ))

    check_boolean(exclusively, 'exclusively')

    if not is_type_homogeneous(data, verbose=False):
        raise TypeError("Argument for 'data' must be type homogeneous but contains values with different types: {0}.".format(
            get_contained_types(data, unique=True, as_string=True)
        ))

    # Get all unique categories in data
    input_array_categories = get_contained_categories(data)

    # Check if all categories can be found
    categories_found = [x in input_array_categories for x in categories]
    if all(categories_found):
        result = True
    else:
        if verbose:
            for index, found in enumerate(categories_found):
                if not found:
                    print("Category '{0}' has not been found.".format(categories[index]))
        result = False

    # Check if additional categories are present if exclusively is set to True
    if exclusively:
        additional_categories = [x in categories for x in input_array_categories]
        additional_categories = np.invert(additional_categories)
        if any(additional_categories):
            if verbose:
                for index, additional_category in enumerate(additional_categories):
                    if additional_category:
                        print("Additional category '{0}' has been found.".format(input_array_categories[index]))
            result = False

    return result


def count_elements_with_category(data, categories, verbose=False):
    """
    Counts all observations in 'data' which match the given category. Returns the sum of it.

    :param data:     1-dimensional numpy array
    :param categories:      List or single value. Must match the type of the values in 'data'.
    :param verbose:         True or False. True for verbose output.

    :return:                Integer. Number of found occurrences.

    """

    check_numpy_array_1d(data, 'data')

    check_boolean(verbose, 'verbose')

    # Convert category to a list, if it is not already one
    if type(categories) is not list:
        categories = [categories]

    # Check for type homogeneity
    if not is_type_homogeneous(data, verbose=False):
        raise TypeError("Argument for 'data' contains values with different types {0}. Please use only type homogeneous arrays.".format(
            get_contained_types(data, unique=True, as_string=True)
        ))

    # Check if types of data and category-argument match
    for category in categories:
        if not isinstance(category, type(data[0])):
            raise TypeError("Type of 'category' ({0}) does not match type of values in 'data' ({1}).".format(
                type(category),
                type(data[0])
            ))  # TODO: maybe add automatic conversion in the future

    # Find matches for each category, get the sum of occurrences and add the sums of all categories together
    sum_found_observations = 0
    for category in categories:
        found_observations = np.sum(data[data == category])
        if verbose:
            print("Found {0} observations of the category '{1]'.".format(found_observations, category))
        sum_found_observations += found_observations

    if verbose:
        print("Found {0} matching observations in total.".format(sum_found_observations))

    return sum_found_observations


def is_boolean(data):
    # TODO: description
    if not is_numeric(data[0]):
        return False
    elif is_integer(data[0]):
        return contains_category(data, [0, 1], exclusively=True, verbose=False)
    elif is_float(data[0]):
        return contains_category(data, [0.0, 0.1], exclusively=True, verbose=False)
    else:
        return False


def get_boolean_columns(data):
    # TODO: description
    # Check if input is valid
    check_pandas_dataframe_nd(data, 'data')

    # Initialize result list
    boolean_columns = []

    # Get all columns which are boolean
    for column in data:
        if is_boolean(data[column]):
            boolean_columns.append(column)

    return boolean_columns
