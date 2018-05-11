import numpy as np

from _helper import _check_numpy_1d
from type_ops import is_type_homogeneous
from type_ops import get_contained_types

# TODO: implement function similar to recode_nan_binary for multiple categories e.g. [1,2,3] = 1, [4,5,6] = 0


def contains_category(input_array, categories, exclusively=False, verbose=True):
    """
    Check if all specified categories are present in the input array. If exclusively is set to True (default), check
    if ONLY the specified categories are present in the input array and return False if additional categories are found.

    :param input_array:     1-dimensional numpy array.
    :param categories:      Single value or list. Must be of the same type as the values in the input array.
    :param exclusively:     True or False. Checks if the input array contains exclusively the specified categories.
    :param verbose:         True or False. Prints additional information to console if True.

    :return:                True or False.

    """

    # Transform categories parameter to list if it is not already one
    if type(categories) is not list:
        categories = [categories]

    # Check if inputs are valid
    _check_numpy_1d(input_array)

    for category in categories:
        if not isinstance(input_array[0], type(category)):
            raise TypeError("Type of category '{0}' ({1}) must match type of the input array values ({2}).".format(
                category,
                str(type(category))[8:-2],
                str(type(input_array[0]))[8:-2]
            ))

    if type(exclusively) is not bool:
        raise TypeError("Parameter 'exclusively' must be boolean (True or False).")

    if not is_type_homogeneous(input_array, verbose=False):
        raise TypeError('Input array must be type homogeneous but contains values with different types: {0}.'.format(
            get_contained_types(input_array, unique=True, as_string=True)
        ))

    # Get all unique categories in the input array
    input_array_categories = np.unique(input_array)

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


def count_elements_with_category(input_array, categories, verbose=False):
    """
    Counts all observations in the input array which match the given category. Returns the sum of it.

    :param input_array:     1-dimensional numpy array
    :param categories:      List or single value. Must match the type of the values in the input array.
    :param verbose:         True or False. True for verbose output.

    :return:                Integer. Number of found occurrences.

    """

    _check_numpy_1d(input_array)

    # Convert category to a list, if it is not already one
    if type(categories) is not list:
        categories = [categories]

    # Check for type homogeneity
    if not is_type_homogeneous(input_array, verbose=False):
        raise TypeError("Input array contains values with different types {0}. Please use only type homogeneous arrays.".format(
            get_contained_types(input_array, unique=True, as_string=True)
        ))

    # Check if types of input array and category-argument match
    for category in categories:
        if not isinstance(category, type(input_array[0])):
            raise TypeError("Type of 'category' ({0}) does not match type of values in 'input_array' ({1}).".format(
                type(category),
                type(input_array[0])
            ))  # TODO: maybe add automatic conversion in the future

    # Find matches for each category, get the sum of occurrences and add the sums of all categories together
    sum_found_observations = 0
    for category in categories:
        found_observations = np.sum(input_array[input_array == category])
        if verbose:
            print("Found {0} observations of the category '{1]'.".format(found_observations, category))
        sum_found_observations += found_observations

    if verbose:
        print("Found {0} matching observations in total.".format(sum_found_observations))

    return sum_found_observations
