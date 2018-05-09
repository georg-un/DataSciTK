import numpy as np
import pandas as pd
from _helper import _check_input_array
from type_ops import *
from utils import match_by_pattern





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
    _check_input_array(input_array)

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

    _check_input_array(input_array)

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


def is_within_range(input_array, lower_bound, upper_bound, verbose=True):
    """
    Check whether the values contained in the input array are within the specified range

    :param input_array:     1-dimensional numpy array
    :param lower_bound:     number in the same type as the values in the input_array
    :param upper_bound:     number in the same type as the values in the input_array
    :param verbose:         True or False. Prints verbose output if set to True.

    :return:                True or False

    """

    # Check if inputs are valid
    _check_input_array(input_array)

    if not isinstance(input_array[0], type(lower_bound)):
        raise TypeError('Type of lower bound ({0}) must match type of the input array values ({1}).'.format(
            type(lower_bound),
            type(input_array[0])))

    if not isinstance(input_array[0], type(upper_bound)):
        raise TypeError('Type of upper bound ({0}) must match type of the input array values ({1}).'.format(
            type(upper_bound),
            type(input_array[0])))

    # Check if values are within range
    within_range = True  # True by default

    if input_array[~np.isnan(input_array)].max() > upper_bound:
        if verbose:
            print('Input array contains values larger than the upper bound.')
        within_range = False
    if input_array[~np.isnan(input_array)].min() < lower_bound:
        if verbose:
            print('Input array contains values smaller than the lower bound.')
        within_range = False

    return within_range


def fulfills_assumptions(input_array, verbosity, **assumptions):
    """
    Check the input array for a variable number of assumptions. Return true if all assumptions are fulfilled.


    Sample call:

    assumptions = {'contains_types':'int','contains_nan':False, 'type_homogeneous':True, 'variable_type':'metric', 'restrictions':[0, 90]})
    fulfills_assumptions(my_array, verbosity='low', **assumptions)

    or:

    fulfills_assumptions(my_array, verbosity='high', **{'contains_nan':True, 'contains_types':['int', 'str']})


    :param input_array:     1-dimensional numpy array
    :param verbosity:       'none', 'low' or 'high'. Sets the verbosity level.
    :param assumptions:     dictionary. Must contain at least one of the following keys:

                            contains_types:     String or list of strings. Checks if these types are contained.
                            type_homogeneous:   True or False. Checks for type homogeneity.
                            contains_nan:       True or False. Check for NaN's.
                            variable_type:      'categorical' or 'metric'. Needed for restrictions processing.
                            restrictions:       List of all categories if categorical. List of lower and upper bound if metric.

    :return:                True or False.

    """

    # Specify private local variables
    __allowed_parameters = ['contains_types', 'type_homogeneous', 'contains_nan', 'variable_type', 'restrictions']
    __allowed_variable_types = ['categorical', 'metric']
    __allowed_verbosity = ['none', 'low', 'high']

    # Check if input array is valid
    _check_input_array(input_array)

    # Check for illegal parameters
    for key in assumptions.keys():
        if key not in __allowed_parameters:
            raise TypeError("Parameter '{0}' is not allowed. Please use only the following parameters: {1}".format(
                key, __allowed_parameters,
                'See help for further information.'
            ))

    # Check if at least one assumption is specified
    if not any([key in __allowed_parameters for key in assumptions.keys()]):
        raise TypeError('No assumption is specified. Please specify at least one assumption.',
                        'See help for further information.')

    # Make sure, 'variable_type' is defined if 'restrictions' are passed
    if 'restrictions' in assumptions.keys() and 'variable_type' not in assumptions.keys():
        raise TypeError("Parameter 'variable_type' must be defined, if parameter 'restrictions' is used.",
                        'See help for further information')

    # Check if 'variable_type' is valid
    if 'variable_type' in assumptions.keys() and assumptions['variable_type'] not in __allowed_variable_types:
        raise TypeError("Value for 'variable type' ({0}) is not valid.".format(assumptions['variable_type']),
                        "Please use only one of the following strings for parameter 'variable_type': {0}".format(
                            __allowed_variable_types
                        ))

    # Check if 'verbosity' is valid
    if type(verbosity) is not str:
        raise TypeError("Parameter 'verbosity' must be a string. Please use one of the following strings: {0}.".format(
            __allowed_verbosity
        ))
    elif verbosity not in __allowed_verbosity:
        raise TypeError("Illegal value has been used for parameter 'verbosity': {0}.".format(verbosity),
                        "Please use only one of the following strings: {0}.".format(__allowed_verbosity))

    # Create result dictionary
    results = {}

    # Check if array contains specified types
    if 'contains_types' in assumptions.keys():
        results['contains_types'] = contains_types(input_array, assumptions['contains_types'],
                                                   exclusively=True, verbose=(verbosity == 'high'))

    # Check if array is type homogeneous
    if 'type_homogeneous' in assumptions.keys():
        if assumptions['type_homogeneous']:
            results['type_homogeneous'] = (is_type_homogeneous(input_array, verbose=(verbosity == 'high')) == assumptions['type_homogeneous'])
        else:
            results['not_type_homogeneous'] = (is_type_homogeneous(input_array, verbose=(verbosity == 'high')) == assumptions['type_homogeneous'])

    # Check if array contains NaN values
    if 'contains_nan' in assumptions.keys():
        if assumptions['contains_nan']:
            results['contains_nan'] = (contains_nan(input_array) == assumptions['contains_nan'])
        else:
            results['contains_no_nan'] = (contains_nan(input_array) == assumptions['contains_nan'])

    # Check if restrictions hold
    if 'restrictions' in assumptions.keys():

        # Check if variable is categorical or boolean
        if assumptions['variable_type'] == 'categorical':
            results['restrictions'] = contains_category(input_array, assumptions['restrictions'],
                                                        exclusively=True, verbose=(verbosity == 'high'))

        # Check if variable is metric
        elif assumptions['variable_type'] == 'metric':
            results['restrictions'] = is_within_range(input_array,
                                                      lower_bound=min(assumptions['restrictions']),
                                                      upper_bound=max(assumptions['restrictions']),
                                                      verbose=(verbosity == 'high'))

        # Raise exception if variable was neither categorical, metric or boolean
        else:
            raise IOError('Variable type was neither categorical, metric or boolean. This may not be your fault.')

    # Summarize results
    all_true = all(results.values())

    # Print summary if 'verbose' is True
    if verbosity == 'low' or verbosity == 'high':
        if all_true:
            print('\nAll tests have been passed successfully:')
        else:
            print('\nWarning: Some tests have failed. Please see the test results below:')

        for key, value in results.items():
            print('{0}: {1}'.format(key, value))
        print('\n')

    # Return result
    return all_true
