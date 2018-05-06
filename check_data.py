import numpy as np


def _check_input_array(input_array):
    """
    Check if the input array is a 1-dimensional numpy array.

    """

    if type(input_array) is not np.ndarray:
        raise TypeError("Parameter 'input_array' must be a numpy array")
    elif input_array.ndim != 1:
        raise TypeError("Parameter 'input_array' must be 1 dimensional")


def get_contained_types(input_array, unique=True, as_string=True):
    """
    Gets all types in the input array
    :param input_array:     1-dimensional numpy array
    :param unique:          True or False. If true types are returned uniquely as strings.
    :param as_string:       True or False. Types are either returned as string or as type.

    :return:                1-dimensional numpy array containing either strings or types.
    """

    # Check if inputs are valid
    _check_input_array(input_array)

    if type(unique) is not bool:
        raise TypeError("Parameter 'unique' must be boolean (True or False).")

    if type(as_string) is not bool:
        raise TypeError("Parameter 'as_string' must be boolean (True or False).")

    if unique and not as_string:
        raise TypeError("Parameter 'as_string' cannot be False as long as parameter 'unique' is True.")

    # Create a list with all the types in the input array
    if as_string:
        types_found = [str(type(element))[8:-2] for element in input_array]
    else:
        types_found = [type(element) for element in input_array]

    if unique:
        types_found = np.unique(types_found)

    return types_found


def contains_types(input_array, types, exclusively=False, verbose=True):
    """
    Check if the input array contains certain types. If exclusively is set to True, check if the input array
    contains ONLY the specified types.

    :param input_array:         1-dimensional numpy array
    :param types:               string or list of strings. Specifies the types (e.g. ['str', 'int']
    :param exclusively:         True or False. If set to True, check if ONLY the specified types are present
    :param verbose:             True or False. Set to true for verbose output.

    :return:                    True or False
    """

    # Make sure parameter 'types' is a list
    if type(types) is not list:
        types = [types]

    # Check if inputs are valid
    if any([type(element) is not str for element in types]):
        raise TypeError("Parameter 'types' contains non-string values. Please use a string or a list of strings.")

    if type(exclusively) is not bool:
        raise TypeError("Parameter 'exclusively' must be boolean (True or False).")

    if type(verbose) is not bool:
        raise TypeError("Parameter 'verbose' must be boolean (True or False).")

    # Get types in input array
    contained_types = get_contained_types(input_array, unique=True, as_string=True)

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

    :param input_array:     1-dimensional numpy array
    :param verbose:         True for verbose output (default)

    :return:                True or False. True if all values have the same type.
    """

    # Check if input is valid
    if type(verbose) is not bool:
        raise TypeError("Parameter 'verbose' must be boolean (True or False).")

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


def contains_nan(input_array):
    """
    Check if array contains NaN values. Works for arrays of strings as well.

    :param input_array:     1-dimensional numpy array
    :return:                True or False

    """

    # Check if input is valid
    _check_input_array(input_array)

    # Check string values for NaN
    if contains_types(input_array, 'str', exclusively=False, verbose=False):
        string_values = input_array[[type(element) is str for element in input_array]]
        if string_values[string_values == 'nan'].size > 0:
            return True
        elif string_values[string_values == 'NaN'].size > 0:
            return True
        elif string_values[string_values == 'NAN'].size > 0:
            return True

    # Check non-string values for NaN
    if not contains_types(input_array, 'str', exclusively=True, verbose=False):
        non_string_values = input_array[[type(element) is not str for element in input_array]]
        try:
            if any(np.isnan(non_string_values)):
                return True
        except TypeError:
            types = get_contained_types(input_array, unique=True, as_string=True)
            print("Warning: Input array contains types which cannot be checked for NaN. Found types are: {0}.".format(
                types
            ))
            return True  # better save than sorry..

    # If no matches have been found
    return False


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
