import numpy as np
from _helper import _check_numpy_array_1d
from _helper import type_as_string


def get_contained_types(input_array, unique=True, as_string=True):
    """
    Gets all types in the input array
    :param input_array:     1-dimensional numpy array
    :param unique:          True or False. If true types are returned uniquely as strings.
    :param as_string:       True or False. Types are either returned as string or as type.

    :return:                1-dimensional numpy array containing either strings or types.

    """

    # Check if inputs are valid
    _check_numpy_array_1d(input_array)

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
    _check_numpy_array_1d(input_array)

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


def match_by_type(input_array, match_types):
    """
    Searches for all values in the input string which match one of the given types. Returns a array in the same length
    as the input array containing True and False values for the indexes where matches have been found or not.

    :param input_array:     1-dimensional numpy array
    :param match_types:     String or list of strings. Defines the types which should be matched, e.g. ['int', 'float']

    :return:                1-dimensional numpy array containing True and False values for the indexes of the matches

    """

    # Check if inputs are valid
    _check_numpy_array_1d(input_array)

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
