import numpy as np
from _helper import _check_numpy_1d
from type_ops import is_type_homogeneous
from type_ops import get_contained_types


def match_by_pattern(input_array, patterns):

    _check_numpy_1d(input_array)

    if type(patterns) is not list:
        patterns = [patterns]

    if not is_type_homogeneous(input_array, verbose=False):
        raise TypeError("Input array contains multiple types ({0}) but must be type homogeneous.".format(
            get_contained_types(input_array, unique=True, as_string=True)
        ))

    matched_array = [value in patterns for value in input_array]

    return matched_array

