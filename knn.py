import numpy as np
import pandas as pd
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from _input_checks import check_integer
from _input_checks import check_list_of_strings
from _input_checks import check_larger
from _input_checks import check_numpy_array_pandas_dataframe_series_1d
from _input_checks import check_pandas_dataframe_nd


def get_best_k(X, y, max_k=30, keep_best_n=10, weights=None):

    # TODO: check X, y. description

    # Set default values
    if max_k is None:
        max_k = len(X)

    if weights is None:
        weights = ['uniform', 'distance']

    # Make weights into a list if it is not already one
    if type(weights) is not list:
        weights = [weights]

    # Check if inputs are valid
    check_pandas_dataframe_nd(X, 'X')

    check_numpy_array_pandas_dataframe_series_1d(y, 'y')

    check_list_of_strings(weights, 'weights')

    check_integer(max_k, 'max_k')
    check_larger(max_k, 'max_k', 1)

    check_integer(keep_best_n, 'keep_best_n')
    check_larger(keep_best_n, 'keep_best_n', 1)

    # Change shape of y if necessary
    y = np.array(y)
    y = y.ravel()

    # Split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Get value for max_k
    max_k = min(max_k, len(X_test))

    # Set up results-list
    best_model = []

    for k in range(1, max_k):
        for weight in weights:
            model = KNeighborsClassifier(n_neighbors=k, weights=weight).fit(X_train, y_train)
            score = model.score(X_test, y_test)
            best_model.append((k, weight, score))

    best_model.sort(key=lambda x: x[2], reverse=True)
    best_model = best_model[0: keep_best_n]
    return best_model
