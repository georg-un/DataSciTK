import numpy as np
import statsmodels.api as sm
import pandas as pd
from itertools import combinations


def brute_force_logit(y, X, variables, benchmark_criterion, keep_best_n=5, show_warning=True):
    # TODO: implement feature for automatic generation of multiplicative terms
    # TODO: implement feature for automatic generation of polynomials
    # TODO: implement some kind of optimization to enhance performance

    """
    Brute force a model with a logit regression. Create all possible model specifications of the given variables and
    run a logit regression for each one. Find the n best performing regressions in terms of the benchmark criterion
    and return them.

    e.g.
    $ brute_force_logit(y=y, X=X, variables=['a', 'b', 'c'], benchmark_criterion='aic', keep_best_n=10)
    will run a logit regression for each possible variable combination
    (y ~ a;  y ~ b;  y ~ c;  y ~ a + b;  y ~ a + c;  y ~ b + c;  y ~ a + b + c)
    and measure each AIC. Subsequently, it will return the 10 best performing regressions as dictionaries.

    :param y:                       Pandas DataFrame with the shape (m, 1). Contains the dependent variable coded as 0 and 1.
    :param X:                       Pandas DataFrame with the shape (m, x). Contains the independent variables coded numerically.
    :param variables:               List of strings. Contains the column names of the independent variables or modifications e.g. 'np.log(some_column)'
    :param benchmark_criterion:     String. Either 'aic' or 'bic'.
    :param keep_best_n:             Integer. Number of best regressions to return.
    :param show_warning:            True or False. If False than no warning will be printed for a high number of variables.

    :return:                        Dictionary of dictionaries. Contains the n best performing regressions.

    """

    if type(y) is not pd.DataFrame:
        raise TypeError("Object for parameter 'y' must be of type pd.DataFrame but is currently of the type {0}.".format(
            type(y)
        ))
    elif y.shape[1] != 1:
        raise TypeError("Data frame for parameter 'y' does not have the right shape ({0}). It should have a shape of (n, 1).".format(
            y.shape
        ))

    if type(X) is not pd.DataFrame:
        raise TypeError("Object for parameter 'X' must be of type pd.DataFrame but is currently of the type {0}.".format(
            type(X)
        ))
    elif len(X) != len(y):
        raise TypeError("Inputs for 'X' and 'y' have a different number of observations. ({0}, {1})".format(
            len(X),
            len(y)
        ))

    if type(variables) is not list:
        raise TypeError("Argument for parameter 'variables' must be a list, but is currently of type {0}.".format(
            type(variables)
        ))
    else:
        for variable in variables:
            if type(variable) is not str:
                raise TypeError("At least one element of the list for 'variables' is not a string.")

    if type(benchmark_criterion) is not str:
        raise TypeError("Argument for 'benchmark_criterion' must be of type string, but is currently of type {0}.".format(
            type(benchmark_criterion)
        ))
    elif benchmark_criterion != 'aic' and benchmark_criterion != 'bic':
        raise TypeError("Argument for 'benchmark_criterion' must be either 'aic' or 'bic'.")

    if type(keep_best_n) is not int:
        raise TypeError("Argument for 'keep_best_n' is not of type integer.")
    elif keep_best_n < 1:
        raise TypeError("Argument for 'keep_best_n' is smaller than 1. Please use a value of 1 or larger.")

    if type(show_warning) is not bool:
        raise TypeError("Value for parameter 'show_warning' must be of type boolean (True or False).")

    # Print warning if there are too many variables
    if len(variables) > 7 and show_warning:
        print("Warning: Your variables list contains {0} variables. This will result in {1} regressions and may take some time.".format(
            len(variables),
            2**len(variables)
        ))

    # Set column name for y if it is not set
    if y.columns.name is None:
        y.columns.name = 'y'

    # Concatenate X and y into one data frame
    data_frame = pd.concat([y, X], axis=1)

    # Get model specifications
    regressions = _get_model_specifications(y.columns.name, variables)

    # Perform for each model specification a regression
    for regression in regressions:

        model = sm.formula.logit(formula=regressions[regression]['formula'], data=data_frame)
        regressions[regression]['fitted'] = model.fit()

    # Get best indices n regressions
    best_regressions = _get_best_n_regressions(regressions, keep_best_n, benchmark_criterion)

    # Write each of the best performing regressions into an output dictionary
    output_dict = {}
    for index, regression in enumerate(best_regressions):
        output_dict[index] = regressions[regression[0]]

    return output_dict


def _get_model_specifications(y_name, variables):
    """
    Creates the regressions-dictionary and fills it with all possible model specifications for the given variables.

    :param y_name:              String. The name of the dependend variable
    :param variables:           List of strings. Contains the variable names

    :return:                    Dictionary of dictionaries. Containing all possible model specifications
    """

    regressions = {}
    regressions_index = 0

    # Create formulas for all different variable combinations as strings and write them into the regressions-dictionary
    for r in range(1, (np.size(variables) + 1)):

        for combination in combinations(variables, r):

            formula = y_name + ' ~ '

            for variable in range(0, r):
                formula += combination[variable]
                if variable < (r - 1):
                    formula += ' + '

            # Write formula to the regressions dictionary
            regressions[regressions_index] = {'formula': formula}
            regressions_index += 1

    return regressions


def _get_best_n_regressions(regressions, n, criterion):
    """
    Takes a regression dictionary with fitted values and finds the n best performing regressions
    in regard to the given criterion.

    :param regressions:     A dictionary of dictionaries. Must contain the key 'fitted' for each regression.
    :param n:               int. Number of best performing regressions to return
    :param criterion:       String. Either 'aic' or 'bic'.

    :return:                A list of lists. Each sublist contains the regression key and the corresponding aic/bic

    """

    if type(regressions) is not dict:
        raise TypeError("Object for parameter 'regressions' has the type '{0}', but must have the type 'dict'.".format(
            type(regressions)
        ))
    for regression in regressions:
        if type(regressions[regression]) is not dict:
            raise TypeError("At least one value of the argument for 'regressions' is not of type dict, but of type {0}. Please use a dictionary of dictionaries.".format(
                type(regressions[regression])
            ))
        if 'fitted' not in regressions[regression].keys():
            raise TypeError("At least one dict for the input of the parameter 'regressions' does not contain the key 'fitted'.")

    if type(n) is not int:
        raise TypeError("Value for parameter 'n' is not of type integer. Please use an integer value.")
    elif n <= 0:
        raise TypeError("Value for parameter 'n' ({0}) is smaller than 1. Please use a value of 1 or larger.".format(
            n
        ))

    if type(criterion) is not str:
        raise TypeError("Argument for parameter 'criterion' is not of type string.")
    elif criterion != 'aic' and criterion != 'bic':
        raise TypeError("Argument for parameter 'criterion' ({0}) does not define a valid criterion. Please use 'aic' or 'bic' as criterion.".format(
            criterion
        ))

    # Get best n regressions
    best_n = []

    for regression in regressions:
        # If list is not yet full -> fill it
        if len(best_n) < n:
            best_n.append([regression, getattr(regressions[regression]['fitted'], criterion)])

        # Else if new criterion is smaller than the largest of the criteria in the list, replace it for the largest
        else:
            best_n.sort(key=lambda x: x[1], reverse=True)

            if getattr(regressions[regression]['fitted'], criterion) < best_n[0][1]:
                best_n[0] = [regression, getattr(regressions[regression]['fitted'], criterion)]
            else:
                pass

    best_n.sort(key=lambda x: x[1], reverse=False)
    return best_n
