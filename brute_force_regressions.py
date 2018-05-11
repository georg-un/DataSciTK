import numpy as np
import statsmodels.api as sm
import pandas as pd
from itertools import combinations
import warnings

from _helper import _check_pandas_dataframe_nd
from utils import contains_negatives_or_zero


def brute_force_logit(y, X, variables, benchmark_criterion, keep_best_n=5, max_exponent=1, max_root=1,
                      include_log=False, include_interactions=False, alpha=0.1, optimization=True, verbose=True):
    """
    Brute force a model with a logit regression. Create all possible model specifications of the given variables and
    run a logit regression for each one. Find the n best performing regressions in terms of the benchmark criterion
    and return them.

    e.g.
    $ brute_force_logit(y=y, X=X, variables=['a', 'b', 'c'], benchmark_criterion='aic', keep_best_n=10, max_exponent=3)
    will run a logit regression for each possible variable combination
    (y ~ a;  y ~ b;  y ~ c;  y ~ a + b;  y ~ a + c;  y ~ b + c;  y ~ a + b + c)
    and measure each AIC. Subsequently, it will return the 10 best performing regressions as dictionaries.

    :param y:                       Pandas DataFrame with the shape (m, 1). Contains the dependent variable coded as 0 and 1.
    :param X:                       Pandas DataFrame with the shape (m, x). Contains the independent variables coded numerically.
    :param variables:               List of strings. Contains the column names of the independent variables or modifications e.g. 'np.log(some_column)'
    :param benchmark_criterion:     String. Either 'aic' or 'bic'.
    :param keep_best_n:             Integer. Number of best regressions to return.
    :param max_exponent:            Integer. Defines the highest exponents to check for each variable.
    :param max_root:                Integer. Defines the highest roots to check for each variable.
    :param include_log:             True or False. Defines if the logarithm should be included for each variable.
    :param include_interactions:    True or False. Defines if interactions between the variables should be included.
    :param alpha:                   Float. Must be between 0 and 1. Defines the significance level for the optimization.
    :param optimization:            True or False. Defines if only significant variables should be included (highly recommended).

    :return:                        Dictionary of dictionaries. Contains the n best performing regressions.

    """
    # TODO: checks

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

    # Concatenate X and y into one data frame
    data_frame = pd.concat([y, X], axis=1)

    # Set column name for dependent variable if it is not already set
    if data_frame.columns[0] == 0:
        data_frame.rename(columns={0: 'y'}, inplace=True)

    # Get names of columns with all values above zero
    columns_larger_zero = []
    for column in data_frame:
        if not contains_negatives_or_zero(data_frame[column]):
            columns_larger_zero.append(column)

    # Get model specifications
    regressions = _get_model_specifications(data=data_frame, regression_type='logit', y_name=data_frame.columns[0],
                                            variables=variables, max_exponent=max_exponent, max_root=max_root,
                                            include_log=include_log, include_interactions=include_interactions,
                                            alpha=alpha, optimization=optimization,
                                            non_negative_columns=columns_larger_zero,
                                            columns_larger_zero=columns_larger_zero)

    # Perform for each model specification a regression
    for regression in regressions:
        model = sm.formula.logit(formula=regressions[regression]['formula'], data=data_frame)
        try:
            # Ignore warnings from failed convergence
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Try to fit model
                regressions[regression]['fitted'] = model.fit(disp=0)  # disp=0 to suppress convergence messages being printed
        except Exception:
            if verbose:
                print("Model skipped:", regressions[regression]['formula'])
            regressions[regression]['fitted'] = None

    # Get best indices n regressions
    best_regressions = _get_best_n_regressions(regressions, keep_best_n, benchmark_criterion)

    # Write each of the best performing regressions into an output dictionary
    output_dict = {}
    for index, regression in enumerate(best_regressions):
        output_dict[index] = regressions[regression[0]]

    return output_dict


def _get_model_specifications(data, regression_type, y_name, variables, max_exponent=1, max_root=1, include_log=False,
                              include_interactions=False, alpha=0.1, optimization=True, non_negative_columns=None,
                              columns_larger_zero=None):
    """
    Creates the regressions-dictionary and fills it with all possible model specifications for the given variables.

    :param y_name:              String. The name of the dependent variable
    :param variables:           List of strings. Contains the variable names

    :return:                    Dictionary of dictionaries. Containing all possible model specifications
    """

    # TODO: Checks, description

    # Get exponential variables
    additional_variables = []
    for exponent in range(2, max_exponent+1):
        for variable in variables:
            new_variable = 'np.power(' + str(variable) + ', ' + str(exponent) + ')'
            # If optimization is enabled, keep variable only if it is relevant
            if optimization:
                if is_relevant_variable(data=data, target_name=y_name, variable_name=new_variable,
                                        regression_type=regression_type, alpha=alpha):
                    additional_variables.append(new_variable)
            else:
                additional_variables.append(new_variable)

    # Get root variables
    for root in range(2, max_root+1):
        for variable in variables:
            if variable in non_negative_columns:
                new_variable = 'np.power(' + str(variable) + ', -' + str(float(root)) + ')'
                # If optimization is enabled, keep variable only if it is relevant
                if optimization:
                    if is_relevant_variable(data=data, target_name=y_name, variable_name=new_variable,
                                            regression_type=regression_type, alpha=alpha):
                        additional_variables.append(new_variable)
                else:
                    additional_variables.append(new_variable)

    # Get log variables
    if include_log:
        for variable in variables:
            if variable in columns_larger_zero:
                new_variable = 'np.log(' + str(variable) + ')'
                # If optimization is enabled, keep variable only if it is relevant
                if optimization:
                    if is_relevant_variable(data=data, target_name=y_name, variable_name=new_variable,
                                            regression_type=regression_type, alpha=alpha):
                        additional_variables.append(new_variable)
                else:
                    additional_variables.append(new_variable)

    # Get multiplicative interactions
    if include_interactions:
        variable_combinations = combinations(variables, 2)
        for combination in variable_combinations:
            new_variable = combination[0] + ' : ' + combination[1]
            # If optimization is enabled, keep variable only if it is relevant
            if optimization:
                if is_relevant_variable(data=data, target_name=y_name, variable_name=new_variable,
                                        regression_type=regression_type, alpha=alpha):
                    additional_variables.append(new_variable)
            else:
                additional_variables.append(new_variable)

    # Remove original variables which are not significant
    if optimization:
        for variable in variables:
            if not is_relevant_variable(data=data, target_name=y_name, variable_name=variable,
                                        regression_type=regression_type, alpha=alpha):
                variables.remove(variable)

    # Add additional variables to variables list
    variables.extend(additional_variables)

    # Set up regressions dictionary
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


def is_relevant_variable(data, target_name, variable_name, regression_type, alpha):

    # TODO: description

    # Define available regressions
    REGRESSION_TYPES = [
        'logit', 'poisson', 'glm', 'gls', 'glsar', 'mnlogit', 'negativebinomial', 'ols', 'probit', 'rlm', 'wls'
    ]

    # Check if inputs are valid
    _check_pandas_dataframe_nd(data)

    if type(variable_name) is not str:
        raise TypeError("Argument for 'variable_name' must be a string.")

    if type(target_name) is not str:
        raise TypeError("Argument for 'target_name' must be a string.")

    if type(regression_type) is not str:
        raise TypeError("Argument for 'regression_type' must be a string.")
    elif regression_type not in REGRESSION_TYPES:
        raise TypeError("'{0}' is not available as regression-type.".format(regression_type))

    if type(alpha) is not float and type(alpha) is not np.float64:
        raise TypeError("Argument for 'alpha' must be of type float or numpy float64.")
    elif alpha <= 0.0 or alpha >= 1.0:
        raise TypeError("Value for 'alpha' must be between 0 and 1.")

    # Make formula
    formula = target_name + ' ~ ' + variable_name

    # Get regression
    if regression_type == 'logit':
        model = sm.formula.logit(formula=formula, data=data)
    elif regression_type == 'poisson':
        model = sm.formula.poisson(formula=formula, data=data)
    elif regression_type == 'glm':
        model = sm.formula.glm(formula=formula, data=data)
    elif regression_type == 'gls':
        model = sm.formula.gls(formula=formula, data=data)
    elif regression_type == 'glsar':
        model = sm.formula.glsar(formula=formula, data=data)
    elif regression_type == 'mnlogit':
        model = sm.formula.mnlogit(formula=formula, data=data)
    elif regression_type == 'negativebinomial':
        model = sm.formula.negativebinomial(formula=formula, data=data)
    elif regression_type == 'ols':
        model = sm.formula.ols(formula=formula, data=data)
    elif regression_type == 'probit':
        model = sm.formula.probit(formula=formula, data=data)
    elif regression_type == 'rlm':
        model = sm.formula.rlm(formula=formula, data=data)
    elif regression_type == 'wls':
        model = sm.formula.wls(formula=formula, data=data)
    else:
        raise IOError("Regression method '{0}' has not been found.".format(regression_type))

    # Fit model and get p-value
    try:
        p_value = model.fit(disp=0).pvalues[1]  # disp=0 to suppress convergence messages being printed
    except Exception:
        p_value = np.nan

    # Check p-value and return respective result
    if pd.isnull(p_value):
        return False
    elif p_value >= alpha:
        return False
    elif p_value < alpha:
        return True
    else:
        raise IOError("p-value was neither equal alpha nor smaller or larger than alpha: {0}.".format(p_value))


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
            try:
                best_n.append([regression, getattr(regressions[regression]['fitted'], criterion)])
            except AttributeError:
                pass

        # Else if new criterion is smaller than the largest of the criteria in the list, replace it for the largest
        else:
            best_n.sort(key=lambda x: x[1], reverse=True)

            try:
                if getattr(regressions[regression]['fitted'], criterion) < best_n[0][1]:
                    best_n[0] = [regression, getattr(regressions[regression]['fitted'], criterion)]
                else:
                    pass
            except AttributeError:
                pass

    best_n.sort(key=lambda x: x[1], reverse=False)
    return best_n
