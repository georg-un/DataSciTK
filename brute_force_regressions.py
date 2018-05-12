import numpy as np
import statsmodels.api as sm
import pandas as pd
from itertools import combinations
import warnings

from _helper import _check_pandas_dataframe_1d
from _helper import _check_pandas_dataframe_nd
from _helper import _check_string
from _helper import _check_list_of_strings
from _helper import _check_integer
from _helper import _check_boolean
from _helper import _check_float

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
    :param verbose:                 True or False. Defines if a warning should be printed when a regression is skipped.

    :return:                        Dictionary of dictionaries. Contains the n best performing regressions.

    """

    # Check if inputs are valid
    _check_pandas_dataframe_1d(y, 'y')

    _check_pandas_dataframe_nd(X)

    if len(X) != len(y):
        raise TypeError("Inputs for 'X' and 'y' have a different number of observations. ({0}, {1})".format(
            len(X),
            len(y)
        ))

    _check_list_of_strings(variables, 'variables')

    _check_string(benchmark_criterion, 'benchmark_criterion')
    if benchmark_criterion != 'aic' and benchmark_criterion != 'bic':
        raise TypeError("Argument for 'benchmark_criterion' must be either 'aic' or 'bic'.")

    _check_integer(keep_best_n, 'keep_best_n')
    if keep_best_n < 1:
        raise TypeError("Argument for 'keep_best_n' is smaller than 1. Please use a value of 1 or larger.")

    _check_integer(max_exponent, 'max_exponent')
    if max_exponent < 1:
        raise TypeError("Argument for 'max_exponent' must not be smaller than 1.")

    _check_integer(max_root, 'max_root')
    if max_root < 1:
        raise TypeError("Argument for 'max_root' must not be smaller than 1.")

    _check_boolean(include_log, 'include_log')

    _check_boolean(include_interactions, 'include_interactions')

    _check_float(alpha, 'alpha')
    if alpha <= 0 or alpha >= 1:
        raise TypeError("Value for 'alpha' must be between 0 and 1.")

    _check_boolean(optimization, 'optimization')

    _check_boolean(verbose, 'verbose')

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
                                            columns_larger_zero=columns_larger_zero)

    # Perform for each model specification a regression
    for regression in regressions:
        model = sm.formula.logit(formula=regressions[regression]['formula'], data=data_frame)
        try:
            # Ignore warnings from failed convergence
            with warnings.catch_warnings():     # TODO: find out what happens in cases where MLE does not converge
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
                              include_interactions=False, alpha=0.1, optimization=True, columns_larger_zero=None):
    """
    Creates the regressions-dictionary and fills it with all possible model specifications for the given variables.

    :param data:                    n-dimensional pandas DataFrame containing the dependent variable as first column TODO: maybe change this
    :param regression_type:         String. Defines the regression type.
    :param y_name:                  String. Defines the column name of the dependent variable.
    :param variables:               List of strings. Defines the column names of the variables which should be included.
    :param max_exponent:            Integer larger than 0. Defines the maximum exponent which should be included for each variable.
    :param max_root:                Integer larger than 0. Defines the maximum root which should be included for each variable.
    :param include_log:             Boolean. Defines if the natural logarithm should be included for each variable.
    :param include_interactions:    Boolean. Defines if multiplicative interactions between the original variables should be included.
    :param alpha:                   Float. Defines the significance level for the optimizations. Only variables which score a p-value
                                    smaller than alpha in a simple linear model are included in the model specifications.
    :param optimization:            Boolean. Defines if an optimization should run (highly recommended).
    :param columns_larger_zero:     List of strings. Log and root are only included for columns which are mentioned in this list.

    :return:                        Dictionary of dictionaries. Contains the formula for each model specification.

    """

    # Define available regressions
    REGRESSION_TYPES = [
        'logit', 'poisson', 'glm', 'gls', 'glsar', 'mnlogit', 'negativebinomial', 'ols', 'probit', 'rlm', 'wls'
    ]

    # Check if inputs are valid
    _check_pandas_dataframe_nd(data)

    _check_string(regression_type, 'regression_type')
    if regression_type not in REGRESSION_TYPES:
        raise TypeError("Argument for parameter 'regression_type' ({0}) is not a valid regression type.".format(
            regression_type
        ))

    _check_string(y_name, 'y_name')

    _check_list_of_strings(variables, 'variables')

    _check_integer(max_exponent, 'max_exponent')
    if max_exponent < 1:
        raise TypeError("Argument for parameter 'max_exponend' must be 1 or larger.")

    _check_integer(max_root, 'max_root')
    if max_root < 1:
        raise TypeError("Argument for parameter 'max_root' must be 1 or larger.")

    _check_boolean(include_log, 'include_log')

    _check_boolean(include_interactions, 'include_interactions')

    _check_float(alpha, 'alpha')
    if alpha <= 0 or alpha >= 1:
        raise TypeError("Argument for parameter 'alpha' must be a value between 0 and 1.")

    _check_boolean(optimization, 'optimization')

    if columns_larger_zero is not None:
        _check_list_of_strings(columns_larger_zero, 'columns_larger_zero')
    if (include_log or max_root > 1) and (columns_larger_zero is None or np.size(columns_larger_zero) == 0):
        print("Argument for parameter 'columns_larger_zero' contains no columns. Logarithm and roots will not be included.")

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
            if variable in columns_larger_zero:
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
    """
    Runs a simple regression (target ~ variable) and returns true if the p-value is smaller than the given alpha.

    :param data:                    n-dimensional pandas DataFrame. Must contain the dependent and the independent variable.
    :param target_name:             String. Defines the column name of the dependent variable.
    :param variable_name:           String. Defines the column name of the independent variable.
    :param regression_type:         String. Defines the regression type (see REGRESSION_TYPES).
    :param alpha:                   Float. Defines the significance level.

    :return:                        True if the p-value of the variable is smaller than alpha. False otherwise.

    """

    # Define available regressions
    REGRESSION_TYPES = [
        'logit', 'poisson', 'glm', 'gls', 'glsar', 'mnlogit', 'negativebinomial', 'ols', 'probit', 'rlm', 'wls'
    ]

    # Check if inputs are valid
    _check_pandas_dataframe_nd(data)

    _check_string(target_name, 'target_name')

    _check_string(variable_name, 'variable_name')

    _check_string(regression_type, 'regression_type')
    if regression_type not in REGRESSION_TYPES:
        raise TypeError("'{0}' is not available as regression-type.".format(regression_type))

    _check_float(alpha, 'alpha')
    if alpha <= 0.0 or alpha >= 1.0:
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

    _check_integer(n, 'n')
    if n <= 0:
        raise TypeError("Value for parameter 'n' ({0}) is smaller than 1. Please use a value of 1 or larger.".format(
            n
        ))

    _check_string(criterion, 'criterion')
    if criterion != 'aic' and criterion != 'bic':
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
