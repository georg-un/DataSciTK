import numpy as np
import statsmodels.api as sm
import pandas as pd
import warnings
from itertools import combinations

from utils import get_columns_larger_zero
from category_ops import get_boolean_columns

from _input_checks import check_numpy_array_pandas_dataframe_series_1d
from _input_checks import check_pandas_dataframe_nd
from _input_checks import check_list_numpy_array
from _input_checks import check_string
from _input_checks import check_integer
from _input_checks import check_boolean


class BruteForceRegression:
    """
    Try to brute force the best fitting model specification for a regression.


    """

    def __init__(self, y, X, variables, regression_method, benchmark_criterion='aic', max_exponent=3, max_root=3,
                 include_log=True, include_interactions=True, verbose=False):
        """
        :param y:                           1-dimensional numpy array, pandas Series or pandas DataFrame.
                                            Contains the dependent variable.

        :param X:                           n-dimensional pandas DataFrame.
                                            Contains the independent variables

        :param variables:                   List of strings.
                                            Defines the variables (column names) that should be included in the checks.

        :param regression_method:           String.
                                            Defines the regression method. Possible methods are:
                                            'logit', 'poisson', 'glm', 'gls', 'glsar', 'mnlogit', 'negativebinomial',
                                            'ols', 'probit', 'rlm', 'wls'.

        :param benchmark_criterion:         String.
                                            Defines the benchmark criterion. Possible criterions are:
                                            'aic', 'bic'.

        :param max_exponent:                Integer.
                                            Defines the maximum exponent for each non-boolean column which should be
                                            included in the checks.

        :param max_root:                    Integer.
                                            Defines the maximum root for each column with values larger than zero which
                                            should be included in the checks.

        :param include_log:                 Boolean.
                                            Defines if the logarithm should be included in the checks for each column
                                            with values larger than zero.

        :param include_interactions:        Boolean.
                                            Defines if multiplicative interactions between all variables should be
                                            included in the checks.

        :param verbose:                     Boolean.
                                            Defines the verbosity level of the output.

        """

        # Define constants
        REGRESSION_TYPES = ['logit', 'poisson', 'glm', 'gls', 'glsar', 'mnlogit', 'negativebinomial', 'ols', 'probit', 'rlm', 'wls']
        BENCHMARK_CRITERIA = ['aic', 'bic']

        # Check if input types are valid
        check_numpy_array_pandas_dataframe_series_1d(y, 'y')
        check_pandas_dataframe_nd(X, 'X')
        check_list_numpy_array(variables, 'variables')
        check_string(regression_method, 'regression_method')
        check_string(benchmark_criterion, 'benchmark_criterion')
        check_integer(max_exponent, 'max_exponent')
        check_integer(max_root, 'max_root')
        check_boolean(include_log, 'include_log')
        check_boolean(include_interactions, 'include_interactions')
        check_boolean(verbose, 'verbose')

        # Check additional restrictions
        if len(variables) < 3:
            raise TypeError("Number of variables has to be at least 3.")

        if regression_method not in REGRESSION_TYPES:
            raise TypeError("Regression method '{0}' is not available. Please use one of the following: {1}.".format(
                regression_method,
                REGRESSION_TYPES
            ))

        if benchmark_criterion not in BENCHMARK_CRITERIA:
            raise TypeError("Benchmark criterion '{0}' is not available. Please use one of the following: {1}.".format(
                benchmark_criterion,
                BENCHMARK_CRITERIA
            ))

        if max_exponent < 1:
            raise TypeError("Argument for parameter 'max_exponent' must be at least 1.")

        if max_root < 1:
            raise TypeError("Argument for parameter 'max_root' must be at least 1.")

        # Assign input variables to object
        self.y = y
        self.X = X
        self.variables = variables
        self.regression_method = regression_method
        self.benchmark_criterion = benchmark_criterion
        self.max_exponent = max_exponent
        self.max_root = max_root
        self.include_log = include_log
        self.include_interactions = include_interactions
        self.verbose = verbose

        self._concat_y_X()
        self._generate_all_variables()

    def get_best_fitting_models(self, n_models=1):
        """
        Run the brute force procedure to find the n best fitting models in terms of the benchmark criterion.

        :param n_models:    Integer. Number of models to return.
        :return:            If n_models == 1, the best fitted model will be returned, else a list of the n best models.

        """

        # Check if input is valid
        check_integer(n_models, 'n_models')
        if n_models < 1:
            raise TypeError("Argument for parameter 'n_models' must be at least 1.")
        if n_models > len(self.variables):
            raise TypeError("Argument for parameter 'n_models' can be maximum {0}".format(len(self.variables)))

        # Set up results list
        models = []

        # Find best fitting model specification for each variable
        for variable in self.variables:
            best_variables, benchmark = self._find_best_fitting_variables(variable)
            models.append((best_variables, benchmark))

        # Sort by benchmark criterion ascending
        models.sort(key=lambda x: x[1])

        # If n_models is 1, run final regression and return a single model
        if n_models == 1:
            return self._run_regression(models[0][0])
        # If n_models is >1, run n regressions and return a list of models
        elif n_models > 1:
            best_n_models = []
            for n in range(0, n_models-1):
                best_n_models.append(self._run_regression(models[n][0]))
            return best_n_models
        else:
            raise IOError("n_models was neither 1 nor a value larger than 1.")

    def _build_formula(self, variables):
        """
        Create a R-like formula for the regression in statsmodels.

        :param variables:       List of strings. Defines the variable names (column names).
        :return:                String. The final formula (e.g. 'y ~ a + b + c').

        """

        # Get the column name for the dependent variable from data_frame
        formula = self.data_frame.columns[0] + " ~ "

        # Add each variable to the string
        for variable in variables:
            formula = formula + variable + " + "

        return formula[:-3]

    def _run_regression(self, variables):
        """
        Run a regression in statsmodels.

        :param variables:       List of strings. Defines the variable names (column names).
        :return:                Model object of the fitted regression.

        """

        # Build formula
        if self.regression_method == 'ols':
            model = sm.formula.ols(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'logit':
            model = sm.formula.logit(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'poisson':
            model = sm.formula.poisson(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'glm':
            model = sm.formula.glm(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'gls':
            model = sm.formula.gls(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'glsar':
            model = sm.formula.glsar(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'mnlogit':
            model = sm.formula.mnlogit(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'negativebinomial':
            model = sm.formula.negativebinomial(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'probit':
            model = sm.formula.probit(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'rlm':
            model = sm.formula.rlm(formula=self._build_formula(variables), data=self.data_frame)
        elif self.regression_method == 'wls':
            model = sm.formula.wls(formula=self._build_formula(variables), data=self.data_frame)
        else:
            model = None

        # Fit model
        try:
            # Ignore warnings from failed convergence
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Try to fit model
                fitted_model = model.fit(disp=0)  # disp=0 to suppress convergence messages being printed
        except Exception:
            if self.verbose:
                print("Model skipped:", self._build_formula(variables))
            fitted_model = None

        return fitted_model

    def _find_best_fitting_variables(self, variable):
        """
        Find the best fitting model by specifying a starting variable. Process goes as follows:

        1. Start with the starting model 'y ~ starting_variable' and measure its benchmark criterion (AIC, BIC).
        2. Add each of the available variables individually to the model and run a regression respectively.
        3. Measure the benchmark criterion for each single regression.
        4. Compare the benchmark of the best performing regression to the benchmark of the starting model.
        5. If the performance did increase, add the additional variable from the best performing regression to the model.
        6. Repeat steps 1 to 5 with the new model instead of the starting model until the benchmark performance does not
           increase anymore.

        :param variable:        String. Column-name of the variable which should be used as starting variable.

        :return:                List of strings. The variable names of the best performing model.
        :return:                Float. The value from the benchmark criterion of the best performing model.

        """

        # Set up variables
        current_variables = [variable]
        remaining_variables = self.variables[self.variables != variable]
        best_model_not_yet_found = True

        # Run a regression with the single variable to get the benchmark
        fitted = self._run_regression([variable])
        current_benchmark = getattr(fitted, self.benchmark_criterion)

        # Print info if verbose is enabled
        if self.verbose:
            print("Start with variable", variable, "with benchmark:", current_benchmark)

        # Run this loop until the benchmark does not get better anymore
        while best_model_not_yet_found:

            regression_result = []

            # Add each variable to the model individually and run a regression. Measure the criterion score
            for test_variable in remaining_variables:

                # Get model and run regression
                test_variables = current_variables
                test_variables.append(test_variable)
                fitted = self._run_regression(test_variables)

                # Get benchmark criterion
                try:
                    criterion = getattr(fitted, self.benchmark_criterion)
                except AttributeError:
                    criterion = np.inf

                # Write tested variable and criterion to results list
                regression_result.append((test_variables.pop(), criterion))

            # Sort regression results ascending by benchmark criterion
            regression_result.sort(key=lambda x: x[1])

            # If best performing additional variable increases model performance
            if regression_result[0][1] < current_benchmark:
                # -> Add it to the model
                current_variables.append(regression_result[0][0])
                # -> Remove it from remaining variables
                remaining_variables = remaining_variables[remaining_variables != regression_result[0][0]]
                # -> Update current benchmark
                current_benchmark = regression_result[0][1]
                # If there are no remaining variables left -> exit loop
                if len(remaining_variables) == 0:
                    best_model_not_yet_found = False
                # Print result if verbose is enabled
                if self.verbose:
                    print("Benchmark increased:", current_benchmark, "New model:", current_variables)

            # If best performing additional variable does not increase model performance
            elif regression_result[0][1] >= current_benchmark:
                # -> Exit loop
                best_model_not_yet_found = False
                # Print final result if verbose is enabled
                if self.verbose:
                    print("Final benchmark:", current_benchmark, "\n")
            else:
                raise IOError("Regression result was neither equal, nor smaller, nor larger than current benchmark.")

        # Return best fitting variables and their benchmark
        return current_variables, current_benchmark

    def _concat_y_X(self):
        """
        Concatenates y and X into a single dataframe and writes it to self.data_frame

        """

        # Concatenate X and y into one data frame
        self.data_frame = pd.concat([self.y, self.X], axis=1)

        # Set column name for dependent variable if it is not already set
        if self.data_frame.columns[0] == 0:
            self.data_frame.rename(columns={0: 'y'}, inplace=True)

    def _generate_all_variables(self):
        """
        Generates all additional variables (exponents, roots, logs, interactions). Exponents are only generated for
        non-boolean columns. Roots and logs are only generated for columns with every value larger than zero.

        Overwrites self.variables with the new variables.

        """

        # Get columns larger zero and boolean columns
        columns_larger_zero = get_columns_larger_zero(self.X)
        columns_boolean = get_boolean_columns(self.X)

        # Get exponential variables
        all_variables = []
        for exponent in range(2, self.max_exponent + 1):
            for variable in self.variables:
                if variable not in columns_boolean:
                    new_variable = 'np.power(' + str(variable) + ', ' + str(exponent) + ')'
                    all_variables.append(new_variable)

        # Get root variables
        for root in range(2, self.max_root + 1):
            for variable in self.variables:
                if variable in columns_larger_zero:
                    new_variable = 'np.power(' + str(variable) + ', -' + str(float(root)) + ')'
                    all_variables.append(new_variable)

        # Get log variables
        if self.include_log:
            for variable in self.variables:
                if variable in columns_larger_zero:
                    new_variable = 'np.log(' + str(variable) + ')'
                    all_variables.append(new_variable)

        # Get interactions
        if self.include_interactions:
            for combination in combinations(self.variables, 2):
                # Check if interaction is always zero (could happen if categories are coded in separate binary columns)
                if sum(self.X[combination[0]] * self.X[combination[1]]) > 0:
                    interaction = str(combination[0]) + ':' + str(combination[1])
                    all_variables.append(interaction)

        self.variables = np.append(self.variables, all_variables)
