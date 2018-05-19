import numpy as np
import statsmodels.api as sm
import pandas as pd
import warnings

from utils import get_columns_larger_zero
from category_ops import get_boolean_columns


class BruteForceRegression:
    def __init__(self, y, X, variables, regression_method, benchmark_criterion='aic', max_exponent=3, max_root=3, include_log=True,
                 include_interactions=True, alpha=0.05, verbose=False):
        # TODO: input checks
        self.y = y
        self.X = X
        self.variables = variables
        self.regression_method = regression_method
        self.benchmark_criterion = benchmark_criterion
        self.max_exponent = max_exponent
        self.max_root = max_root
        self.include_log = include_log
        self.include_interactions = include_interactions
        self.alpha = alpha
        self.verbose = verbose

        self._concat_y_X()
        self.generate_all_variables()

    def get_best_fitting_models(self, n_models=1):
        # TODO: input checks

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
            return self.run_regression(models[0][0])
        # If n_models is >1, run n regressions and return a list of models
        elif n_models > 1:
            best_n_models = []
            for n in range(0, n_models-1):
                best_n_models.append(self.run_regression(models[n][0]))
            return best_n_models
        else:
            raise IOError("n_models was neither 1 nor a value larger than 1.")

    def _concat_y_X(self):
        # Concatenate X and y into one data frame
        self.data_frame = pd.concat([self.y, self.X], axis=1)

        # Set column name for dependent variable if it is not already set
        if self.data_frame.columns[0] == 0:
            self.data_frame.rename(columns={0: 'y'}, inplace=True)

    def generate_all_variables(self):

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

        self.variables = np.append(self.variables, all_variables)

    def _find_best_fitting_variables(self, variable):

        # Set up variables
        current_variables = [variable]
        remaining_variables = self.variables[self.variables != variable]  # TODO: this will later throw exception if there are only 2 variables -> input check
        best_model_not_yet_found = True

        # Run a regression with the single variable to get the benchmark
        fitted = self.run_regression([variable])
        current_benchmark = getattr(fitted, self.benchmark_criterion)

        # Print
        print("Start with variable", variable, "with benchmark:", current_benchmark)

        # Run this loop until the benchmark does not get better anymore
        while best_model_not_yet_found:

            regression_result = []

            # Add each variable to the model individually and run a regression. Measure the criterion score
            for test_variable in remaining_variables:

                # Get model and run regression
                test_variables = current_variables
                test_variables.append(test_variable)
                fitted = self.run_regression(test_variables)

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

    def run_regression(self, variables):

        # Build model
        if self.regression_method == 'logit':  # TODO: add additional methods
            model = sm.formula.logit(formula=self._build_formula(variables), data=self.data_frame)
        else:
            model = None

        # Fit model
        try:
            # Ignore warnings from failed convergence
            with warnings.catch_warnings():  # TODO: find out what happens in cases where MLE does not converge
                warnings.filterwarnings('ignore')
                # Try to fit model
                fitted_model = model.fit(disp=0)  # disp=0 to suppress convergence messages being printed
        except Exception:
            if self.verbose:
                print("Model skipped:", self._build_formula(variables))
            fitted_model = None

        return fitted_model

    def _build_formula(self, variables):
        formula = self.data_frame.columns[0] + " ~ "

        for variable in variables:
            formula = formula + variable + " + "

        return formula[:-3]













