# Copyright (C) 2018 Georg Unterholzner
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ==============================================================================
""" Distributions module """

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

from _input_checks import check_numpy_array_pandas_dataframe_series_1d
from _input_checks import check_float
from _input_checks import check_string
from _input_checks import check_boolean
from _input_checks import check_integer


def is_not_normally_distributed(data, alpha=0.05, alternative='two-sided', mode='approx', verbose=False):
    """
    Performs a Kolmogorov-Smirnov-Test for normal distribution. The tested hypothesis is that the data is not
    normally distributed. If the p-value is smaller than alpha, it returns True.

    :param data:                1-dimensional numpy array or pandas DataFrame of shape (m, 1).
    :param alpha:               Float. Defines the significance level.
    :param alternative:         String. Either 'two-sided', 'less' or 'greater'. Defines the alternative hypothesis.
    :param mode:                String. Either 'approx' or 'asymp'. See scipy.stats.kstest for more info.
    :param verbose:             True or False. True for verbose output.

    :return:                    True if data is not normally distributed. False if alternative hypothesis cannot be rejected.

    """

    # Check if inputs are valid
    check_numpy_array_pandas_dataframe_series_1d(data, 'data')

    check_float(alpha, 'alpha')
    if alpha <= 0.0 or alpha >= 1:
        raise TypeError("Value for 'alpha' is {0}, but must be a value between 0 and 1.".format(alpha))

    check_string(alternative, 'alternative')
    if alternative not in ['two-sided', 'less', 'greater']:
        raise TypeError("Value for parameter 'alternative' must be either 'two-sided', 'less' or 'greater'.")

    check_string(mode, 'mode')
    if mode not in ['approx', 'asymp']:
        raise TypeError("Value for parameter 'mode' must be either 'approx' or 'asymp'.")

    check_boolean(verbose, 'verbose')

    # Test alternative hypothesis
    alternative_hypothesis = st.kstest(data, 'norm', alternative=alternative, mode=mode)

    # Compare the p-value with the given alpha and return the respective result
    if alternative_hypothesis.pvalue < alpha:
        if verbose:
            print("Not normally distributed with a p-value of {0}.".format(alternative_hypothesis.pvalue))
        return True
    elif alternative_hypothesis.pvalue >= alpha:
        if verbose:
            print("Normally distributed with a p-value of {0}.".format(alternative_hypothesis.pvalue))
        return False
    else:
        raise IOError("Did not get a p-value for the Kolmogorov-Smirnov-Test.")


class DistributionFitter:
    """
    Fit theoretical distributions to a 1-dimensional data.

    Use it for a single distribution to plot it's density function or to predict the probability of a specific value:
    fit(distribution)

    Or use it to find the best fitting distribution in a set of 89 theoretical distributions:
    best_n_fitting(n)


    This is a modification of the code from the great answer from tmthydvnprt on stackoverflow.com:
    https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python#answer-37616966

    """

    def __init__(self, data, n_bins=200, verbose=False):
        """
        :param data:        1-dimensional numpy array, pandas Series or pandas Dataframe.
        :param n_bins:      Integer (at least 10). Defines the number of bins for the histogram and the precision.
        :param verbose:     True or False. Defines the verbosity of the output.

        """

        # Check if input types are valid
        check_numpy_array_pandas_dataframe_series_1d(data, 'data')
        check_integer(n_bins, 'n_bins')
        check_boolean(verbose, 'verbose')

        # Convert data to pandas.Series if it is a numpy ndarray
        if type(data) is np.ndarray:
            data = pd.Series(data)

        # Make sure n_bins is larger than 10
        if n_bins < 10:
            raise TypeError("Argument for parameter 'n_bins' must be at least 10.")

        # Resize n_bins if it is too large
        if n_bins > len(data):
            n_bins = len(data)

        # Assign input variables to object
        self.data = data
        self.n_bins = n_bins
        self.verbose = verbose

    def fit(self, distribution):
        """
        Fit a scipy.stats distribution to the data.

        :param distribution:        A scipy.stats distribution object (e.g. scipy.stats.norm). Defines the theoretical
                                    distribution which will be fitted to the data.

        :return:                    An object of class _Fitted_Distribution containing the distribution, the fitted
                                    distribution parameters, the sum of squared errors of the distribution and the data.

        """

        # Check if distribution is valid
        if distribution not in _get_distributions():
            raise TypeError("Argument '{0}' for parameter 'distribution' is not a valid distribution. Please use a scipy.stats distribution object.".format(
                distribution,
            ))

        # Get histogram and bin_edges of data
        histogram, bin_edges = np.histogram(self.data, bins=self.n_bins, density=True)
        bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # Fit distribution to data
                parameters = distribution.fit(self.data)

                # Get parameters from the fitted distribution
                arg = parameters[:-2]
                mean = parameters[-2]
                standard_deviation = parameters[-1]

                # Calculate the fitted PDF
                pdf = distribution.pdf(bin_edges, loc=mean, scale=standard_deviation, *arg)

                # Calculate the SSE
                sse = np.sum(np.power(histogram - pdf, 2.0))

                # Create fitted distribution object
                fitted_distribution = _FittedDistribution(distribution=distribution,
                                                          standard_deviation=standard_deviation,
                                                          mean=mean,
                                                          arg=arg,
                                                          parameters=parameters,
                                                          sse=sse,
                                                          data=self.data)

                return fitted_distribution

        # Catch all exceptions and print them if verbose is True
        except Exception as e:
            if self.verbose:
                print("Error at distribution '{0}':".format(distribution.name), e)

            return None

    def best_n_fitting(self, n):
        """
        Fit 89 scipy.stats distributions to the data. Return the n best distributions in terms of their SSE.

        :param n:       Integer (at least 1). Defines the number of distributions to return.

        :return:        An object of class _Fitted_Distributions which contains n objects of class _Fitted_Distribution.
                        All contain the respective distribution, the fitted distribution parameters, the sum of squared
                        errors (SSE) of the distribution and the data.

        """

        # Check if input is valid
        check_integer(n, 'n')

        if n < 1:
            raise TypeError("Argument for parameter 'n' must be at least 1.")

        # Initialize results object
        fitted = _FittedDistributions(data=self.data)

        # Estimate fit for each distribution
        for distribution in _get_distributions():

            # Get fitting results for distribution
            distribution_fit = self.fit(distribution)

            # Write results to results-list
            if distribution_fit is not None:
                fitted.distributions.append(distribution_fit)

        # Sort ascending by SSE
        fitted.distributions.sort(key=lambda item: item.sse)

        # Make sure n is not larger than the number of fitted distributions
        if n > len(fitted.distributions):
            n = len(fitted.distributions)

        # Keep only the best n results and return them
        fitted.distributions = fitted.distributions[0:n]
        return fitted


class _FittedDistribution:
    """
    Class for a single fitted distribution.

    """

    def __init__(self, distribution, standard_deviation, mean, arg, parameters, sse, data):
        """
        :param distribution:            A scipy.stats distribution object
        :param standard_deviation:      Float. Standard deviation of the fitted distribution.
        :param mean:                    Float. Mean of the fitted distribution.
        :param arg:                     Tuple or list. Additional parameters of the fitted distribution.
        :param parameters:              List of standard_deviation, mean and arg
        :param sse:                     Float. Sum of squared errors of the fitted distribution.
        :param data:                    1-dimensional pandas Series or DataFrame
        """

        # Assign input to object variables
        self.distribution = distribution
        self.standard_deviation = standard_deviation
        self.mean = mean
        self.arg = arg
        self.parameters = parameters
        self.sse = sse
        self.data = data

    def plot(self, x_label, title='default', y_label='Frequency', legend=True):
        """
        Plot a histogram of the data and the probability density function of the fitted distribution.

        :param x_label:         String. Title of the x-axis.
        :param title:           String. Title of the plot. If 'default', the default title will be used.
        :param y_label:         String. Title of the y-axis.
        :param legend:          Boolean. Defines if a legend will be shown.

        """

        # Check if input types are valid
        check_string(x_label, 'x_label')
        check_string(y_label, 'y_label')
        check_string(title, 'title')
        check_boolean(legend, 'legend')

        # Get string of additional parameters
        if len(self.arg) > 0:
            parameters = str([round(x, 2) for x in self.arg])[1:-1]
        else:
            parameters = 'None'

        # Set default title
        if title == 'default':
            title = "Histogram of {0} with the theoretical distribution {1}.\nSD: {2}, Mean: {3}, Additional parameters: {4}.".format(
                x_label,
                self.distribution.name.capitalize(),
                round(self.standard_deviation, 2),
                round(self.mean, 2),
                parameters
            )

        # Create main plot
        plt.figure(figsize=(12, 8))
        ax = self.data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=legend)
        y_lim = (ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
        x_lim = ax.get_xlim()

        # Get probability density function and plot it
        pdf = _get_pdf(distribution=self.distribution, parameters=self.parameters)
        pdf.plot(lw=2, label=self.distribution.name.capitalize(), legend=legend, ax=ax)

        # Set focus on histogram
        plt.ylim(y_lim)
        plt.xlim(x_lim)

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel=x_label)
        ax.set_ylabel(ylabel=y_label)

    def probability_x_smaller_equal(self, value):
        """
        Get the probability of a random sample of the fitted distribution being smaller or equal the given value.
        Calls the cumulative distribution function (CDF).

        :param value:       Array_like. Defines the values for which the probability will be returned.

        :return:            1-dimensional numpy array. Contains the probability values.

        """
        return self.distribution.cdf(value, *[self.arg, self.mean, self.standard_deviation])

    def probability_x_larger_equal(self, value):
        """
        Get the probability of a random sample of the fitted distribution being larger or equal the given value.
        Calls the survival function (SF), a.k.a. complemaentary cumulative distribution function.

        :param value:       Array_like. Defines the values for which the probability will be returned.

        :return:            1-dimensional numpy array. Contains the probability values.

        """
        return self.distribution.sf(value, *[self.arg, self.mean, self.standard_deviation])

    def value_for_probability_x(self, probability):
        """
        Get the value which is needed to provide a probability of x.
        Calls the percent-point function (PPF), a.k.a. quantile function.

        :param probability:     Array_like. Defines the probability for which the value will be returned.

        :return:                1-dimensional numpy array. Contains the respective values.

        """
        return self.distribution.ppf(probability, *[self.arg, self.mean, self.standard_deviation])


class _FittedDistributions:
    """
    Class for multiple fitted distributions.

    """

    def __init__(self, data):
        """
        :param data:        1-dimensional pandas Series or DataFrame.

        """

        self.distributions = []
        self.data = data

    def plot(self, x_label, title='default', y_label='Frequency', legend=True):
        """
        Plot a histogram of the data and the probability density functions of the n best fitting distributions.

        :param x_label:         String. Title of the x-axis.
        :param title:           String. Title of the plot. If 'default', the default title will be used.
        :param y_label:         String. Title of the y-axis.
        :param legend:          Boolean. Defines if a legend will be shown.

        """

        # Check if input types are valid
        check_string(x_label, 'x_label')
        check_string(y_label, 'y_label')
        check_string(title, 'title')
        check_boolean(legend, 'legend')

        # Set default title
        if title == 'default':
            title = "Comparison between the best {0} fitting distributions.".format(len(self.distributions))

        # Create main plot
        plt.figure(figsize=(12, 8))
        ax = self.data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=legend)
        y_lim = (ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
        x_lim = ax.get_xlim()

        # Plot the best n distributions
        for index in range(0, len(self.distributions)):
            # Get distribution and parameter
            distribution = self.distributions[index].distribution
            distribution_name = distribution.name
            parameters = self.distributions[index].parameters

            # Get PDF and plot it
            pdf = _get_pdf(distribution=distribution, parameters=parameters)
            pdf.plot(lw=2, label=distribution_name.capitalize(), legend=legend, ax=ax)

        # Set focus on histogram
        plt.ylim(y_lim)
        plt.xlim(x_lim)

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel=x_label)
        ax.set_ylabel(ylabel=y_label)


def _get_distributions():
    """
    :return:    List of scipy.stats distributions

    This is a modification of the code from the great answer from tmthydvnprt on stackoverflow.com:
    https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python#answer-37616966

    """

    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f,
        st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto,
        st.gennorm, st.genexpon, st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat,
        st.gompertz, st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm,
        st.hypsecant, st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign,
        st.laplace, st.levy, st.levy_l, st.levy_stable, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax,
        st.maxwell, st.mielke, st.nakagami, st.ncx2, st.ncf, st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw,
        st.powerlognorm, st.powernorm, st.rdist, st.reciprocal, st.rayleigh, st.rice, st.recipinvgauss, st.semicircular,
        st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line,
        st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]

    return DISTRIBUTIONS


def _get_pdf(distribution, parameters, size=1000):
    """
    Generate the probability density function of a distribution.

    :param dist:        A scipy.stats distribution
    :param params:      Tuple or list of floats. Parameters from fitted distribution.
    :param size:        Integer. Number of data points to generate.

    :return:            pandas Series of shape (1000,). Contains the PDF y values for each X.


    This is a modification of the code from the great answer from tmthydvnprt on stackoverflow.com:
    https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python#answer-37616966

    """

    if distribution not in _get_distributions():
        raise TypeError("Distribution must be a scipy.stats distribution and defined in _get_distributions().")

    if type(parameters) is not list and type(parameters) is not tuple:
        raise TypeError("Input for 'parameters' must be either a list or a tuple.")

    if type(size) is not int:
        raise TypeError("Value for 'size' must be of type integer.")

    # Extract parameters
    arg = parameters[:-2]
    mean = parameters[-2]
    standard_deviation = parameters[-1]

    # Get start and end points of distribution
    if arg:
        start = distribution.ppf(0.01, *arg, loc=mean, scale=standard_deviation)
        end = distribution.ppf(0.99, *arg, loc=mean, scale=standard_deviation)
    else:
        start = distribution.ppf(0.01, loc=mean, scale=standard_deviation)
        end = distribution.ppf(0.99, loc=mean, scale=standard_deviation)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = distribution.pdf(x, loc=mean, scale=standard_deviation, *arg)
    pdf = pd.Series(y, x)

    return pdf
