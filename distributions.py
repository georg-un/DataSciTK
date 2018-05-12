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

# TODO: Add function which predicts the probability of a value given a data sample (plot would be nice)


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


def plot_best_n_fitting(data, fitted_distributions, best_n, x_label, title='default', y_label='Frequency', legend=True):
    """
    Plot a histogram of the input as well as the probability distribution function of the n best matching distributions.

    :param data:                        1-dimensional numpy array or pandas DataFrame with shape (m, 1).
    :param fitted_distributions:        Dictionary or list of dictionaries. Contains the result from the fitted distributions.
    :param best_n:                      Integer. Defines the number of distributions to add to the plot.
    :param x_label:                     String. Label for the x-axis.
    :param title:                       String. Title of the plot.
    :param y_label:                     String. Label for the y-axis.
    :param legend:                      True or False. If true legend is plotted.

    :return:                            None. Creates a plot.

    """

    # Check if inputs are valid
    check_numpy_array_pandas_dataframe_series_1d(data, 'data')

    check_integer(best_n, 'best_n')
    if best_n <= 0:
        raise TypeError("Value for 'best_n' is zero or smaller. Please use a value of at least 1.")

    if type(fitted_distributions) != list:
        raise TypeError("Input for 'fitted_distributions' must be a list of dictionaries.")

    best_n = min(best_n, len(fitted_distributions))

    for index in range(0, best_n):
        if type(fitted_distributions[index]) is not dict:
            raise TypeError("At least one element inside 'fitted_distributions' is not of type dict.")
        if 'distribution' not in fitted_distributions[index].keys():
            raise TypeError("At least one dict inside 'fitted_distribution does not contain the key 'distribution'.")
        if 'parameters' not in fitted_distributions[index].keys():
            raise TypeError("At least one dict inside 'fitted_distribution does not contain the key 'parameters'.")

    check_string(x_label, 'x_label')

    check_string(y_label, 'y_label')

    check_string(title, 'title')

    check_boolean(legend, 'legend')

    # Set default title
    if title == 'default':
        title = "Comparison between the best {0} fitting distributions.".format(best_n)

    # Create main plot
    plt.figure(figsize=(12, 8))
    ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=legend)
    y_lim = (ax.get_ylim()[0], ax.get_ylim()[1] * 1.2)
    x_lim = ax.get_xlim()

    # Plot the best n distributions
    for index in range(0, best_n):
        # Get distribution and parameter
        distribution_name = fitted_distributions[index]['distribution']
        distribution = getattr(st, distribution_name)
        parameters = fitted_distributions[index]['parameters']

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


def best_fitting_distribution(data, best_n=5, n_bins=200, verbose=False):
    """
    Go over all defined distributions and fit them to the data. Sort them by their SSE and return a list of
    dictionaries containing all fitted distribution parameters sorted by SSE in ascending order.

    :param data:                1-dimensional numpy array or pandas DataFrame of shape (m, 1)
    :param best_n               Integer. Number of best distributions to return.
    :param n_bins:              Integer. Number of bins for histogram.
    :param verbose:             True or False. True for verbose output

    :return:                    List of dictionaries. Containts all distributions and their fitted parameters and SSE.

    This is a modification of the code from the great answer from tmthydvnprt on stackoverflow.com:
    https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python#answer-37616966

    """
    # Check if input is valid
    check_integer(best_n, 'best_n')

    check_integer(n_bins, 'n_bins')

    check_boolean(verbose, 'verbose')

    # Results list
    results = []

    # Estimate fit for each distribution
    for distribution in _get_distributions():

        # Get fitting results for distribution
        distribution_fit = fit_distribution_to_data(data, distribution, n_bins=n_bins, verbose=verbose)

        # Write results to results-list
        if distribution_fit is not None:
            results.append(distribution_fit)

    # Sort ascending by SSE
    results.sort(key=lambda item: item['sse'])

    # Keep only the best n results and return them
    results = results[0:best_n]
    return results


def fit_distribution_to_data(data, distribution, n_bins=200, verbose=False):
    """
    Try to fit the given distribution to the data and get the respective parameters. Additionally, calculate
    the SSE. Return everything as dict.

    :param data:                1-dimensional numpy array or pandas DataFrame of shape (m, 1)
    :param distribution:        Distribution from scipy.stats (e.g. scipy.stats.norm)
    :param n_bins:              Interger. Number of bins for histogram.
    :param verbose:             True or False. True for verbose output.

    :return:                    Dictionary containing: distribution name (string), sse (np.float64), parameters (tuple of floats)


    This is a modification of the code from the great answer from tmthydvnprt on stackoverflow.com:
    https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python#answer-37616966

    """

    # Check if inputs are valid
    check_numpy_array_pandas_dataframe_series_1d(data, 'data')

    if distribution not in _get_distributions():
        raise TypeError("Distribution must be a scipy.stats distribution and defined in _get_distributions().")

    check_integer(n_bins, 'n_bins')
    if n_bins < 20:
        raise TypeError("Value for 'n_bins' must be at least 20 (200 would be better).")
    elif n_bins > data.size:
        raise TypeError("Value for 'n_bins' cannot be higher than the number of observations in the data.")

    check_boolean(verbose, 'verbose')

    # Get histogram and bin_edges of data
    histogram, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0

    # Try to fit the distribution
    try:
        # Ignore warnings from data that can't be fit
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            # Fit distribution to data
            parameters = distribution.fit(data)

            # Get parameters from the fitted distribution
            arg = parameters[:-2]
            mean = parameters[-2]
            standard_deviation = parameters[-1]

            # Calculate the fitted PDF
            pdf = distribution.pdf(bin_edges, loc=mean, scale=standard_deviation, *arg)

            # Calculate the SSE
            sse = np.sum(np.power(histogram - pdf, 2.0))

            # Create the result dictionary and return it
            result = {'distribution': distribution.name,
                      'sse': sse,
                      'parameters': parameters}

            return result

    # Catch all exceptions and print them if verbose is True
    except Exception as e:
        if verbose:
            print("Error at distribution '{0}':".format(distribution.name), e)

        return None


def _get_distributions():
    """
    :return:    List of scipy.stats distributions

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
    Generate the probability distribution function of a distribution.

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
        start= distribution.ppf(0.01, loc=mean, scale=standard_deviation)
        end = distribution.ppf(0.99, loc=mean, scale=standard_deviation)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = distribution.pdf(x, loc=mean, scale=standard_deviation, *arg)
    pdf = pd.Series(y, x)

    return pdf
