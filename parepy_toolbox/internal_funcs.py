"""Internal functions for PAREpy toolbox. This module contains functions that are not intended to be used directly by users but are essential for the internal workings of the toolbox."""
from typing import Optional, Callable

import numpy as np
import pandas as pd
import scipy as sc

import parepy_toolbox.common_library as parepyco


def first_second_order_derivative_numerical_differentiation_unidimensional(obj: Callable, x: list, pos: str, method: str, order: str = 'first', h: float = 1E-5, args: Optional[tuple] = None) -> float:
    """
    Computes the numerical derivative of a function at a given point in the given dimension using the central, backward and forward difference method.

    :param obj: The objective function: obj(x, args) -> float or obj(x) -> float, where x is a list with shape n and args is a tuple fixed parameters needed to completely specify the function
    :param x: Point at which to evaluate the derivative
    :param pos: Dimension in the list x where the derivative is to be calculated. When use order 'xy', pos is a str contain first and second dimension separated by a comma (e.g., '0,1' for the first and second dimensions)
    :param method: Method to use for differentiation. Supported values: 'center', 'forward', or 'backward'
    :param order: Order of the derivative to compute (default is first for first-order derivative). Supported values: 'first', 'second', or 'xy'
    :param h: Step size for the finite difference approximation (default is 1e-10)
    :param args: Extra arguments to pass to the objective function (optional)

    :return: Numerical derivative of order n of the function at point x in dimension pos
    """

    if order == 'first':
        a = x.copy()
        b = x.copy()
        if method == "forward":
            for i in range(len(x)):
                if i == int(pos):
                    a[i] += h
            den = h
        elif method == "backward":
            for i in range(len(x)):
                if i == int(pos):
                    b[i] -= h
            den = h
        elif method == "center":
            for i in range(len(x)):
                if i == int(pos):
                    a[i] += h
                    b[i] -= h
            den = 2 * h
        fa = obj(a, args) if args is not None else obj(a)
        fb = obj(b, args) if args is not None else obj(b)
        diff = (fa - fb) / den
    elif order == 'second':
        a = x.copy()
        b = x.copy()
        x_aux = x.copy()
        den = h ** 2
        if method == "forward":
            for i in range(len(x)):
                if i == int(pos):
                    a[i] += 2*h
                    x_aux[i] += h
            fa = obj(a, args) if args is not None else obj(a)
            fx = obj(x_aux, args) if args is not None else obj(x_aux)
            fb = obj(b, args) if args is not None else obj(b)
        elif method == "backward":
            for i in range(len(x)):
                if i == int(pos):
                    b[i] -= 2*h
                    x_aux[i] -= h
            fa = obj(a, args) if args is not None else obj(a)
            fx = obj(x_aux, args) if args is not None else obj(x_aux)
            fb = obj(b, args) if args is not None else obj(b)
        elif method == "center":
            for i in range(len(x)):
                if i == int(pos):
                    a[i] += h
                    b[i] -= h
            fa = obj(a, args) if args is not None else obj(a)
            fx = obj(x_aux, args) if args is not None else obj(x_aux)
            fb = obj(b, args) if args is not None else obj(b)
        diff = (fa - 2 * fx + fb) / den
    elif order == 'xy':
        pos_x = int(pos.split(',')[0])
        pos_y = int(pos.split(',')[1])
        a = x.copy()
        a[pos_x] += h
        a[pos_y] += h
        b = x.copy()
        b[pos_x] += h
        b[pos_y] -= h
        c = x.copy()
        c[pos_x] -= h
        c[pos_y] += h
        d = x.copy()
        d[pos_x] -= h
        d[pos_y] -= h
        fa = obj(a, args) if args is not None else obj(a)
        fb = obj(b, args) if args is not None else obj(b)
        fc = obj(c, args) if args is not None else obj(c)
        fd = obj(d, args) if args is not None else obj(d)
        diff = (fa - fb - fc + fd) / (4 * h ** 2)

    return diff


def jacobian_matrix(obj: Callable, x: list, method: str, h: float = 1E-5, args: Optional[tuple] = None) -> np.ndarray:
    """
    Computes Jacobian matrix of a vector-valued function using finite difference methods.

    :param obj: The objective function: obj(x, args) -> float or obj(x) -> float, where x is a list with shape n and args is a tuple fixed parameters needed to completely specify the function
    :param x: Point at which to evaluate the derivative
    :param method: Method to use for differentiation. Supported values: 'center', 'forward', or 'backward'
    :param h: Step size for the finite difference approximation (default is 1e-5)
    :param args: Extra arguments to pass to the objective function (optional)

    :return: Numerical Jacobian matrix at point x
    """

    jacob = np.zeros((len(x), 1))
    for i in range(len(x)):
        jacob[i, 0] = first_second_order_derivative_numerical_differentiation_unidimensional(obj, x, str(i), method, 'first', h=h, args=args) if args is not None else first_second_order_derivative_numerical_differentiation_unidimensional(obj, x, str(i), method, 'first', h=h)

    return jacob


def hessian_matrix(obj: Callable, x: list, method: str, h: float = 1E-5, args: Optional[tuple] = None) -> np.ndarray:
    """
    Computes Hessian matrix of a vector-valued function using finite difference methods.

    :param obj: The objective function: obj(x, args) -> float or obj(x) -> float, where x is a list with shape n and args is a tuple fixed parameters needed to completely specify the function
    :param x: Point at which to evaluate the derivative
    :param method: Method to use for differentiation. Supported values: 'center', 'forward', or 'backward'
    :param h: Step size for the finite difference approximation (default is 1e-5)
    :param args: Extra arguments to pass to the objective function (optional)

    :return: Numerical Hessian matrix at point x
    """

    hessian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                hessian[i, j] = first_second_order_derivative_numerical_differentiation_unidimensional(obj, x, str(i), method, 'second', h=h, args=args) if args is not None else first_second_order_derivative_numerical_differentiation_unidimensional(obj, x, str(i), method, 'second', h=h)
            else:
                hessian[i, j] = first_second_order_derivative_numerical_differentiation_unidimensional(obj, x, f'{j},{i}', method, 'xy', h=h, args=args) if args is not None else first_second_order_derivative_numerical_differentiation_unidimensional(obj, x, f'{j},{i}', method, 'xy', h=h)

    return hessian


def summarize_pf_beta(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates a summary DataFrame containing the probability of failure (pf) and reliability index (β) for each indicator function column in the input DataFrame.

    :param df: Random samples, objective function evaluations and indicator functions

    :return: output [0] = Probability of failure values for each indicator function, output[1] = Reliability index values for each indicator function

    """
    
    pf_values = {}
    beta_values = {}

    for col in df.columns:
        if col.startswith("I_"):
            idx = col.split("_")[1]
            pf = df[col].mean()
            beta = parepyco.beta_equation(pf)
            pf_values[f"pf_{idx}"] = pf
            beta_values[f"beta_{idx}"] = beta

    pf_df = pd.DataFrame([pf_values])
    beta_df = pd.DataFrame([beta_values])

    return pf_df, beta_df


def convergence_probability_failure(df: pd.DataFrame, column: str) -> tuple[list, list, list, list, list]:
    """
    Calculates the convergence rate of the probability of failure.

    :param df: Random samples, objective function evaluations and indicator functions
    :param column: Name of the column to be analyzed. Supported values: 'I_0', 'I_1', ..., 'I_n' where n is the number of limit state functions

    :return: output[0] = Sample sizes considered at each step, output[1] = m: Mean values (estimated probability of failure), output[2] = Lower confidence interval values of the column, output[3] = ci_u: Upper confidence interval values of the column, output[4] = Variance values of the column
    """

    column_values = df[column].to_list()
    step = 1000
    div = [i for i in range(step, len(column_values), step)]
    m = []
    ci_u = []
    ci_l = []
    var = []
    for i in range(0, len(div)+1):
        if i == len(div):
            aux = column_values.copy()
            div.append(len(column_values))
        else:
            aux = column_values[:div[i]]
        mean = np.mean(aux)
        std = np.std(aux, ddof=1)
        n = len(aux)
        confidence_level = 0.95
        t_critic = sc.stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        margin = t_critic * (std / np.sqrt(n))
        confidence_interval = (mean - margin, mean + margin)
        m.append(mean)
        ci_u.append(confidence_interval[1])
        ci_l.append(confidence_interval[0])
        var.append((mean * (1 - mean))/n)

    return div, m, ci_l, ci_u, var


def convert_params_to_scipy(dist: str, parameters: dict) -> dict:
    """
    Convert user-provided distribution parameters to the format required by "scipy.stats".

    :param parameters: Original distribution parameters. (a) 'uniform': keys 'min' and 'max', (b) 'normal': keys 'mean' and 'std', (c) 'lognormal': keys 'mean' and 'std', (d) 'gumbel max': keys 'mean' and 'std', (e) 'gumbel min': keys 'mean' and 'std', (f) 'triangular': keys 'min', 'mode' and 'max', or (g) 'gamma': keys 'mean' and 'std'

    :return: Distribution parameters according scipy.stats documentation
    """   

    if dist.lower() == 'uniform':
        parameters_scipy = {'loc': parameters['min'], 'scale': parameters['max'] - parameters['min']}
    elif dist.lower() == 'normal':
        parameters_scipy = {'loc': parameters['mean'], 'scale': parameters['std']}
    elif dist.lower() == 'lognormal':
        epsilon = np.sqrt(np.log(1 + (parameters['std'] / parameters['mean']) ** 2))
        lambda_ = np.log(parameters['mean']) - 0.5 * epsilon ** 2
        parameters_scipy = {'s': epsilon, 'loc': 0.0, 'scale': np.exp(lambda_)}
    elif dist.lower() == 'gumbel max':
        gamma = 0.5772156649015329
        alpha = parameters['std'] * np.sqrt(6) / np.pi
        beta = parameters['mean'] - alpha * gamma
        parameters_scipy = {'loc': beta, 'scale': alpha}
    elif dist.lower() == 'gumbel min':
        gamma = 0.5772156649015329
        alpha = parameters['std'] * np.sqrt(6) / np.pi
        beta = parameters['mean'] + alpha * gamma
        parameters_scipy = {'loc': beta, 'scale': alpha}
    elif dist.lower() == 'triangular':
        parameters_scipy = {'c': (parameters['mode'] - parameters['min']) / (parameters['max'] - parameters['min']), 'loc': parameters['min'], 'scale': parameters['max'] - parameters['min']}
    elif dist.lower() == 'gamma':
        a = (parameters['mean'] / parameters['std']) ** 2
        scale = parameters['std'] ** 2 / parameters['mean']
        parameters_scipy = {'a': a, 'loc': 0.0, 'scale': scale}

    return parameters_scipy
