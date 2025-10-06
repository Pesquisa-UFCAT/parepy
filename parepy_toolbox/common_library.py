"""Common library for PAREpy toolbox"""
from typing import Optional, Callable

import scipy as sc
import numpy as np
import pandas as pd

import parepy_toolbox.distributions as parepydi


def std_matrix(std: list) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract D matrix and D^-1 matrix from a list of variables. Used in Y to X or X to Y transformation.

    :param std: Standard deviation parameters

    return: output[0] = D matrix, output[1] = D^-1 matrix
    """

    dneq = np.zeros((len(std), len(std)))
    dneq1 = np.zeros((len(std), len(std)))
    for i, sigma in enumerate(std):
        dneq[i, i] = sigma
        dneq1[i, i] = 1 / sigma

    return dneq, dneq1


def mu_matrix(mean: list) -> np.ndarray:
    """
    Extract mean matrix from a list of variables. Used in Y to X or X to Y transformation.

    :param mu: Mean parameters

    return: Mean matrix
    """

    mu_neq = np.zeros((len(mean), 1))
    for i, mu in enumerate(mean):
        mu_neq[i, 0] = mu

    return mu_neq


def x_to_y(x: np.ndarray, dneq1: np.ndarray, mu_neq: np.ndarray) -> np.ndarray:
    """
    Transforms a vector of random variables from the X space to the Y space.

    :param x: Random variables in the X space
    :param dneq1: D^-1 matrix
    :param mu_neq: Mean matrix

    :return: Transformed random variables in the Y space.
    """

    return dneq1 @ (x - mu_neq)


def y_to_x(y: np.ndarray, dneq: np.ndarray, mu_neq: np.ndarray) -> np.ndarray:
    """
    Transforms a vector of random variables from the Y space to the X space.

    :param y: Random variables in the Y space
    :param dneq: D matrix
    :param mu_neq: Mean matrix

    :return: Transformed random variables in the X space
    """

    return dneq @ y + mu_neq


def pf_equation(beta: float) -> float:
    """
    Calculates the probability of failure (pf) for a given reliability index (β), using the cumulative distribution function (CDF) of the standard normal distribution.

    :param beta: Reliability index (β)
    
    :return: Probability of failure (pf)

    Example
    ==============
    >>> # pip install parepy-toolbox or pip install -U parepy-toolbox
    >>> from parepy_toolbox import pf_equation
    >>> beta = 3.5
    >>> pf = pf_equation(beta)
    >>> print(f"Probability of failure: {pf:.5e}")
    Probability of failure: 2.32629e-04
    """

    return sc.stats.norm.cdf(-beta)


def beta_equation(pf: float) -> float:
    """
    Calculates the reliability index value for a given probability of failure (pf), using the inverse cumulative distribution function (ICDF) of the standard normal distribution.

    :param pf: Probability of failure (pf)

    :return: Reliability index (β)

    Example
    ==============
    >>> # pip install parepy-toolbox or pip install -U parepy-toolbox
    >>> from parepy_toolbox import beta_equation
    >>> pf = 2.32629e-04
    >>> beta = beta_equation(pf)
    >>> print(f"Reliability index: {beta:.5f}")
    Reliability index: 3.50000
    """

    return -sc.stats.norm.ppf(pf)


def sampling_kernel_without_time(obj: Callable, random_var_settings: list, method: str, n_samples: int, number_of_limit_functions: int, args: Optional[tuple] = None) -> pd.DataFrame:
    """
    Generates random samples from a specified distribution using kernel density estimation.

    :param obj: The objective function: obj(x, args) -> float or obj(x) -> float, where x is a list with shape n and args is a tuple fixed parameters needed to completely specify the function
    :param random_var_settings: Containing the distribution type and parameters. Example: {'type': 'normal', 'parameters': {'mean': 0, 'std': 1}}. Supported distributions: (a) 'uniform': keys 'min' and 'max', (b) 'normal': keys 'mean' and 'std', (c) 'lognormal': keys 'mean' and 'std', (d) 'gumbel max': keys 'mean' and 'std', (e) 'gumbel min': keys 'mean' and 'std', (f) 'triangular': keys 'min', 'mode' and 'max', or (g) 'gamma': keys 'mean' and 'std'
    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling), 'mcs' (Crude Monte Carlo Sampling) or 'sobol' (Sobol Sampling)
    :param n_samples: Number of samples. For Sobol sequences, this variable represents the exponent "m" (n = 2^m)
    :param number_of_limit_functions: Number of limit state functions or constraints
    :param args: Extra arguments to pass to the objective function (optional)

    :return: Random samples, objective function evaluations and indicator functions

    .. list-table:: **Table 1: Supported distributions**
       :widths: 20 40
       :header-rows: 1

       * - **Name**
         - **Expected parameters**
       * - ``"uniform"``
         - ``"min"`` and ``"max"``
       * - ``"normal"``
         - ``"mean"`` and ``"std"``
       * - ``"lognormal"``
         - ``"mean"`` and ``"std"``
       * - ``"gumbel max"``
         - ``"mean"`` and ``"std"``
       * - ``"gumbel min"``
         - ``"mean"`` and ``"std"``
       * - ``"triangular"``
         - ``"min"``, ``"mode"``, and ``"max"``
       * - ``"gamma"``
         - ``"mean"`` and ``"std"``

    Example
    ==============
    >>> # pip install parepy-toolbox or pip install -U parepy-toolbox
    >>> import time
    >>> from parepy_toolbox import sampling_kernel_without_time
    >>> def obj(x):
            g_0 = 12.5 * x[0] ** 3 - x[1]
            return [g_0]
    >>> d = {'type': 'normal', 'parameters': {'mean': 1.0, 'std': 0.1}}
    >>> l = {'type': 'normal', 'parameters': {'mean': 10.0, 'std': 1.0}}
    >>> var = [d, l]
    >>> number_of_limit_functions = 1
    >>> method = 'mcs'
    >>> n_samples = 10000
    >>> start = time.perf_counter()
    >>> df = sampling_kernel_without_time(obj, var, method, n_samples, number_of_limit_functions)
    >>> end = time.perf_counter()
    >>> print(f"Time elapsed: {(end-start):.5f}")
    >>> print(df)
            X_0        X_1        G_0     I_0
    0     1.193612  10.539209  10.717671  0.0
    1     1.041650  10.441663   3.686170  0.0
    2     1.133054   9.232075   8.950766  0.0
    3     0.983667  10.080005   1.817470  0.0
    4     0.908051  10.095981  -0.736729  1.0
    ...        ...        ...        ...  ...
    9995  1.016563   9.815083   3.316392  0.0
    9996  0.998764   9.623686   2.830013  0.0
    9997  0.826956   9.338711  -2.269712  1.0
    9998  1.060813   9.721774   5.200211  0.0
    9999  1.107219   9.239544   7.727668  0.0
    [10000 rows x 4 columns]
    """

    n_real_samples = 2**n_samples if method == 'sobol' else n_samples
    random_data = np.zeros((n_real_samples, len(random_var_settings)))

    # Generate random samples for each variable
    for i, dist_info in enumerate(random_var_settings):
        random_data[:, i] = parepydi.random_sampling(dist_info['type'], dist_info['parameters'], method, n_samples)

    # Evaluate objective function for each sample
    g_matrix = np.zeros((n_real_samples, number_of_limit_functions))
    indicator_matrix = np.zeros_like(g_matrix)
    for idx, sample in enumerate(random_data):
        g_values = obj(list(sample), args) if args is not None else obj(list(sample))
        g_matrix[idx, :] = g_values
        indicator_matrix[idx, :] = [1 if g <= 0 else 0 for g in g_values]

    # Build DataFrame
    df = pd.DataFrame(random_data, columns=[f'X_{i}' for i in range(len(random_var_settings))])
    for j in range(number_of_limit_functions):
        df[f'G_{j}'] = g_matrix[:, j]
        df[f'I_{j}'] = indicator_matrix[:, j]

    # dataset_x = {}
    # for i, value in enumerate(random_var_settings):
    #     dataset_x[f'X_{i}'] = parepydi.random_sampling(value['type'], value['parameters'], method, n_samples)
    # random_data = pd.DataFrame(dataset_x)
    # results = random_data.apply(lambda row: obj(list(row), args), axis=1) if args is not None else random_data.apply(lambda row: obj(list(row)), axis=1)
    # g_names = []
    # for i in range(number_of_limit_functions):
    #     g_names.append(f'G_{i}')
    # random_data[g_names] = pd.DataFrame(results.tolist(), index=random_data.index)
    # for col in g_names:
    #     random_data[f'I_{col}'] = np.where(random_data[col] <= 0, 1, 0)

    return df


# def sampling(n_samples: int, model: dict, variables_setup: list) -> np.ndarray:
#     """
#     Generates a set of random numbers according to a specified probability distribution model.

#     :param n_samples: Number of samples to generate.
#     :param model: Dictionary containing the model parameters.
#     :param variables_setup: List of dictionaries, each containing parameters for a random variable.

#     :return: Numpy array with the generated random samples.
#     """

#     # Model settings
#     model_sampling = model['model sampling'].upper()
#     id_type = []
#     id_corr = []
#     for v in variables_setup:
#         if 'parameters' in v and 'corr' in v['parameters']:
#             id_type.append('g-corr-g_var')
#             id_corr.append(v['parameters']['corr']['var'])
#         else:
#             id_type.append('g')
#     for k in id_corr:
#         id_type[k] = 'g-corr-b_var'

#     if model_sampling in ['MCS', 'LHS']:
#         random_sampling = np.zeros((n_samples, len(variables_setup)))

#         for j, variable in enumerate(variables_setup):
#             if id_type[j] == 'g-corr-b_var':
#                 continue
#             type_dist = variable['type'].upper()
#             seed_dist = variable['seed']
#             params = variable['parameters']

#             if (type_dist == 'NORMAL' or type_dist == 'GAUSSIAN') and id_type[j] == 'g':
#                 mean = params['mean']
#                 sigma = params['sigma']
#                 parameters = {'mean': mean, 'sigma': sigma}
#                 random_sampling[:, j] = parepydi.normal_sampling(parameters, method=model_sampling.lower(), n_samples=n_samples, seed=seed_dist)

#             elif (type_dist == 'NORMAL' or type_dist == 'GAUSSIAN') and id_type[j] == 'g-corr-g_var':
#                 mean = params['mean']
#                 sigma = params['sigma']
#                 parameters_g = {'mean': mean, 'sigma': sigma}
#                 pho = params['corr']['pho']
#                 m = params['corr']['var']
#                 parameters_b = variables_setup[m]['parameters']
#                 random_sampling[:, m], random_sampling[:, j] = parepydi.corr_normal_sampling(parameters_b, parameters_g, pho, method=model_sampling.lower(), n_samples=n_samples, seed=seed_dist)

#             elif type_dist == 'UNIFORM' and id_type[j] == 'g':
#                 min_val = params['min']
#                 max_val = params['max']
#                 parameters = {'min': min_val, 'max': max_val}
#                 random_sampling[:, j] = parepydi.uniform_sampling(parameters, method=model_sampling.lower(), n_samples=n_samples, seed=seed_dist)

#             elif type_dist == 'GUMBEL MAX' and id_type[j] == 'g':
#                 mean = params['mean']
#                 sigma = params['sigma']
#                 parameters = {'mean': mean, 'sigma': sigma}
#                 random_sampling[:, j] = parepydi.gumbel_max_sampling(parameters, method=model_sampling.lower(), n_samples=n_samples, seed=seed_dist)

#             elif type_dist == 'GUMBEL MIN' and id_type[j] == 'g':
#                 mean = params['mean']
#                 sigma = params['sigma']
#                 parameters = {'mean': mean, 'sigma': sigma}
#                 random_sampling[:, j] = parepydi.gumbel_min_sampling(parameters, method=model_sampling.lower(), n_samples=n_samples, seed=seed_dist)

#             elif type_dist == 'LOGNORMAL' and id_type[j] == 'g':
#                 mean = params['mean']
#                 sigma = params['sigma']
#                 parameters = {'mean': mean, 'sigma': sigma}
#                 random_sampling[:, j] = parepydi.lognormal_sampling(parameters, method=model_sampling.lower(), n_samples=n_samples, seed=seed_dist)

#             elif type_dist == 'TRIANGULAR' and id_type[j] == 'g':
#                 min_val = params['min']
#                 max_val = params['max']
#                 mode = params['mode']
#                 parameters = {'min': min_val, 'max': max_val, 'mode': mode}
#                 random_sampling[:, j] = parepydi.triangular_sampling(parameters, method=model_sampling.lower(), n_samples=n_samples, seed=seed_dist)
#     elif model_sampling in ['MCS-TIME', 'MCS_TIME', 'MCS TIME', 'LHS-TIME', 'LHS_TIME', 'LHS TIME']:
#         time_analysis = model['time steps']
#         random_sampling = np.empty((0, len(variables_setup)))
#         match = re.search(r'\b(MCS|LHS)\b', model_sampling.upper(), re.IGNORECASE)
#         model_sampling = match.group(1).upper()

#         for _ in range(n_samples):
#             temporal_sampling = np.zeros((time_analysis, len(variables_setup)))

#             for j, variable in enumerate(variables_setup):
#                 if id_type[j] == 'g-corr-b_var':
#                     continue
#                 type_dist = variable['type'].upper()
#                 seed_dist = variable['seed']
#                 sto = variable['stochastic variable']
#                 params = variable['parameters']

#                 if (type_dist == 'NORMAL' or type_dist == 'GAUSSIAN') and id_type[j] == 'g':
#                     mean = params['mean']
#                     sigma = params['sigma']
#                     parameters = {'mean': mean, 'sigma': sigma}
#                     if sto is False:
#                         temporal_sampling[:, j] = parepydi.normal_sampling(parameters, method=model_sampling.lower(), n_samples=1, seed=seed_dist)
#                         temporal_sampling[1:, j]
#                     else:
#                         temporal_sampling[:, j] = parepydi.normal_sampling(parameters, method=model_sampling.lower(), n_samples=time_analysis, seed=seed_dist)

#                 elif (type_dist == 'NORMAL' or type_dist == 'GAUSSIAN') and id_type[j] == 'g-corr-g_var':
#                     mean = params['mean']
#                     sigma = params['sigma']
#                     parameters_g = {'mean': mean, 'sigma': sigma}
#                     pho = params['corr']['pho']
#                     m = params['corr']['var']
#                     parameters_b = variables_setup[m]['parameters']
#                     if sto is False:
#                         temporal_sampling[:, m], temporal_sampling[:, j] = parepydi.corr_normal_sampling(parameters_b, parameters_g, pho, method=model_sampling.lower(), n_samples=1, seed=seed_dist)
#                         temporal_sampling[1:, j]
#                         temporal_sampling[1:, m]
#                     else:
#                         temporal_sampling[:, m], temporal_sampling[:, j] = parepydi.corr_normal_sampling(parameters_b, parameters_g, pho, method=model_sampling.lower(), n_samples=time_analysis, seed=seed_dist)

#                 elif type_dist == 'UNIFORM' and id_type[j] == 'g':
#                     min_val = params['min']
#                     max_val = params['max']
#                     parameters = {'min': min_val, 'max': max_val}
#                     if sto is False:
#                         temporal_sampling[:, j] = parepydi.uniform_sampling(parameters, method=model_sampling.lower(), n_samples=1, seed=seed_dist)
#                         temporal_sampling[1:, j]
#                     else:
#                         temporal_sampling[:, j] = parepydi.uniform_sampling(parameters, method=model_sampling.lower(), n_samples=time_analysis, seed=seed_dist)

#                 elif type_dist == 'GUMBEL MAX' and id_type[j] == 'g':
#                     mean = params['mean']
#                     sigma = params['sigma']
#                     parameters = {'mean': mean, 'sigma': sigma}
#                     if sto is False:
#                         temporal_sampling[:, j] = parepydi.gumbel_max_sampling(parameters, method=model_sampling.lower(), n_samples=1, seed=seed_dist)
#                         temporal_sampling[1:, j]
#                     else:
#                         temporal_sampling[:, j] = parepydi.gumbel_max_sampling(parameters, method=model_sampling.lower(), n_samples=time_analysis, seed=seed_dist)

#                 elif type_dist == 'GUMBEL MIN' and id_type[j] == 'g':
#                     mean = params['mean']
#                     sigma = params['sigma']
#                     parameters = {'mean': mean, 'sigma': sigma}
#                     if sto is False:
#                         temporal_sampling[:, j] = parepydi.gumbel_min_sampling(parameters, method=model_sampling.lower(), n_samples=1, seed=seed_dist)
#                         temporal_sampling[1:, j]
#                     else:
#                         temporal_sampling[:, j] = parepydi.gumbel_min_sampling(parameters, method=model_sampling.lower(), n_samples=time_analysis, seed=seed_dist)

#                 elif type_dist == 'LOGNORMAL' and id_type[j] == 'g':
#                     mean = params['mean']
#                     sigma = params['sigma']
#                     parameters = {'mean': mean, 'sigma': sigma}
#                     if sto is False:
#                         temporal_sampling[:, j] = parepydi.lognormal_sampling(parameters, method=model_sampling.lower(), n_samples=1, seed=seed_dist)
#                         temporal_sampling[1:, j]
#                     else:
#                         temporal_sampling[:, j] = parepydi.lognormal_sampling(parameters, method=model_sampling.lower(), n_samples=time_analysis, seed=seed_dist)

#                 elif type_dist == 'TRIANGULAR' and id_type[j] == 'g':
#                     min_val = params['min']
#                     max_val = params['max']
#                     mode = params['mode']
#                     parameters = {'min': min_val, 'max': max_val, 'mode': mode}
#                     if sto is False:
#                         temporal_sampling[:, j] = parepydi.triangular_sampling(parameters, method=model_sampling.lower(), n_samples=1, seed=seed_dist)
#                         temporal_sampling[1:, j]
#                     else:
#                         temporal_sampling[:, j] = parepydi.triangular_sampling(parameters, method=model_sampling.lower(), n_samples=time_analysis, seed=seed_dist)

#             random_sampling = np.concatenate((random_sampling, temporal_sampling), axis=0)  

#         time_sampling = np.zeros((time_analysis * n_samples, 1))
#         cont = 0
#         for _ in range(n_samples):
#             for m in range(time_analysis):
#                 time_sampling[cont, 0] = int(m)
#                 cont += 1
#         random_sampling = np.concatenate((random_sampling, time_sampling), axis=1)   

#     return random_sampling


# def calc_pf_beta(df_or_path: Union[pd.DataFrame, str], numerical_model: str, n_constraints: int) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Calculates the probability of failure (pf) and reliability index (β) based on the columns of a DataFrame
#     that start with 'I' (indicator function). If a .txt file path is passed, this function evaluates pf and β values too.

#     :param df_or_path: A DataFrame containing boolean indicator columns prefixed with 'I', or a string path to a .txt file.

#     :param numerical_model: Dictionary containing the numerical model.

#     :param n_constraints: Number of limit state functions or constraints.

#     :return: Tuple of DataFrames:

#         - df_pf: probability of failure values for each column prefixed with 'G'.
#         - df_beta: reliability index values for each column prefixed with 'G'.
#     """

#     # Read dataset
#     if isinstance(df_or_path, str) and df_or_path.endswith('.txt'):
#         df = pd.read_csv(df_or_path, delimiter='\t')
#     else:
#         df = df_or_path

#     # Calculate pf and beta values
#     if numerical_model.upper() in ['MCS', 'LHS']:
#         filtered_df = df.filter(like='I_', axis=1)
#         pf_results = filtered_df.mean(axis=0)
#         df_pf = pd.DataFrame([pf_results.to_list()], columns=pf_results.index)
#         beta_results = [beta_equation(pf) for pf in pf_results.to_list()] 
#         df_beta = pd.DataFrame([beta_results], columns=pf_results.index)
#     elif numerical_model.upper() in ['TIME-MCS', 'TIME-LHS', 'TIME MCS', 'TIME LHS', 'MCS TIME', 'LHS TIME', 'MCS-TIME', 'LHS-TIME']:
#         df_pf = pd.DataFrame()
#         df_beta = pd.DataFrame()
#         for i in range(n_constraints):
#             filtered_df = df.filter(like=f'I_{i}', axis=1)
#             pf_results = filtered_df.mean(axis=0)
#             beta_results = [beta_equation(pf) for pf in pf_results.to_list()]
#             df_pf[f'G_{i}'] = pf_results.to_list()
#             df_beta[f'G_{i}'] = beta_results

#     return df_pf, df_beta


# def fbf(algorithm: str, n_constraints: int, time_analysis: int, results_about_data: pd.DataFrame) -> tuple[pd.DataFrame, list]:
#     """
#     This function application first barrier failure algorithm.

#     :param algorithm: Name of the algorithm.
#     :param n_constraints: Number of constraints analyzed.
#     :param time_analysis: Time period for analysis.
#     :param results_about_data: DataFrame containing the results to be processed.

#     :return: Updated DataFrame after processing.
#     """

#     if algorithm.upper() in ['MCS-TIME', 'MCS_TIME', 'MCS TIME']:
#         i_columns = []
#         for i in range(n_constraints):
#             aux_column_names = []
#             for j in range(time_analysis):
#                 aux_column_names.append('I_' + str(i) + '_t=' + str(j))
#             i_columns.append(aux_column_names)

#         for i in i_columns:
#             matrixx = results_about_data[i].values
#             for id, linha in enumerate(matrixx):
#                 indice_primeiro_1 = np.argmax(linha == 1)
#                 if linha[indice_primeiro_1] == 1:
#                     matrixx[id, indice_primeiro_1:] = 1
#             results_about_data = pd.concat([results_about_data.drop(columns=i),
#                                             pd.DataFrame(matrixx, columns=i)], axis=1)
#     else:
#         i_columns = []
#         for i in range(n_constraints):
#             i_columns.append(['I_' + str(i)])
    
#     return results_about_data, i_columns


# def log_message(message: str) -> None:
#     """
#     Logs a message with the current time.

#     :param message: The message to log.
    
#     :return: None
#     """
#     current_time = datetime.now().strftime('%H:%M:%S')
#     print(f'{current_time} - {message}')


# def norm_array(ar: list) -> float:
#     """
#     Evaluates the norm of the array ar.

#     :param ar: A list of numerical values (floats) representing the array.

#     :return: The norm of the array.
#     """
#     norm_ar = [i ** 2 for i in ar]
#     norm_ar = sum(norm_ar) ** 0.5
#     return norm_ar


# def hasofer_lind_rackwitz_fiessler_algorithm(y_k: np.ndarray, g_y: float, grad_y_k: np.ndarray) -> np.ndarray:
#     """
#     Calculates the new y value using the Hasofer-Lind-Rackwitz-Fiessler algorithm.

#     :param y_k: Current y value.
#     :param g_y: Objective function at point `y_k`.
#     :param grad_y_k: Gradient of the objective function at point `y_k`.

#     :return: New y value.
#     """

#     num = np.dot(np.transpose(grad_y_k), y_k) - np.array([[g_y]])
#     print("num: ", num)
#     num = num[0][0]
#     den = (np.linalg.norm(grad_y_k)) ** 2
#     print("den: ", den)
#     aux = num / den
#     y_new = aux * grad_y_k

#     return y_new

# def goodness_of_fit(data: Union[np.ndarray, list], distributions: Union[str, list] = 'all') -> dict:
#     """
#     Evaluates the fit of distributions to the provided data.

#     This function fits various distributions to the data using the distfit library and returns the top three distributions based on the fit score.

#     Args:
#         data (np.array or list): Data to which distributions will be fitted. It should be a list or array of numeric values.
#         distributions (str or list, optional): Distributions to be tested. If 'all', all available distributions will be tested. Otherwise, it should be a list of strings specifying the names of the distributions to test. The default is 'all'.

#     Returns:
#         dict: A dictionary containing the top three fitted distributions. Each entry is a dictionary with the following keys:
#             - 'rank': Ranking of the top three distributions based on the fit score.
#             - 'type' (str): The name of the fitted distribution.
#             - 'params' (tuple): Parameters of the fitted distribution.
    
#     Raises:
#         ValueError: If the expected 'score' column is not present in the DataFrame returned by `dist.summary()`.
#     """

#     if distributions == 'all':
#         dist = distfit()
#     else:
#         dist = distfit(distr=distributions)
    
#     dist.fit_transform(data)
#     summary_df = dist.summary
#     sorted_models = summary_df.sort_values(by='score').head(3)
    
#     top_3_distributions = {
#         f'rank_{i+1}': {
#             'type': model['name'],
#             'params': model['params']
#         }
#         for i, model in sorted_models.iterrows()
#     }
    
#     return top_3_distributions

# def newton_raphson(f: Callable, df: Callable, x0: float, tol: float) -> float:
#     """
#     Calculates the root of a function using the Newton-Raphson method.

#     :param f: Function for which the root is sought.
#     :param df: Derivative of the function.
#     :param x0: Initial guess for the root.
#     :param tol: Tolerance for convergence.

#     :return: Approximated root of the function.
#     """

#     if abs(f(x0)) < tol:
#         return x0
#     else:
#         return newton_raphson(f, df, x0 - f(x0)/df(x0), tol)