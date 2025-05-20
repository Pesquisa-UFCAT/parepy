"""Probabilistic Approach to Reliability Engineering (PAREPY)"""
import time
import copy
import os
import itertools
from datetime import datetime
from multiprocessing import Pool
from typing import Callable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, lognorm, gumbel_r, gumbel_l
from scipy.optimize import minimize

import parepy_toolbox.common_library as parepyco


def cornell_algorithm_structural_analysis(obj: Callable, vars: List[Dict], args: Optional[Tuple] = None) -> Tuple[float, float]:
    """
    Computes the Cornell reliability index and probability of failure.

    :param obj: Objective function that returns a list of limit state function (g) values.
    :param vars: Random variables configurations. Expect keys in each dictionary: "type", "parameters".
    :param args: Extra arguments to pass to the objective function (optional).

    :return: Results of reliability analysis.
            - pf_value: Probability of Failure (float).
            - beta_value: Reliability Index (float).
    """

    # Extract statistical parameters from variable configurations
    sigma = []
    var = []
    for var_config in vars:
        sigma.append(var_config['parameters']['mean'])
        var.append(var_config['parameters']['sigma'])

    # Evaluate limit-state function
    if args is not None:
        g = obj(sigma, args)
    else:
        g = obj(sigma)

    # Compute reliability metrics
    std_safety_margin = np.sqrt(sum(std ** 2 for std in var))
    beta_value = g / std_safety_margin
    pf_value = norm.cdf(-beta_value)

    return float(pf_value), float(beta_value)


def deterministic_algorithm_structural_analysis(obj: Callable, vars: List[Dict], tolerance: float, z0: List, args: Optional[Tuple] = None) -> Tuple[float, float]:
    """
    Computes the reliability index and probability of failure using FORM (First Order Reliability Method).

    :param obj: Objective function that returns a list of limit state function (g) values.
    :param vars: Random variables configurations. Expect keys in each dictionary: "type", "parameters".
    :param tolerance: Tolerance for convergence.
    :param z0: Initial guess.
    :param args: Extra arguments to pass to the objective function (optional).

    :return: Results of reliability analysis.
            - pf_value: Probability of Failure (float).
            - beta_value: Reliability Index (float).
    """

    # Extract statistical parameters from variable configurations
    pdf_type = []
    mu = []
    std = []
    for var_config in vars:
        pdf_type.append(var_config['type'])
        mu.append(var_config['parameters']['mean'])
        std.append(var_config['parameters']['sigma'])

    # Optimization function
    def funobj(z):
        d = np.sqrt(sum(z**2))
        return d

    # Transform random variables to standard normal space
    if args is not None:
        args_sbjto = (pdf_type, mu, std, args, obj)
        def sbjto(z, type, mu, std, args, obj):
            y = []
            for i, var in enumerate(type):
                if var == 'lognormal':
                    sigma_log_r = np.sqrt(np.log(1 + (std[i]**2) / (mu[i]**2)))
                    mu_log_r = np.log(mu[i]) - (sigma_log_r**2) / 2
                    scale_r = np.exp(mu_log_r)
                    z_aux = lognorm.ppf(norm.cdf(z[i]), s=sigma_log_r, scale=scale_r)
                elif var == 'gumbel max':
                    loc = mu[i] - np.sqrt(6)*(0.5772/np.pi)*std[i]
                    scale = np.sqrt(6)*std[i]/np.pi
                    z_aux = gumbel_r.ppf(norm.cdf(z[i]), loc, scale)
                elif var == 'gumbel min':
                    loc = mu[i] + np.sqrt(6) * (0.5772 / np.pi) * std[i]
                    scale = np.sqrt(6) * std[i] / np.pi
                    z_aux = gumbel_l.ppf(norm.cdf(z[i]), loc=loc, scale=scale)
                else:
                    z_aux = norm.ppf(norm.cdf(z[i]), mu[i], std[i])
                y.append(z_aux)
            g = obj(y, args)
            return g
    else:
        args_sbjto = (pdf_type, mu, std, obj)
        def sbjto(z, type, mu, std, obj):
            y = []
            for i, var in enumerate(type):
                if var == 'lognormal':
                    sigma_log_r = np.sqrt(np.log(1 + (std[i]**2) / (mu[i]**2)))
                    mu_log_r = np.log(mu[i]) - (sigma_log_r**2) / 2
                    scale_r = np.exp(mu_log_r)
                    z_aux = lognorm.ppf(norm.cdf(z[i]), s=sigma_log_r, scale=scale_r)
                elif var == 'gumbel max':
                    loc = mu[i] - np.sqrt(6)*(0.5772/np.pi)*std[i]
                    scale = np.sqrt(6)*std[i]/np.pi
                    z_aux = gumbel_r.ppf(norm.cdf(z[i]), loc, scale)
                elif var == 'gumbel min':
                    loc = mu[i] + np.sqrt(6) * (0.5772 / np.pi) * std[i]
                    scale = np.sqrt(6) * std[i] / np.pi
                    z_aux = gumbel_l.ppf(norm.cdf(z[i]), loc=loc, scale=scale)
                else:
                    z_aux = norm.ppf(norm.cdf(z[i]), mu[i], std[i])
                y.append(z_aux)
            g = obj(y)
            return g

    # Optimization procedure
    cons = {'type': 'eq', 'fun': sbjto, 'args': args_sbjto}
    res = minimize(funobj, z0, method='SLSQP', constraints=cons, tol=tolerance)
    z = res.x
    beta_value = res.fun
    pf_value = norm.cdf(-beta_value)

    return float(pf_value), float(beta_value)


def sampling_algorithm_structural_analysis_kernel(setup: dict) -> pd.DataFrame:
    """
    Creates samples and evaluates the limit state functions in structural reliability problems.

    :param objective_function: User-defined Python function to evaluate the limit state(s).
    :param number_of_samples: Number of samples to generate.
    :param numerical_model: Dictionary with model configuration (e.g., sampling type).
    :param variables_settings: List of variable definitions with distribution parameters.
    :param number_of_limit_functions: Number of limit state functions or constraints.
    :param none_variable: Optional auxiliary input to be passed to the objective function.

    :return: DataFrame with reliability analysis results.
    """

    # Ensure all variables have seeds
    for var in variables_settings:
        if 'seed' not in var:
            var['seed'] = None

    n_dimensions = len(variables_settings)
    algorithm = numerical_model['model sampling']
    is_time_analysis = algorithm.upper() in ['MCS-TIME', 'MCS_TIME', 'MCS TIME', 'LHS-TIME', 'LHS_TIME', 'LHS TIME']
    time_analysis = numerical_model['time steps'] if is_time_analysis else None

    # Generate samples
    dataset_x = parepyco.sampling(
        n_samples=number_of_samples,
        model=numerical_model,
        variables_setup=variables_settings
    )

    # Initialize output arrays
    capacity = np.zeros((len(dataset_x), number_of_limit_functions))
    demand = np.zeros((len(dataset_x), number_of_limit_functions))
    state_limit = np.zeros((len(dataset_x), number_of_limit_functions))
    indicator_function = np.zeros((len(dataset_x), number_of_limit_functions))

    # Evaluate objective function
    for idx, sample in enumerate(dataset_x):
        c_i, d_i, g_i = objective_function(list(sample), none_variable)
        capacity[idx, :] = c_i
        demand[idx, :] = d_i
        state_limit[idx, :] = g_i
        indicator_function[idx, :] = [1 if val <= 0 else 0 for val in g_i]

    # Stack all results
    results = np.hstack((dataset_x, capacity, demand, state_limit, indicator_function))

    # Format results into DataFrame
    if is_time_analysis:
        block_size = int(len(results) / number_of_samples)
        all_rows = []
        for i in range(number_of_samples):
            block = results[i * block_size:(i + 1) * block_size, :].T.flatten().tolist()
            all_rows.append(block)
        results_about_data = pd.DataFrame(all_rows)
    else:
        results_about_data = pd.DataFrame(results)

    # Create column names
    column_names = []

    for i in range(n_dimensions):
        if is_time_analysis:
            for t in range(time_analysis):
                column_names.append(f'X_{i}_t={t}')
        else:
            column_names.append(f'X_{i}')

    if is_time_analysis:
        for t in range(time_analysis):
            column_names.append(f'STEP_t_{t}')

    for i in range(number_of_limit_functions):
        if is_time_analysis:
            for t in range(time_analysis):
                column_names.append(f'R_{i}_t={t}')
        else:
            column_names.append(f'R_{i}')

    for i in range(number_of_limit_functions):
        if is_time_analysis:
            for t in range(time_analysis):
                column_names.append(f'S_{i}_t={t}')
        else:
            column_names.append(f'S_{i}')

    for i in range(number_of_limit_functions):
        if is_time_analysis:
            for t in range(time_analysis):
                column_names.append(f'G_{i}_t={t}')
        else:
            column_names.append(f'G_{i}')

    for i in range(number_of_limit_functions):
        if is_time_analysis:
            for t in range(time_analysis):
                column_names.append(f'I_{i}_t={t}')
        else:
            column_names.append(f'I_{i}')

    results_about_data.columns = column_names

    # First Barrier Failure (FBF) adjustment if time-dependent
    if is_time_analysis:
        results_about_data, _ = parepyco.fbf(
            algorithm,
            number_of_limit_functions,
            time_analysis,
            results_about_data
        )

    return results_about_data


def sampling_algorithm_structural_analysis(objective_function: callable, number_of_samples: int, numerical_model: dict, variables_settings: list, number_of_limit_functions: int, none_variable = None,name_simulation: str = None, verbose: bool = False) -> tuple[pd.DataFrame, list, list]:
    """
    Creates samples and evaluates limit state functions for structural reliability problems.

    :param objective_function: Python function to evaluate the limit state(s).
    :param number_of_samples: Number of samples to generate.
    :param numerical_model: Dictionary containing model configuration (e.g., sampling method).
    :param variables_settings: List of random variable definitions (distributions and parameters).
    :param number_of_limit_functions: Number of limit state functions or constraints.
    :param none_variable: Auxiliary variable used inside the objective function (optional).
    :param name_simulation: Optional output filename base (if saving is required).
    :param verbose: Boolean flag to print log messages (default is False).

    :return: 
        - results_about_data: DataFrame with simulation results.
        - failure_prob_list: List of failure probabilities.
        - beta_list: List of reliability indexes.
    """
    if verbose:
        parepyco.log_message('Checking input parameters...')

    if not isinstance(number_of_samples, int):
        raise TypeError('"number_of_samples" must be an integer.')
    if not isinstance(numerical_model, dict):
        raise TypeError('"numerical_model" must be a dictionary.')
    if not isinstance(variables_settings, list):
        raise TypeError('"variables_settings" must be a list.')
    if not isinstance(number_of_limit_functions, int):
        raise TypeError('"number_of_limit_functions" must be an integer.')
    if not callable(objective_function):
        raise TypeError('"objective_function" must be a Python function.')
    if not isinstance(name_simulation, (str, type(None))):
        raise TypeError('"name_simulation" must be a string or None.')

    if verbose:
        parepyco.log_message('Input check passed.')
        parepyco.log_message('Starting limit state function evaluation (g)...')

    algorithm = numerical_model['model sampling']
    div = number_of_samples // 10
    mod = number_of_samples % 10

    setups = []
    for i in range(10):
        samples = div + mod if i == 9 else div
        setups.append((
            objective_function,
            samples,
            numerical_model,
            variables_settings,
            number_of_limit_functions,
            none_variable
        ))

    start_time = time.perf_counter()
    with Pool() as pool:
        results = pool.starmap(sampling_algorithm_structural_analysis_kernel, setups)
    end_time = time.perf_counter()

    results_about_data = pd.concat(results, ignore_index=True)

    if verbose:
        parepyco.log_message(f'Finished (g) evaluation in {end_time - start_time:.2e} seconds.')
        parepyco.log_message('Starting evaluation of failure probability and reliability index (beta)...')

    start_time = time.perf_counter()
    failure_prob_list, beta_list = parepyco.calc_pf_beta(
        results_about_data,
        algorithm.upper(),
        number_of_limit_functions
    )
    end_time = time.perf_counter()

    if verbose:
        parepyco.log_message(f'Finished Pf and beta evaluation in {end_time - start_time:.2e} seconds.')

    if name_simulation:
        file_name = datetime.now().strftime('%Y%m%d-%H%M%S')
        file_name_txt = f'{name_simulation}_{algorithm.upper()}_{file_name}.txt'
        results_about_data.to_csv(file_name_txt, sep='\t', index=False)
        if verbose:
            parepyco.log_message(f'Results saved in {file_name_txt}')
    else:
        if verbose:
            parepyco.log_message('Simulation completed without saving.')

    return results_about_data, failure_prob_list, beta_list

def concatenates_txt_files_sampling_algorithm_structural_analysis(setup: dict) -> tuple[pd.DataFrame, list, list]:
    """
    Concatenates .txt files generated by the sampling_algorithm_structural_analysis algorithm and calculates
    probabilities of failure and reliability indexes based on the data.

    :param setup: Dictionary containing the input settings, including:

        - 'folder_path': Path to the folder containing the .txt files.
        - 'number of state limit functions or constraints': Number of limit state functions or constraints.
        - 'simulation name': Name of the simulation (string or None).

    :return: Tuple containing:

        - results_about_data: DataFrame with the concatenated results from the .txt files.
        - failure_prob_list: List of calculated failure probabilities for each indicator function.
        - beta_list: List of calculated reliability indexes (beta) for each indicator function.
    """


    try:
        # General settings
        if not isinstance(setup, dict):
            raise TypeError('The setup parameter must be a dictionary.')

        folder_path = setup['folder_path']
        algorithm = setup['numerical model']['model sampling']
        n_constraints = setup['number of state limit functions or constraints']

        # Check folder path
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f'The folder path {folder_path} does not exist.')

        # Concatenate files
        start_time = time.perf_counter()
        parepyco.log_message('Uploading files!')
        results_about_data = pd.DataFrame()
        for file_name in os.listdir(folder_path):
            # Check if the file has a .txt extension
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                temp_df = pd.read_csv(file_path, delimiter='\t')
                results_about_data = pd.concat([results_about_data, temp_df], ignore_index=True)
        end_time = time.perf_counter()
        final_time = end_time - start_time
        parepyco.log_message(f'Finished Upload in {final_time:.2e} seconds!')

        # Failure probability and beta index calculation
        parepyco.log_message('Started evaluation beta reliability index and failure probability...')
        start_time = time.perf_counter()
        failure_prob_list, beta_list = parepyco.calc_pf_beta(results_about_data, algorithm.upper(), n_constraints)
        end_time = time.perf_counter()
        final_time = end_time - start_time
        parepyco.log_message(f'Finished evaluation beta reliability index and failure probability in {end_time - start_time:.2e} seconds!')

        # Save results in .txt file
        if setup['name simulation'] is not None:
            name_simulation = setup['name simulation']
            file_name = str(datetime.now().strftime('%Y%m%d-%H%M%S'))
            file_name_txt = f'{name_simulation}_{algorithm.upper()}_{file_name}.txt'
            results_about_data.to_csv(file_name_txt, sep='\t', index=False)
            parepyco.log_message(f'Voilà!!!!....simulation results are saved in {file_name_txt}')
        else:
            parepyco.log_message('Voilà!!!!....simulation results were not saved in a text file!')

        return results_about_data, failure_prob_list, beta_list

    except (Exception, TypeError, ValueError) as e:
        print(f"Error: {e}")
        return None, None, None


def sobol_algorithm(setup):
    """
    Calculates the Sobol sensitivity indices in structural reliability problems.

    :param setup: Dictionary containing the input settings, including:

        - 'number of samples': Number of samples.
        - 'variables settings': List of variable definitions (as dictionaries).
        - 'number of state limit functions or constraints': Number of limit state functions or constraints.
        - 'none variable': Auxiliary variable used in the objective function (can be None, list, float, dict, str, etc.).
        - 'objective function': User-defined function to evaluate the limit state(s).

    :return: Dictionary containing the first-order and total-order Sobol sensitivity indices for each input variable.
    """

    n_samples = setup['number of samples']
    obj = setup['objective function']
    none_variable = setup['none variable']

    dist_a = sampling_algorithm_structural_analysis_kernel(setup)
    dist_b = sampling_algorithm_structural_analysis_kernel(setup)
    y_a = dist_a['G_0'].to_list()
    y_b = dist_b['G_0'].to_list()
    f_0_2 = (sum(y_a) / n_samples) ** 2

    A = dist_a.drop(['R_0', 'S_0', 'G_0', 'I_0'], axis=1).to_numpy()
    B = dist_b.drop(['R_0', 'S_0', 'G_0', 'I_0'], axis=1).to_numpy()
    K = A.shape[1]

    s_i = []
    s_t = []
    p_e = []
    for i in range(K):
        C = np.copy(B) 
        C[:, i] = A[:, i]
        y_c_i = []
        for j in range(n_samples):
            _, _, g = obj(list(C[j, :]), none_variable)
            y_c_i.append(g[0])  
        
        y_a_dot_y_c_i = [y_a[m] * y_c_i[m] for m in range(n_samples)]
        y_b_dot_y_c_i = [y_b[m] * y_c_i[m] for m in range(n_samples)]
        y_a_dot_y_a = [y_a[m] * y_a[m] for m in range(n_samples)]
        s_i.append((1/n_samples * sum(y_a_dot_y_c_i) - f_0_2) / (1/n_samples * sum(y_a_dot_y_a) - f_0_2))
        s_t.append(1 - (1/n_samples * sum(y_b_dot_y_c_i) - f_0_2) / (1/n_samples * sum(y_a_dot_y_a) - f_0_2))

    s_i = [float(i) for i in s_i]
    s_t = [float(i) for i in s_t]
    dict_sobol = pd.DataFrame(
        {'s_i': s_i,
         's_t': s_t}
    )

    return dict_sobol


def generate_factorial_design(level_dict):
    """
    Generates a full factorial design based on the input dictionary of variable levels.

    Computes all possible combinations of the provided levels for each variable and returns them in a structured DataFrame.

    :param level_dict: Dictionary where:
    
        - Keys represent variable names.
        - Values are lists, arrays, or sequences representing the levels of each variable.

    :return: DataFrame containing all possible combinations of the levels provided in the input dictionary.
             Each column corresponds to a variable defined in level_dict, and each row represents one combination
             of the factorial design.
    """

    combinations = list(itertools.product(*level_dict.values()))
    df = pd.DataFrame(combinations, columns=level_dict.keys())

    return df
