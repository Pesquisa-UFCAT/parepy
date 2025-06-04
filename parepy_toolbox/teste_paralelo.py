import time
import pandas as pd
import numpy as np
from typing import Callable, Any, Optional
from multiprocessing import Pool

import parepy_toolbox.distributions as parepydi

def sampling_kernel_without_time(obj: Callable, random_var_settings: list, method: str, n_samples: int, number_of_limit_functions: int, args: Optional[tuple] = None) -> pd.DataFrame:
    """
    Generates random samples from a specified distribution using kernel density estimation.

    :param random_var_settings: Containing the distribution type and parameters. Example: {'type': 'normal', 'parameters': {'mean': 0, 'std': 1}}.
    :param method: Sampling method. Supported values: 'lhs' (Latin Hypercube Sampling), 'mcs' (Crude Monte Carlo Sampling) or 'sobol' (Sobol Sampling).
    :param n_samples: Number of samples. For Sobol sequences, this variable represents the exponent "m" (n = 2^m).

    :return: Random samples.
    """

    n_vars = len(random_var_settings)
    random_data = np.zeros((n_samples, n_vars))
    
    # Generate random samples for each variable
    for i, dist_info in enumerate(random_var_settings):
        random_data[:, i] = parepydi.random_sampling(
            dist_info['type'], dist_info['parameters'], method, n_samples
        )

    # Evaluate objective function for each sample
    g_matrix = np.zeros((n_samples, number_of_limit_functions))
    indicator_matrix = np.zeros_like(g_matrix)

    for idx, sample in enumerate(random_data):
        g_values = obj(list(sample), *args) if args else obj(list(sample))
        g_matrix[idx, :] = g_values
        indicator_matrix[idx, :] = [1 if g <= 0 else 0 for g in g_values]

    # Build DataFrame
    df = pd.DataFrame(random_data, columns=[f'X{i}' for i in range(n_vars)])
    for j in range(number_of_limit_functions):
        df[f'g_{j}'] = g_matrix[:, j]
        df[f'I_{j}'] = indicator_matrix[:, j]

    return df   

def sampling_algorithm_structural_analysis_(
    objective_function: Callable,
    number_of_samples: int,
    method: str,
    variables_settings: list,
    number_of_limit_functions: int,
    none_variable: Optional[object],
    block_size: int
) -> pd.DataFrame:

    # Cria blocos de amostragem
    n_blocks = number_of_samples // block_size
    remainder = number_of_samples % block_size

    # Cada entrada da lista setups é uma tupla com todos os argumentos que serão passados à função
    setups = [
        (objective_function, variables_settings, method, block_size, number_of_limit_functions, (none_variable,))
        for _ in range(n_blocks)
    ]

    if remainder > 0:
        setups.append((
            objective_function, variables_settings, method, remainder, number_of_limit_functions, (none_variable,)
        ))

    start_time = time.perf_counter()

    # Executa os blocos em paralelo com starmap
    with Pool() as pool:
        results = pool.starmap(sampling_kernel_without_time, setups)

    end_time = time.perf_counter()
    print(f"Amostragem concluída em {end_time - start_time:.2f} segundos.")

    return pd.concat(results, ignore_index=True)


def sampling_algorithm_structural_analysis_loop(
    objective_function: Callable,
    number_of_samples: int,
    method: str,
    variables_settings: list,
    number_of_limit_functions: int,
    none_variable: Optional[object],
    block_size: int
) -> pd.DataFrame:
    """
    Versão sequencial do algoritmo de amostragem estrutural.
    Processa os blocos um por um, sem paralelização.
    """
    n_blocks = number_of_samples // block_size
    remainder = number_of_samples % block_size

    setups = [
        (objective_function, variables_settings, method, block_size, number_of_limit_functions, (none_variable,))
        for _ in range(n_blocks)
    ]

    if remainder > 0:
        setups.append((
            objective_function, variables_settings, method, remainder, number_of_limit_functions, (none_variable,)
        ))

    start_time = time.perf_counter()

    results = []
    for args in setups:
        result = sampling_kernel_without_time(*args)
        results.append(result)

    end_time = time.perf_counter()
    print(f"Amostragem concluída em {end_time - start_time:.2f} segundos.")

    return pd.concat(results, ignore_index=True)



## ===========# Exemplo de uso do algoritmo #========== ##


def obj(x, *args):
    g_0 = 12.5 * x[0] ** 3 - x[1]
    time.sleep(10**-6) 
    return [g_0]

if __name__ == "__main__":
    d = {'type': 'normal', 'parameters': {'mean': 1.0, 'std': 0.1}}
    l = {'type': 'normal', 'parameters': {'mean': 10.0, 'std': 1.0}}
    var = [d, l]

    num_amostras = 50000
    num_blocos = 5000
    num_limit_functions = 3
    print("\n--- Executando com multiprocessing ---")
    result_parallel = sampling_algorithm_structural_analysis_(
        objective_function=obj,
        number_of_samples=num_amostras,
        method='mcs',
        variables_settings=var,
        number_of_limit_functions=num_limit_functions,
        none_variable=None,
        block_size=num_blocos
    )

    print("\n--- Executando com loop sequencial ---")
    result_sequential = sampling_algorithm_structural_analysis_loop(
        objective_function=obj,
        number_of_samples=num_amostras,
        method='mcs',
        variables_settings=var,
        number_of_limit_functions=num_limit_functions,
        none_variable=None,
        block_size=num_blocos
    )

    # Mostra apenas as 5 primeiras linhas para checar
    # print("\nResultado (5 primeiras linhas):")
    # print(result_parallel.head())