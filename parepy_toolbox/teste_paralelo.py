import time
import pandas as pd
import numpy as np
from typing import Callable, Any, Optional
from multiprocessing import Pool

import sys
sys.path.append(r'C:\Users\rezio\OneDrive\Documentos\.git codes\parepy')

from parepy_toolbox import common_library as parepyco
from parepy_toolbox import pare as parepy


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
    num_limit_functions = 1

    result_parallel = parepy.sampling_algorithm_structural_analysis_(
        objective_function=obj,
        number_of_samples=num_amostras,
        method='mcs',
        variables_settings=var,
        number_of_limit_functions=num_limit_functions,
        none_variable=None,
        block_size=num_blocos
    )

    print("Result of the parallel sampling algorithm:")
    print(result_parallel)

    print("-----"*10)
    print("Summarizing the results...")
    pf_df, beta_df = parepyco.summarize_failure_probabilities(result_parallel)

    print("Failure Probability DataFrame:")
    print(pf_df)

    print("-----"*10)

    print("Beta DataFrame:")
    print(beta_df)