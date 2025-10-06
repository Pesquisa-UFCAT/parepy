import time 
import pandas as pd
import numpy as np
from typing import Callable, Any, Optional
from multiprocessing import Pool
import matplotlib.pyplot as plt

import sys
sys.path.append(r'C:\Users\rezio\OneDrive\Documentos\.git codes\parepy')

from parepy_toolbox import common_library as parepyco
from parepy_toolbox import pare as parepy

def obj(x, *args):
    g_0 = 12.5 * x[0] ** 3 - x[1]
    time.sleep(10**-6) 
    return [g_0]

if __name__ == "__main__":
    d = {'type': 'normal', 'parameters': {'mean': 1.0, 'std': 0.1}}
    l = {'type': 'normal', 'parameters': {'mean': 10.0, 'std': 1.0}}
    var = [d, l]

    m_sobol = 12 
    num_amostras = 2**m_sobol      
    num_limit_functions = 1
    methods = ['sobol', 'mcs', 'lhs']
    results = {}

    for method in methods:
        print(f"Rodando amostragem com método: {method}")
        amostras = m_sobol if method == 'sobol' else num_amostras
        print(f"Número de amostras: {amostras}")
        df, pf, beta = parepy.sampling_algorithm_structural_analysis(
            obj=obj,
            random_var_settings=var,
            method=method,
            n_samples=amostras,
            number_of_limit_functions=num_limit_functions,
            parallel=True,
            verbose=False,
            args=None
        )
        results[method] = (df, pf, beta)

    convergencias = {}

    for method in methods:
        df = results[method][0]
        col_I = [c for c in df.columns if c.startswith("I_")]
        col = col_I[0]
        conv = parepyco.convergence_probability_failure(df, col)
        convergencias[method] = conv
    
        print(f"Tamanho da amostra para {method}: {len(df)}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, method in enumerate(methods):
        div, m, ci_l, ci_u, var = convergencias[method]
        ax = axes[i]
        ax.plot(div, m, label='pf estimado', color='blue')
        ax.fill_between(div, ci_l, ci_u, color='blue', alpha=0.2, label='IC 95%')
        ax.set_title(f"Método: {method.upper()}")
        ax.set_xlabel("Amostras")
        ax.set_yscale("log")  
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        if i == 0:
            ax.set_ylabel("Probabilidade de Falha (escala log)")
        ax.legend()

    plt.savefig("convergencia_probabilidade_falha.png", dpi=300)
    plt.tight_layout()
    # plt.show()
