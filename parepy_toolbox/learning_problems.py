"""Learning and problems to use in PAREpy toolbox"""
from typing import Callable

def structural_problems_1(type_: str, name: str) -> tuple[Callable, list]:
    """
    Define problemas estruturais para análise numérica.

    :param type_: Tipo de algoritmo a ser usado na solução numérica. Valores suportados: 
                  (a) 'sampling' - retorno em lista para métodos de amostragem,
                  (b) 'derivative' - retorno direto para métodos determinísticos.
    :param name: Nome do problema. Valores suportados: 
                 (a) 'Chang-p558' - exemplo do livro de Chang, pág. 558,
                 (b) 'NowakCollins-p123' - exemplo do livro Reliability of Structures, pág. 123,
                 (c) 'NowakCollins-p129' - exemplo do livro Reliability of Structures, pág. 129.

    :return: output[0] = função objetivo, output[1] = lista com tipo de distribuição e parâmetros das variáveis aleatórias.
    """

    if type_ == 'sampling':
        if name == 'Chang-p558':
            def obj_(x):
                g_0 = 12.5 * x[0] ** 3 - x[1]
                return [g_0]
            obj = obj_
            d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
            l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
            random_var_settings = [d, l]

        elif name == 'NowakCollins-p123':
            def obj_(x):
                r, q = x
                return [r - q]
            obj = obj_
            r = {'type': 'lognormal', 'parameters': {'mean': 200, 'std': 20}}
            q = {'type': 'gumbel max', 'parameters': {'mean': 100, 'std': 12}}
            random_var_settings = [r, q]

        elif name == 'NowakCollins-p129':
            def obj_(x):
                z, fy, m = x
                return [z * fy - m]
            obj = obj_
            z = {'type': 'normal', 'parameters': {'mean': 100, 'std': 0.04 * 100}}
            fy = {'type': 'lognormal', 'parameters': {'mean': 40, 'std': 0.10 * 40}}
            m = {'type': 'gumbel max', 'parameters': {'mean': 2000, 'std': 0.10 * 2000}}
            random_var_settings = [z, fy, m]

    elif type_ == 'derivative':
        if name == 'Chang-p558':
            def obj_(x):
                g_0 = 12.5 * x[0] ** 3 - x[1]
                return g_0
            obj = obj_
            d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
            l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
            random_var_settings = [d, l]

        elif name == 'NowakCollins-p123':
            def obj_(x):
                r, q = x
                return r - q
            obj = obj_
            r = {'type': 'lognormal', 'parameters': {'mean': 200, 'std': 20}}
            q = {'type': 'gumbel max', 'parameters': {'mean': 100, 'std': 12}}
            random_var_settings = [r, q]

        elif name == 'NowakCollins-p129':
            def obj_(x):
                z, fy, m = x
                return z * fy - m
            obj = obj_
            z = {'type': 'normal', 'parameters': {'mean': 100, 'std': 0.04 * 100}}
            fy = {'type': 'lognormal', 'parameters': {'mean': 40, 'std': 0.10 * 40}}
            m = {'type': 'gumbel max', 'parameters': {'mean': 2000, 'std': 0.10 * 2000}}
            random_var_settings = [z, fy, m]

    else:
        raise ValueError("Tipo de algoritmo inválido. Use 'sampling' ou 'derivative'.")

    return obj, random_var_settings

