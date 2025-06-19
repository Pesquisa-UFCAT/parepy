from typing import Callable

def structural_problems_1(type_: str, name: str) -> tuple[Callable, list]:
    """
    Define problemas estruturais para análise numérica.

    :param type_: Tipo de algoritmo a ser usado na solução numérica. Valores suportados: 
                  (a) 'sampling' - retorno em lista para métodos de amostragem,
                  (b) 'derivative' - retorno direto para métodos determinísticos.
    :param name: Nome do problema. Valores suportados: 
                 (a) 'Chang-p558',
                 (b) 'NowakCollins-p123',
                 (c) 'NowakCollins-p129',
                 (d) 'Jacinto-p155'.
    :return: output[0] = função objetivo, output[1] = lista com tipo de distribuição e parâmetros das variáveis aleatórias.
    """
    if type_ == 'sampling':
        if name == 'Chang-p558':
            def obj_(x):
                return [12.5 * x[0] ** 3 - x[1]]
            d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
            l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
            random_var_settings = [d, l]

        elif name == 'NowakCollins-p123':
            def obj_(x):
                r, q = x
                return [r - q]
            r = {'type': 'lognormal', 'parameters': {'mean': 200, 'std': 20}}
            q = {'type': 'gumbel max', 'parameters': {'mean': 100, 'std': 12}}
            random_var_settings = [r, q]

        elif name == 'NowakCollins-p129':
            def obj_(x):
                z, fy, m = x
                return [z * fy - m]
            z = {'type': 'normal', 'parameters': {'mean': 100, 'std': 0.04 * 100}}
            fy = {'type': 'lognormal', 'parameters': {'mean': 40, 'std': 0.10 * 40}}
            m = {'type': 'gumbel max', 'parameters': {'mean': 2000, 'std': 0.10 * 2000}}
            random_var_settings = [z, fy, m]

        elif name == 'Jacinto-p155':
            def obj_(x):
                g, q, fy = x
                a_s = 4 * 0.79e-4
                return [fy * a_s - 6.75 * (g + q)]
            g = {'type': 'normal', 'parameters': {'mean': 12., 'std': 0.05 * 12.}}
            q = {'type': 'gumbel max', 'parameters': {'mean': 6., 'std': 0.10 * 6.}}
            fy = {'type': 'normal', 'parameters': {'mean': 560E3, 'std': 0.05 * 560E3}}
            random_var_settings = [g, q, fy]

        elif name == 'Grandhi-Wang-p74':
            def obj_(x):
                x1, x2 = x
                return [x1 ** 3 + x2 ** 3 - 18]
            x1 = {'type': 'normal', 'parameters': {'mean': 10., 'std': 5.}}
            x2 = {'type': 'normal', 'parameters': {'mean': 10., 'std': 5.}}
            random_var_settings = [x1, x2]
        else:
            raise ValueError(f"Problema '{name}' não reconhecido para 'sampling'.")

    elif type_ == 'derivative':
        if name == 'Chang-p558':
            def obj_(x):
                return 12.5 * x[0] ** 3 - x[1]
            d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
            l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
            random_var_settings = [d, l]

        elif name == 'NowakCollins-p123':
            def obj_(x):
                r, q = x
                return r - q
            r = {'type': 'lognormal', 'parameters': {'mean': 200, 'std': 20}}
            q = {'type': 'gumbel max', 'parameters': {'mean': 100, 'std': 12}}
            random_var_settings = [r, q]

        elif name == 'NowakCollins-p129':
            def obj_(x):
                z, fy, m = x
                return z * fy - m
            z = {'type': 'normal', 'parameters': {'mean': 100, 'std': 0.04 * 100}}
            fy = {'type': 'lognormal', 'parameters': {'mean': 40, 'std': 0.10 * 40}}
            m = {'type': 'gumbel max', 'parameters': {'mean': 2000, 'std': 0.10 * 2000}}
            random_var_settings = [z, fy, m]

        elif name == 'Jacinto-p155':
            def obj_(x):
                g, q, fy = x
                a_s = 4 * 0.79e-4
                return fy * a_s - 6.75 * (g + q)
            g = {'type': 'normal', 'parameters': {'mean': 12., 'std': 0.05 * 12.}}
            q = {'type': 'gumbel max', 'parameters': {'mean': 6., 'std': 0.10 * 6.}}
            fy = {'type': 'normal', 'parameters': {'mean': 560E3, 'std': 0.05 * 560E3}}
            random_var_settings = [g, q, fy]

        elif name == 'Grandhi-Wang-p74':
            def obj_(x):
                x1, x2 = x
                return x1 ** 3 + x2 ** 3 - 18
            x1 = {'type': 'normal', 'parameters': {'mean': 10., 'std': 5.}}
            x2 = {'type': 'normal', 'parameters': {'mean': 10., 'std': 5.}}
            random_var_settings = [x1, x2]

        else:
            raise ValueError(f"Problema '{name}' não reconhecido para 'derivative'.")

    else:
        raise ValueError("Tipo de algoritmo inválido. Use 'sampling' ou 'derivative'.")

    return obj_, random_var_settings
