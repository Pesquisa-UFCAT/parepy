from typing import Callable

# Pasta de referência: 


def structural_problems_1(type_: str, name: str) -> tuple[Callable, list]:
    """
    This function contains several problems about structural reliability.

    :param type_: Choose the type of algorithm you will use in the numerical solution. Supported values: (a) 'sampling' and (b) 'derivative'.
                 (a) 'sampling' - list return for sampling methods,
                 (b) 'derivative' - direct return for deterministic methods.
    :param name: name of problems: 
        (a) 'Chang-p558'           - Exemplo do livro *e-Design Computer-Aided Engineering Design* (2015), pág. 558, example 10.5, [chang_e-design_2015];
        (b) 'NowakCollins-p123'    - Exemplo do livro *Reliability of Structures* (Nowak & Collins, 2000), pág. 123, example 5.9, [nowak_reliability_2000];
        (c) 'NowakCollins-p129'    - Exemplo do livro *Reliability of Structures* (Nowak & Collins, 2000), pág. 127, example 5.11 [nowak_reliability_2000];
        (d) 'Jacinto-p155'         - Exemplo do livro *Segurança Estrutural – Uma introdução com aplicações à segurança de estruturas existentes* (Jacinto, 2023), pág. 155, example 9.1, [jacinto_segurancestrutural_2023];
        (e) 'Grandhi-Wang-p74'     - Exemplo do livro *Structural Reliability Analysis and Optimization: Use of Approximations*  (1999), pág. 74, [grandhi_structural_1999];
        (f) 'Desconhecido-123'     - Problema genérico com parâmetros hipotéticos.
        (g) 'wanderlei_2025'       - Problema de projeto de laje com parede [Autoria: Iniciação Científica de Wanderlei, 2025].
        (h) 'jacinto-p165'         - Exemplo do livro *Segurança Estrutural – Uma introdução com aplicações à segurança de estruturas existentes* (Jacinto, 2023), pág. 165, example 9.4, [jacinto_segurancestrutural_2023].


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

        elif name == 'NowakCollins-p127':
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

        elif name == 'Desconhecido-123':
            def obj_(x):
                e, i, p = x
                return [e * i - 78.12 * p]
            e = {'type': 'normal', 'parameters': {'mean': 2e7, 'std': 0.5e7}}
            i = {'type': 'normal', 'parameters': {'mean': 1e-4, 'std': 0.2e-4}}
            p = {'type': 'gumbel max', 'parameters': {'mean': 4., 'std': 1.}}
            random_var_settings = [e, i, p]

        elif name == 'wanderlei_2025':
            def obj_(x):
                f_pk = x[0]     
                g_laje = x[1]   
                q_laje = x[2]   
                g_par = x[3]    
                l_par = x[4]    
                quinhao = x[5]  

                # Solicitação
                f_g = (g_laje * quinhao) + (g_par * l_par * 2.80)
                f_q = (q_laje * quinhao)
                s = (f_g + f_q) * 1.4 * 4 

                # Resistência
                f_pd = 0.7 * f_pk / 2 * 0.8
                lambd = 2.80 / 0.15
                cr = (1 - (lambd / 40) ** 3)
                n_rd = cr * f_pd * (0.15 * l_par)
                r = n_rd
                g = r - s

                return [g]

            f_pk =    {'type': 'normal',      'parameters': {'mean': 3.0,     'std': 0.3}}       
            g_laje =  {'type': 'normal',      'parameters': {'mean': 3.0,     'std': 0.3}}       
            q_laje =  {'type': 'gumbel max',  'parameters': {'mean': 2.0,     'std': 0.4}}       
            g_par = {'type': 'normal', 'parameters': {'mean': 12.0, 'std': 1.2}}  
            l_par =   {'type': 'normal',      'parameters': {'mean': 5.0,     'std': 0.1}}       
            quinhao = {'type': 'normal',      'parameters': {'mean': 0.24026, 'std': 0.01}}      

            random_var_settings = [f_pk, g_laje, q_laje, g_par, l_par, quinhao]

        elif name == 'jacinto-p165':
            def obj_(x):
                g, q, fy = x
                a_s = 4 * 0.79e-4 
                return [fy * a_s - 6.75 * (g + q)]

            g = {'type': 'normal',     'parameters': {'mean': 12.0,    'std': 0.05 * 12.0}}     
            q = {'type': 'gumbel max', 'parameters': {'mean': 6.0,     'std': 0.10 * 6.0}}      
            fy = {'type': 'normal',    'parameters': {'mean': 560e3,   'std': 0.05 * 560e3}}    

            random_var_settings = [g, q, fy]

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

        elif name == 'NowakCollins-p127':
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

        elif name == 'Desconhecido-123':
            def obj_(x):
                e, i, p = x
                return e * i - 78.12 * p
            e = {'type': 'normal', 'parameters': {'mean': 2e7, 'std': 0.5e7}}
            i = {'type': 'normal', 'parameters': {'mean': 1e-4, 'std': 0.2e-4}}
            p = {'type': 'gumbel max', 'parameters': {'mean': 4., 'std': 1.}}
            random_var_settings = [e, i, p]

        elif name == 'wanderlei_2025':  
            def obj_(x):
                f_pk = x[0]     
                g_laje = x[1]   
                q_laje = x[2]   
                g_par = x[3]    
                l_par = x[4]    
                quinhao = x[5]  

                f_g = (g_laje * quinhao) + (g_par * l_par * 2.80)
                f_q = (q_laje * quinhao)
                s = (f_g + f_q) * 1.4 * 4 
   
                f_pd = 0.7 * f_pk / 2 * 0.8
                lambd = 2.80 / 0.15
                cr = (1 - (lambd / 40) ** 3)
                n_rd = cr * f_pd * (0.15 * l_par)
                r = n_rd
                g = r - s

                return g

            f_pk =    {'type': 'normal',      'parameters': {'mean': 3.0,     'std': 0.3}}       
            g_laje =  {'type': 'normal',      'parameters': {'mean': 3.0,     'std': 0.3}}      
            q_laje =  {'type': 'gumbel max',  'parameters': {'mean': 2.0,     'std': 0.4}}       
            g_par =   {'type': 'normal',      'parameters': {'mean': 12.0,    'std': 1.2}}       
            l_par =   {'type': 'normal',      'parameters': {'mean': 5.0,     'std': 0.1}}       
            quinhao = {'type': 'normal',      'parameters': {'mean': 0.24026, 'std': 0.01}} ## Atenção      
            random_var_settings = [f_pk, g_laje, q_laje, g_par, l_par, quinhao]

        elif name == 'jacinto-p165':
            def obj_(x):
                g, q, fy = x
                a_s = 4 * 0.79e-4 
                return fy * a_s - 6.75 * (g + q)

            g = {'type': 'normal',     'parameters': {'mean': 12.0,    'std': 0.05 * 12.0}}     
            q = {'type': 'gumbel max', 'parameters': {'mean': 6.0,     'std': 0.10 * 6.0}}      
            fy = {'type': 'normal',    'parameters': {'mean': 560e3,   'std': 0.05 * 560e3}}    

            random_var_settings = [g, q, fy]


        else:
            raise ValueError(f"Problema '{name}' não reconhecido para 'derivative'.")

    else:
        raise ValueError("Tipo de algoritmo inválido. Use 'sampling' ou 'derivative'.")

    return obj_, random_var_settings
