"""Learning and problems to use in PAREpy toolbox"""
from typing import Callable


def structural_problems(type_: str, name: str) -> tuple[Callable, list]:
    """
    Provides benchmark structural reliability problems, including objective functions and probabilistic definitions for variables.

    :param type_: Method type used for reliability analysis. Supported values:
        (a) ``"sampling"`` – the objective function returns a list: ``[g(x)]``;
        (b) ``"derivative"`` – the objective function returns a scalar: ``g(x)``.
        Use ``"sampling"`` for Monte Carlo, LHS, or Sobol methods, and ``"derivative"`` for FORM/SORM or gradient-based algorithms.

    :param name: Name of the structural problem. Supported examples:
        (a) ``"Chang-p558"`` – Example 10.5, page 558 from *e-Design: Computer-Aided Engineering Design* (Chang, 2015) [chang_e-design_2015];
        (b) ``"NowakCollins-p123"`` – Example 5.9, page 123 from *Reliability of Structures* (Nowak & Collins, 2000) [nowak_reliability_2000];
        (c) ``"NowakCollins-p127"`` – Example 5.11, page 127 from *Reliability of Structures* (Nowak & Collins, 2000) [nowak_reliability_2000];
        (d) ``"Jacinto-p155"`` – Example 9.1, page 155 from *Segurança Estrutural* (Jacinto, 2023) [jacinto_segurancestrutural_2023];
        (e) ``"Grandhi-Wang-p74"`` – Problem from page 74 in *Structural Reliability Analysis and Optimization* (Grandhi & Wang, 1999) [grandhi_structural_1999];
        (f) ``"Desconhecido-123"`` – Hypothetical example with arbitrary parameters;
        (g) ``"wanderlei_2025"`` – Structural slab-wall design [Undergraduate Research – Wanderlei, 2025];
        (h) ``"jacinto-p165"`` – Example 9.4, page 165 from *Segurança Estrutural* (Jacinto, 2023) [jacinto_segurancestrutural_2023].

    :return: Tuple with:
        - Objective function ``obj(x)``: Callable. Returns a list or scalar depending on ``type_``;
        - Random variable settings: List of dictionaries defining each distribution. Format:  
          ``{"type": str, "parameters": {"mean": float, "std": float}}``.
          Supported distributions (see more in Table 1):
            * ``"normal"``: keys ``"mean"``, ``"std"``
            * ``"lognormal"``: keys ``"mean"``, ``"std"``
            * ``"gumbel max"``: keys ``"mean"``, ``"std"``

    :raises ValueError: If an unsupported ``type_`` or ``name`` is provided.
    """

    def make_return(obj_scalar):
        if type_ == 'sampling':
            return lambda x: [obj_scalar(x)], dist_list
        elif type_ == 'derivative':
            return obj_scalar, dist_list
        else:
            raise ValueError("type_ must be 'sampling' or 'derivative'.")

    if name == 'Chang-p558':
        def obj(x): return 12.5 * x[0] ** 3 - x[1]
        d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
        l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
        dist_list = [d, l]

    elif name == 'NowakCollins-p123':
        def obj(x): return x[0] - x[1]
        r = {'type': 'lognormal', 'parameters': {'mean': 200, 'std': 20}}
        q = {'type': 'gumbel max', 'parameters': {'mean': 100, 'std': 12}}
        dist_list = [r, q]

    elif name == 'NowakCollins-p127':
        def obj(x): return x[0] * x[1] - x[2]
        z = {'type': 'normal', 'parameters': {'mean': 100, 'std': 0.04 * 100}}
        fy = {'type': 'lognormal', 'parameters': {'mean': 40, 'std': 0.10 * 40}}
        m = {'type': 'gumbel max', 'parameters': {'mean': 2000, 'std': 0.10 * 2000}}
        dist_list = [z, fy, m]

    elif name == 'Jacinto-p155':
        def obj(x):
            g, q, fy = x
            a_s = 4 * 0.79e-4
            return fy * a_s - 6.75 * (g + q)
        g = {'type': 'normal', 'parameters': {'mean': 12., 'std': 0.05 * 12.}}
        q = {'type': 'gumbel max', 'parameters': {'mean': 6., 'std': 0.10 * 6.}}
        fy = {'type': 'normal', 'parameters': {'mean': 560E3, 'std': 0.05 * 560E3}}
        dist_list = [g, q, fy]

    elif name == 'Grandhi-Wang-p74':
        def obj(x): return x[0] ** 3 + x[1] ** 3 - 18
        x1 = {'type': 'normal', 'parameters': {'mean': 10., 'std': 5.}}
        x2 = {'type': 'normal', 'parameters': {'mean': 10., 'std': 5.}}
        dist_list = [x1, x2]

    elif name == 'Desconhecido-123':
        def obj(x): return x[0] * x[1] - 78.12 * x[2]
        e = {'type': 'normal', 'parameters': {'mean': 2e7, 'std': 0.5e7}}
        i = {'type': 'normal', 'parameters': {'mean': 1e-4, 'std': 0.2e-4}}
        p = {'type': 'gumbel max', 'parameters': {'mean': 4., 'std': 1.}}
        dist_list = [e, i, p]

    elif name == 'wanderlei_2025':
        def obj(x):
            f_pk, g_laje, q_laje, g_par, l_par, quinhao = x
            f_g = (g_laje * quinhao) + (g_par * l_par * 2.80)
            f_q = (q_laje * quinhao)
            s = (f_g + f_q) * 1.4 * 4
            f_pd = 0.7 * f_pk / 2 * 0.8
            lambd = 2.80 / 0.15
            cr = (1 - (lambd / 40) ** 3)
            n_rd = cr * f_pd * (0.15 * l_par)
            return n_rd - s
        f_pk = {'type': 'normal', 'parameters': {'mean': 3.0, 'std': 0.3}}
        g_laje = {'type': 'normal', 'parameters': {'mean': 3.0, 'std': 0.3}}
        q_laje = {'type': 'gumbel max', 'parameters': {'mean': 2.0, 'std': 0.4}}
        g_par = {'type': 'normal', 'parameters': {'mean': 12.0, 'std': 1.2}}
        l_par = {'type': 'normal', 'parameters': {'mean': 5.0, 'std': 0.1}}
        quinhao = {'type': 'normal', 'parameters': {'mean': 0.24026, 'std': 0.01}}
        dist_list = [f_pk, g_laje, q_laje, g_par, l_par, quinhao]

    elif name == 'jacinto-p165':
        def obj(x):
            g, q, fy = x
            a_s = 4 * 0.79e-4
            return fy * a_s - 6.75 * (g + q)
        g = {'type': 'normal',     'parameters': {'mean': 12.0,    'std': 0.05 * 12.0}}     
        q = {'type': 'gumbel max', 'parameters': {'mean': 6.0,     'std': 0.10 * 6.0}}      
        fy = {'type': 'normal',    'parameters': {'mean': 560e3,   'std': 0.05 * 560e3}} 
        dist_list = [g, q, fy]

    else:
        raise ValueError(f"Problema '{name}' não reconhecido.")

    return make_return(obj)

