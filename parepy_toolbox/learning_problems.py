"""Learning and problems to use in PAREpy toolbox"""
from typing import Callable

def structural_problems(type_: str, name: str) -> tuple[Callable, list]:
    """
    This function contain several problems about structural reliability.

    :param type_: Choose type the algorithm you will use in numerical solution. Supported values: (a) 'sampling' and (b) 'derivative'.
    :param name: Name of problem. Supported values: (a) 'Chang-p558' - aqui a ref zotero, example 10.5 page 558, (b) xxxxxx, (c) xxxxxx

    :return output[0] = The objective function, output[1] = Containing the distribution type and parameters.

    # Examples
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
            pass
    else:
        if name == 'Chang-p558':
            def obj_(x):
                g_0 = 12.5 * x[0] ** 3 - x[1]
                return g_0
            obj = obj_
            d = {'type': 'normal', 'parameters': {'mean': 1., 'std': 0.1}}
            l = {'type': 'normal', 'parameters': {'mean': 10., 'std': 1.}}
            random_var_settings = [d, l]
        elif name == 'NowakCollins-p123':
            pass

    return obj, random_var_settings  