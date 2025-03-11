from parepy_toolbox import sampling_algorithm_structural_analysis

import streamlit as st

import textwrap
import tempfile
import importlib.util
import sys


def generate_function(capacity_expr, demand_expr):
    function_code = f"""
    def nowak_collins_example(x, none_variable):
        # Random variables
        f_y = x[0]
        p_load = x[1]
        w_load = x[2]
        
        capacity = {capacity_expr}
        demand = {demand_expr}

        # State limit function
        constraint = capacity - demand

        return [capacity], [demand], [constraint]
    """
    
    with open("obj_functions.py", "w") as f:
        f.write(textwrap.dedent(function_code))
    
    return function_code

st.title("Parepy")

# Entrada do usuário
capacity_input = st.text_area("Capacity:", "80 * x[0]")
demand_input = st.text_area("Demand:", "54 * x[1] + 5832 * x[2]")

if st.button("Gerar Função"):
    function_str = generate_function(capacity_input, demand_input)
    st.code(textwrap.dedent(function_str), language="python")


if st.button("Executar Algoritmo"):

    from obj_functions import nowak_collins_example
    
    # Statement random variables
    f = {
            'type': 'normal', 
            'parameters': {'mean': 40.3, 'sigma': 4.64}, 
            'stochastic variable': False, 
        }

    p = {
            'type': 'gumbel max',
            'parameters': {'mean': 10.2, 'sigma': 1.12}, 
            'stochastic variable': False, 
        }

    w = {
            'type': 'lognormal',
            'parameters': {'mean': 0.25, 'sigma': 0.025}, 
            'stochastic variable': False, 
        }
    var = [f, p, w]

    # PAREpy setup
    setup = {
                'number of samples': 1000, 
                'numerical model': {'model sampling': 'mcs'}, 
                'variables settings': var, 
                'number of state limit functions or constraints': 1, 
                'none variable': None,
                'objective function': nowak_collins_example,
                'name simulation': None,
            }

    # Call algorithm
    results, pf, beta = sampling_algorithm_structural_analysis(setup)
    st.write(results)
    st.write(pf)
    st.write(beta)

    with open("obj_functions.py", "w") as f:
        f.write(textwrap.dedent(""))