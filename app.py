from parepy_toolbox import sampling_algorithm_structural_analysis, convergence_probability_failure
from io import BytesIO

import streamlit as st
import pandas as pd

import textwrap
import json


def generate_function(capacity_expr, demand_expr):
    function_code = f"""
    def obj_function(x, none_variable):
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

st.title("PAREpy")

# Entrada do usuário
capacity_input = st.text_area("Capacity:", "80 * x[0]")
demand_input = st.text_area("Demand:", "54 * x[1] + 5832 * x[2]")


st.subheader("Configuração do Modelo")

# Lista para armazenar as variáveis
if "var" not in st.session_state:
    st.session_state.var = []

# Definir número de variáveis
num_vars = st.number_input("Número de variáveis aleatórias", min_value=1, step=1, value=max(1, len(st.session_state.var)))

# Ajustar o número de variáveis armazenadas
while len(st.session_state.var) < num_vars:
    st.session_state.var.append({
        'type': 'normal',
        'parameters': {'mean': 40.3, 'sigma': 4.64},
        'stochastic variable': False
    })
while len(st.session_state.var) > num_vars:
    st.session_state.var.pop()

# Opções de distribuição
distribution_types = ["uniform", "normal", "lognormal", "gumbel max", "gumbel min", "triangular"]

# Criar inputs para cada variável
with st.container():
    for i in range(num_vars):
        with st.expander(f"Variável {i+1}"):
            var_type = st.selectbox(f"Tipo da variável {i+1}", distribution_types, key=f"type_{i}", index=distribution_types.index(st.session_state.var[i]['type']))
            
            if var_type == "triangular":
                min_val = st.number_input(f"Mínimo da variável {i+1}", key=f"min_{i}", value=st.session_state.var[i]['parameters'].get('min', 0.0))
                mode = st.number_input(f"Moda da variável {i+1}", key=f"mode_{i}", value=st.session_state.var[i]['parameters'].get('mode', 0.0))
                max_val = st.number_input(f"Máximo da variável {i+1}", key=f"max_{i}", value=st.session_state.var[i]['parameters'].get('max', 0.0))
                parameters = {'min': min_val, 'mode': mode, 'max': max_val}
            else:
                mean = st.number_input(f"Média da variável {i+1}", key=f"mean_{i}", value=st.session_state.var[i]['parameters'].get('mean', 0.0))
                sigma = st.number_input(f"Sigma da variável {i+1}", key=f"sigma_{i}", value=st.session_state.var[i]['parameters'].get('sigma', 1.0))
                parameters = {'mean': mean, 'sigma': sigma}

            
            # Atualizar valores
            st.session_state.var[i] = {
                'type': var_type,
                'parameters': parameters,
            }

# Configuração do setup
st.subheader("Configuração do PAREpy")
num_samples = st.number_input("Número de amostras", min_value=1, step=1, value=1000)
model_sampling = st.selectbox("Método de amostragem", ["mcs"], index=0)


if st.button("Executar Algoritmo"):
    function_str = generate_function(capacity_input, demand_input)
    st.code(textwrap.dedent(function_str), language="python")

    from obj_functions import obj_function
    
    setup = {
        'number of samples': num_samples,
        'numerical model': {'model sampling': model_sampling},
        'variables settings': st.session_state.var,
        'number of state limit functions or constraints': 1,
        'none variable': None,
        'objective function': obj_function,
        'name simulation': None,
    }

    # st.json(setup, expanded=False)
    
    # Call algorithm
    results, pf, beta = sampling_algorithm_structural_analysis(setup)

    st.subheader("Resultados:")

    st.write(results)
    st.write(pf)
    st.write(beta)

    st.subheader('Convergence Rate:')
    x, m, l, u = convergence_probability_failure(results, 'I_0')

    with open("obj_functions.py", "w") as f:
        f.write(textwrap.dedent(""))

    final_results = BytesIO()
    with pd.ExcelWriter(final_results, engine="xlsxwriter") as writer:
        results.to_excel(writer, index=False, sheet_name="Pareto Front")
    final_results.seek(0)
    st.download_button("Download Resultados", final_results, file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")





