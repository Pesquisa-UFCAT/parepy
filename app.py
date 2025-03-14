from parepy_toolbox import sampling_algorithm_structural_analysis, convergence_probability_failure, sobol_algorithm
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import textwrap
import json


def generate_function(capacity_expr, demand_expr):
    function_code = f"""
    def obj_function(x, none_variable):
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
st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed laoreet nisl quis quam mattis molestie. Aliquam efficitur, risus et fringilla pellentesque, est sapien finibus sapien, vitae scelerisque nisl nunc vel justo. Nullam ut ornare diam. Ut convallis ex velit, eu condimentum ligula porttitor nec. Sed id magna ut elit fermentum convallis. Curabitur tincidunt tellus tortor, et ultrices massa faucibus sit amet. Suspendisse aliquam, massa et posuere dictum, ipsum purus egestas leo, a placerat metus felis non magna. Fusce ac sem aliquam, egestas velit vel, laoreet mi. Nulla lacinia tortor id interdum faucibus. Ut laoreet felis at purus congue, eget viverra metus blandit. Donec placerat finibus laoreet. Quisque luctus sodales felis, in sollicitudin sem tristique eu. Aenean aliquet nunc sem, vel scelerisque nisi ornare eu. Nulla orci turpis, molestie non ex at, fringilla elementum enim. Cras dictum, dui nec tincidunt scelerisque, neque augue ullamcorper leo, sit amet vulputate ex diam vitae nisl.")

st.subheader("Objective Function parameters")
capacity_input = st.text_area("Capacity:", "80 * x[0]")
demand_input = st.text_area("Demand:", "54 * x[1] + 5832 * x[2]")

# Configuração do setup
st.write("")
st.subheader("Setup Configuration")
num_samples = st.number_input("Number of samples", min_value=1, step=1, value=10000)
model_sampling = st.selectbox("Model Sampling", ["mcs", "lhs"], index=0) 

st.write("")
st.subheader("Model Configuration")


if "var" not in st.session_state:
    st.session_state.var = []

num_vars = st.number_input("Number of Variables", min_value=1, step=1, value=max(1, len(st.session_state.var)))

while len(st.session_state.var) < num_vars:
    st.session_state.var.append({'type': 'normal', 'parameters': {}, 'stochastic variable': False})
while len(st.session_state.var) > num_vars:
    st.session_state.var.pop()

distribution_types = ["uniform", "normal", "lognormal", "gumbel max", "gumbel min", "triangular"]

with st.container():
    for i in range(num_vars):
        with st.expander(f"Variable X_{i+1}"):
            var_type = st.selectbox(f"Type", distribution_types, key=f"type_{i}", index=distribution_types.index(st.session_state.var[i]['type']))
            
            parameters = {}
            if var_type == "triangular":
                min_val = st.number_input(f"Min", key=f"min_{i}", value=None, placeholder="Enter value")
                mode = st.number_input(f"Mode", key=f"mode_{i}", value=None, placeholder="Enter value")
                max_val = st.number_input(f"Max", key=f"max_{i}", value=None, placeholder="Enter value")
                parameters = {'min': min_val, 'mode': mode, 'max': max_val}
            else:
                mean = st.number_input(f"Mean", key=f"mean_{i}", value=None, placeholder="Enter value")
                sigma = st.number_input(f"Sigma", key=f"sigma_{i}", value=None, placeholder="Enter value")
                parameters = {'mean': mean, 'sigma': sigma}

            st.session_state.var[i] = {
                'type': var_type,
                'parameters': parameters,
            }

if st.button("Run Simulation"):
    function_str = generate_function(capacity_input, demand_input)

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

    results, pf, beta = sampling_algorithm_structural_analysis(setup)
    print(setup)
    print(results)
    # Gráficos
    st.session_state.text_convergence = "Convergence Rate:"
    div, m, ci_l, ci_u, var = convergence_probability_failure(results, 'I_0')
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(div, m, label="Failure Probability Rate", color='b', linestyle='-')
    ax1.fill_between(div, ci_l, ci_u, color='b', alpha=0.2, label="95% Confidence Interval")
    ax1.set_xlabel("Sample Size (div)")
    ax1.set_ylabel("Failure Probability Rate")
    ax1.set_title("Convergence of Failure Probability")
    ax1.legend()
    ax1.grid(True)
    st.session_state.fig1 = fig1

    st.session_state.text_sobol = "Sobol Sensitivity Analysis:"
    data_sobol = sobol_algorithm(setup)
    variables = ['x_0', 'x_1', 'x_2']
    s_i = [data_sobol.iloc[var]['s_i'] for var in range(len(variables))]
    s_t = [data_sobol.iloc[var]['s_t'] for var in range(len(variables))]

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    x = range(len(variables))
    width = 0.35
    ax2.bar(x, s_i, width, label='First-order (s_i)', color='blue', alpha=0.7)
    ax2.bar([p + width for p in x], s_t, width, label='Total-order (s_t)', color='orange', alpha=0.7)
    ax2.set_xlabel("Variables")
    ax2.set_ylabel("Sobol Index")
    ax2.set_xticks([p + width / 2 for p in x])
    ax2.set_xticklabels(variables)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    st.session_state.fig2 = fig2

    st.session_state.results = results
    st.session_state.pf = pf
    st.session_state.beta = beta
    st.session_state.data_sobol = data_sobol

# Re-rendering everything, checking session state to ensure content persists
if "text_convergence" in st.session_state:
    st.subheader(st.session_state.text_convergence)

if "fig1" in st.session_state:
    st.pyplot(st.session_state.fig1)

if "text_sobol" in st.session_state:
    st.subheader(st.session_state.text_sobol)

if "fig2" in st.session_state:
    st.pyplot(st.session_state.fig2)

# Download
if "results" in st.session_state:
    results = st.session_state.results  # Access results from session state
    final_results = BytesIO()
    with pd.ExcelWriter(final_results, engine="xlsxwriter") as writer:
        results.to_excel(writer, index=False, sheet_name="Results")
    final_results.seek(0)
    st.download_button("Download Results", final_results, file_name="results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")





