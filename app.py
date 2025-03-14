from parepy_toolbox import sampling_algorithm_structural_analysis, convergence_probability_failure, sobol_algorithm
from io import BytesIO

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import base64
import textwrap


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

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = image_to_base64("assets/images/logo.png")
img_html = f'<img src="data:image/png;base64,{img_base64}" width="150"/>'


st.markdown(rf""" 
<table>
    <tr>
        <td style="width:70%;"><p align="justify">
            The PAREpy (<b>Probabilistic Approach to Reliability Engineering</b>) framework is a software developed by the research group headed by 
            <a href="http://lattes.cnpq.br/2268506213083114" target="_blank" rel="noopener noreferrer">Professor Wanderlei M. Pereira Junior</a> 
            in Engineering College at Universidade Federal de Catalão. It is a framework for applying probabilistic concepts to analyze a system containing random variables. 
            The platform is built in Python and can be used in any environment that supports this programming language.
        </p></td>
        <td style="width:50%;">{img_html}</td>  
    </tr>
</table>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .suggestions-box1 {
        border: 2px solid #00008B;
        background-color: #ADD8E6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    
    <div class="suggestions-box1">
        <h4>Suggestions</h4>
        <p>If you have any suggestions or error reports regarding the algorithm's functioning, 
        please email us at <a href="mailto:wanderlei_junior@ufcat.edu.br">wanderlei_junior@ufcat.edu.br</a>. 
        We will be happy to improve the framework.</p>
    </div>
    """,
    unsafe_allow_html=True)

st.write("")

st.markdown(
    """    
    <style>
    .suggestions-box {
        border: 2px solid #FFA500;
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .suggestions-box p {
        margin: 5px 0;
    }
    .suggestions-box a {
        text-decoration: none;
        color: #007BFF;
        font-weight: bold;
    }
    </style>

    <div class="suggestions-box">
        <h4>Team</h4>
        <p><a href="http://lattes.cnpq.br/2268506213083114" target="_blank">Prof. PhD Wanderlei Malaquias Pereira Junior</a></p>
        <p><a href="http://lattes.cnpq.br/8801080897723883" target="_blank">Prof. PhD Daniel de Lima Araújo</a></p>
        <p><a href="http://lattes.cnpq.br/4319075758352865" target="_blank">Prof. PhD André Teófilo Beck</a></p>
        <p><a href="http://lattes.cnpq.br/7623383075429186" target="_blank">Prof. PhD André Luis Christoforo</a></p>
        <p><a href="http://lattes.cnpq.br/6429652195589650" target="_blank">Msc Murilo Carneiro Rodrigues</a></p>
        <p><a href="http://lattes.cnpq.br/3180484792983028" target="_blank">Prof. PhD Iuri Fazolin Fraga</a></p>
        <p><a href="http://lattes.cnpq.br/8465474056220474" target="_blank">Msc Matheus Henrique Morato Moraes</a></p>
        <p><a href="http://orcid.org/0000-0002-5798-8553" target="_blank">Prof. PhD Mauro Alexandre Paula de Sousa</a></p>
        <p><a href="http://orcid.org/0000-0002-8533-1575" target="_blank">Prof. PhD Antover Panazzolo Sarmento</a></p>
        <p><a href="http://orcid.org/0009-0008-4084-2137" target="_blank">Dis. Luiz Henrique Ferreira Rezio</a></p>
    </div>
    """,
    unsafe_allow_html=True
)


st.write("")
st.markdown(
    """    
    <div class="suggestions-box1">
        <p>Version</p>
        <p><a href="https://pypi.org/project/parepy-toolbox/#history" target="_blank" style="text-decoration: none; color: blue;">1.0.1</a></p>
    </div>
    """,
    unsafe_allow_html=True)

st.write("")
st.subheader("How to use")
st.markdown("""
To use PAREpy, you need to define a state limit function. In this case, fill the boxes with your capacity and demand functions. 
This framework uses Python, so you need to start declaring your variables as x[0], x[1], and so on, depending on the number of variables in your problem. See an example:

Consider the simply supported beam show in example 5.1 Nowak and Collins <a href="#ref1">[1]</a>. The beam is subjected to a concentrated live load $p$ and a uniformly distributed dead load $w$. 
Assume $P$ (concentrated live load), $W$ (uniformly distributed dead load) and the yield stress, $F_y$, are random quantities; the length $l$ and the plastic setion modulus $z$ are assumed to be precisely know (deterministic). 
The distribution parameters for $P$, $W$ and $F_y$ are given bellow:
""", unsafe_allow_html=True)

st.table({
    'Variable': ['Yield stress $F_y$', 'Live load $(P)$', 'Dead load $(W)$'],
    'Distribution': ['Normal', 'Gumbel max.', 'Log-normal'],
    'Mean': [40.3, 10.2, 0.25],
    'Coefficient of Variation (COV)': [0.115, 0.110, 0.100]
        })

st.write("")

st.markdown(r"""
    The limit state function for beam bending can be expressed as:
    $$
    \begin{align*}
    R &= 80 \cdot F_y \tag{1} \\
    S &= 54 \cdot P + 5832 \cdot W \tag{2} \\
    G &= R - S \begin{cases}
                    \leq 0, \text{failure} \\
                    > 0, \text{safe}
                \end{cases} \tag{3}
    \end{align*}
    $$
""", unsafe_allow_html=True) 

st.write("")
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
        with st.expander(f"Variable [{i+1}]"):
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





