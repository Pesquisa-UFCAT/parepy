---
layout: home
parent: algorithms
grand_parent: Framework
nav_order: 4
has_children: false
has_toc: false
title: sobol_algorithm
---

<!--Don't delete this script-->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h3>sobol_algorithm</h3>

<p align="justify">
    Calculates the Sobol indices for structural reliability problems using Monte Carlo sampling. This function computes the first-order and total-order Sobol sensitivity indices for a given numerical model and variable settings.
</p>

```python
s_i, s_t = sobol_algorithm(setup)
```

Input variables
{: .label .label-yellow }

<table style="width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead>
    <tr>
        <td><code>setup</code></td>
        <td>
            A dictionary containing the settings for the numerical model and analysis.
            <ul>
                <li><code>'number of samples'</code>: An integer defining the number of samples.</li>
                <li><code>'objective function'</code>: A Python function defining the state limit function.</li>
                <li><code>'numerical model'</code>: A dictionary containing the model type (<code>'model'</code>) and additional settings.</li>
                <li><code>'variables settings'</code>: A list of dictionaries defining variable properties (e.g., <code>'mean'</code>, <code>'sigma'</code>).</li>
                <li><code>'number of state limit functions or constraints'</code>: An integer specifying the number of state limit functions or constraints.</li>
                <li><code>'none variable'</code>: Additional user-defined input, used in the objective function.</li>
            </ul>
        </td>
        <td>Dictionary</td>
    </tr>
</table>

Output variables
{: .label .label-yellow }

<table style="width:100%">
   <thead>
     <tr>
       <th>Name</th>
       <th>Description</th>
       <th>Type</th>
     </tr>
   </thead>
   <tr>
       <td><code>data_sobol</code></td>
       <td>
           A dictionary containing the first-order and total-order Sobol sensitivity indices for each input variable. 
       </td>
       <td>Dict</td>
   </tr>
   <tr>
   </tr>
</table>

EXAMPLE
{: .label .label-blue }

This example demonstrates how to use the `sobol_algorithm` function to calculate the Sobol indices for a structural reliability problem.

of_FILE.PY
{: .label .label-red }

<p align="justify">
The <strong>Ishigami function</strong> is commonly used as a test function for comparing global sensitivity analysis methods due to its nonlinear properties and the presence of variable interactions. This function is particularly valuable for benchmarking different sensitivity analysis methods, making it a classic example in this field. 
<br><br>
The function takes as input a vector \( x = [x_0, x_1, x_2] \), which represents three independent variables. Its analytical expression is defined as:
</p>

$$
f(x) = \sin(x_0) + a \cdot \sin^2(x_1) + b \cdot x_2^4 \cdot \sin(x_0)
$$

<div style="text-align: justify;">
<p>where:</p>
<ul>
    <li>\( x = \{x_0, x_1, x_2\} \in [-\pi, \pi]^3 \) are the input variables, limited to the domain \([-\pi, \pi]\);</li>
    <li>\( a \) and \( b \) are adjustable parameters that control the relative impact of each term in the function.</li>
</ul>
</div>


<p align="justify">
This function is widely used to evaluate the influence of each input variable on the final output, as well as the interactions between them. It is particularly effective in global sensitivity analysis methods, such as the calculation of Sobol indices, providing a robust basis for investigating the individual and combined contributions of variables.</p>

```python
def ishigami(x, none_variable):
    """Objective function for the Nowak example (tutorial).
    """
    a = 7
    b = 0.10
    # Random variables
    x_0 = x[0]
    x_1 = x[1]
    x_2 = x[2]
    result = np.sin(x_0) + a * np.sin(x_1) ** 2 + b * (x_2 ** 4) * np.sin(x_0)

    return [None], [None], [result]
```

YOUR_PROBLEM.IPYNB
{: .label .label-red }

```python
from parepy_toolbox import sobol_algorithm

# Dataset
f = {'type': 'uniform', 'parameters': {'min': -3.14, 'max': 3.14}, 'stochastic variable': False}
p = {'type': 'uniform', 'parameters': {'min': -3.14, 'max': 3.14}, 'stochastic variable': False}
w = {'type': 'uniform', 'parameters': {'min': -3.14, 'max': 3.14}, 'stochastic variable': False}
var = [f, p, w]

# PAREpy setup
setup = {
             'number of samples': 50000, 
             'number of dimensions': len(var), 
             'numerical model': {'model sampling': 'lhs'}, 
             'variables settings': var, 
             'number of state limit functions or constraints': 1, 
             'none variable': None,
             'objective function': ishigami,
             'name simulation': None,
        }

# Call algorithm
data_sobol = sobol_algorithm(setup)
```

<h3>Post-processing Sobol Indices</h3>

<p align="justify">
    After executing the <code>sobol_algorithm</code>, it is essential to analyze the results, which include the first-order (\( s_i \)) and total-order (\( s_t \)) Sobol indices. This section demonstrates how to print, plot, and interpret these indices for a structural reliability problem. The analysis helps determine the relative contribution of input variables to the model output and their interactions. 
</p>

<h4>Show results - Sobol indices</h4>

<p align="justify">
    How do we display the Sobol indices calculated for the Ishigami function?
</p>

```python
# Show results in notebook file (use the dictionary's variable name in the code cell)
data_sobol

# or 
# Show results in Python file (using the print function)
print(data_sobol)
```

<p align="justify">
    To analyze the first-order and total-order Sobol indices for each variable:
</p>

```python
# Convert Sobol indices to a table
from tabulate import tabulate
print(tabulate(data_sobol, headers='keys', tablefmt='psql'))
```

Expected output:

```bash
+----+-----------+----------+
|    |       s_i |      s_t |
|----+-----------+----------|
|  0 | 0.312931  | 0.547654 |
|  1 | 0.44652   | 0.433496 |
|  2 | 0.0097489 | 0.220905 |
+----+-----------+----------+
```

<ul>
    <li><code>s_i</code>: First-order Sobol index, representing the individual contribution of the variable to the output variance.</li>
    <li><code>s_t</code>: Total-order Sobol index, representing the overall contribution, including interactions with other variables.</li>
</ul>

<h4>Plot results - Sobol indices</h4>

<p align="justify">
    How do we visualize the Sobol indices as bar charts to interpret the results?
</p>

```python
# Libraries
import matplotlib.pyplot as plt

# Extract values
variables = ['x_0', 'x_1', 'x_2']
s_i = [data_sobol.iloc[var]['s_i'] for var in range(len(variables))]
s_t = [data_sobol.iloc[var]['s_t'] for var in range(len(variables))]

# Plot bar chart for Sobol indices
x = range(len(variables))
width = 0.35

plt.bar(x, s_i, width, label='First-order (s_i)', color='blue', alpha=0.7)
plt.bar([p + width for p in x], s_t, width, label='Total-order (s_t)', color='orange', alpha=0.7)

plt.xlabel("Variables")
plt.ylabel("Sobol Indices")
plt.xticks([p + width / 2 for p in x], variables)
plt.legend()
plt.show()
```

Expected plot:

<center>
    <img src="assets/images/sobol_output.png" height="auto">
    <p align="center"><b>Figure 1.</b> Sobol indices for the Ishigami function.</p>
</center>

<h4>Save results to a file</h4>

<p align="justify">
    To save the Sobol indices for further analysis or reporting:
</p>

```python
# Save results to a CSV file
data_sobol.to_excel('sobol_indices.xlsx', index=False)

print("Sobol indices saved to 'sobol_indices.xlsx'")
```

