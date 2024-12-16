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
       <td><code>s_i</code></td>
       <td>
           A list containing the first-order Sobol sensitivity indices for each input variable.
       </td>
       <td>List</td>
   </tr>
   <tr>
       <td><code>s_t</code></td>
       <td>
           A list containing the total-order Sobol sensitivity indices for each input variable.
       </td>
       <td>List</td>
   </tr>
</table>

EXAMPLE
{: .label .label-blue }

This example demonstrates how to use the `sobol_algorithm` function to calculate the Sobol indices for a structural reliability problem.

of_FILE.PY
{: .label .label-red }

```python
def nowak_collins_example(x, none_variable):
    """Objective function for the Nowak example (tutorial).
    """

    # Random variables
    f_y = x[0]
    p_load = x[1]
    w_load = x[2]
    capacity = 80 * f_y
    demand = 54 * p_load + 5832 * w_load

    # State limit function
    constraint = capacity - demand

    return [capacity], [demand], [constraint]
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
             'number of samples': 10000, 
             'number of dimensions': len(var), 
             'numerical model': {'model sampling': 'mcs'}, 
             'variables settings': var, 
             'number of state limit functions or constraints': 1, 
             'none variable': None,
             'objective function': nowak_collins_example,
             'name simulation': None,
        }
s_i, s_t = sobol_algorithm(setup)
print(f"First-order Sobol indices: {s_i}")
print(f"Total-order Sobol indices: {s_t}")
```

Example Output:
```bash
```