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
       <td>Dict</td>
   </tr>
   <tr>
       <td><code>s_t</code></td>
       <td>
           A list containing the total-order Sobol sensitivity indices for each input variable.
       </td>
       <td>Dict</td>
   </tr>
</table>

EXAMPLE
{: .label .label-blue }

This example demonstrates how to use the `sobol_algorithm` function to calculate the Sobol indices for a structural reliability problem.

of_FILE.PY
{: .label .label-red }

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
s_i, s_t = sobol_algorithm(setup)
print(s_i)
print(s_t)
```

Example Output:
```bash
[-0.4364895442617733, 0.24087159929570906, -1.3529608667470359]
[-0.1942959571784033, 2.453684230822736, 0.9118407402354742]
```
