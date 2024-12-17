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

where:  
- \( x = \{x_0, x_1, x_2\} \in [-\pi, \pi]^3 \) are the input variables, limited to the domain \([-\pi, \pi]\);  
- \( a \) and \( b \) are adjustable parameters that control the relative impact of each term in the function.

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

OUTPUT
{: .label .label-red }

```bash
+----+-----------+-----------+
|    |       s_i |       s_t |
|----+-----------+-----------|
|  0 |  0.290423 | -0.611217 |
|  1 |  1.08608  | -0.405668 |
|  2 | -0.426312 |  0.298219 |
+----+-----------+-----------+
```


