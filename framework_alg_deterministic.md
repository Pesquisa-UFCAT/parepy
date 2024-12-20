<!-- ---
layout: home
parent: algorithms
grand_parent: Framework
nav_order: 3
has_children: false
has_toc: false
title: deterministic_algorithm_structural_analysis
--- -->

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h3>deterministic_algorithm_structural_analysis</h3>
<p align="justify">
    Solves the deterministic problem in structural reliability analysis using methods like First-Order Second-Moment (FOSM), FORM, and SORM. This function calculates reliability indices and failure probabilities based on the numerical model and variable settings provided.
</p>

```python
results_about_data, failure_prob_list, beta_list = deterministic_algorithm_structural_analysis(setup)
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
                <li><code>'objective function'</code>: A Python function defining the state limit function.</li>
                <br>
                <li><code>'gradient objective function'</code>: A Python function defining the gradient of the objective function.</li>
                <br>
                <li><code>'numerical model'</code>: A dictionary containing the model type (<code>'model'</code>) and the initial guess (<code>'initial guess'</code>).</li>
                <br>
                <li><code>'variables settings'</code>: A list of dictionaries defining variable properties (<code>'mean'</code> and <code>'sigma'</code>).</li>
                <br>
                <li><code>'number of iterations'</code>: An integer defining the number of iterations to perform.</li>
                <br>
                <li><code>'none variable'</code>: Additional user-defined input, used in the objective and gradient functions.</li>
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
       <td><code>results_about_data</code></td>
       <td>
           A DataFrame containing intermediate and final results of the algorithm.
           <ul>
               <li><code>'x0', 'x1'</code>: Components of \(X\) in the original variable space.</li>
               <li><code>'y0', 'y1'</code>: Components of \(Y\) in the normalized variable space.</li>
               <li><code>'state limit function'</code>: Values of \(G(X)\) at each iteration.</li>
               <li><code>'ϐ new'</code>: Calculated reliability indices (beta).</li>
           </ul>
       </td>
       <td>DataFrame</td>
   </tr>
   <tr>
       <td><code>failure_prob_list</code></td>
       <td>Placeholder value for failure probability.</td>
       <td>float</td>
   </tr>
   <tr>
       <td><code>beta_list</code></td>
       <td>Placeholder value for reliability index.</td>
       <td>float</td>
   </tr>
</table>

EXAMPLE
{: .label .label-blue }

This example demonstrates how to use the `deterministic_algorithm_structural_analysis` function to solve a structural reliability problem using the First-Order Second-Moment (FOSM) method.

of_FILE.PY
{: .label .label-red }

```python	
def form_1(x, none_variable):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    g = x1*x2 - 1400
    
    return g

def grad_form_1(x, none_variable):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    g = [x2, x1, x3] 
    
    return g
```

YOUR_PROBLEM.IPYNB
{: .label .label-red }

```python
from parepy_toolbox import deterministic_algorithm_structural_analysis
from obj_function import form_1, grad_form_1

# Dataset
f = {'type': 'normal', 'parameters': {'mean': 40.3, 'sigma': 4.64}, 'stochastic variable': True}
p = {'type': 'gumbel max', 'parameters': {'mean': 10.2, 'sigma': 1.12}, 'stochastic variable': False}
w = {'type': 'lognormal', 'parameters': {'mean': 0.25, 'sigma': 0.025}, 'stochastic variable': False}
var = [f, p, w]

setup = {   
            'objective function': form_1,
            'gradient objective function': grad_form_1,
             'numerical model': {'model': 'fosm', 'initial guess': [0, 0, 0]}, 
             'tolerance': 1e-6, 
             'max iterations': 1000,
             'none variable': None,
             'variables settings': var,  
             'number of iterations': 10,
        }


results_about_data, failure_prob_list, beta_list = deterministic_algorithm_structural_analysis(setup)
```

Example Output:
```bash
```