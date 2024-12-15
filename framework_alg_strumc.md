---
layout: home
parent: algorithms
grand_parent: Framework
nav_order: 1
has_children: false
has_toc: false
title: sampling_algorithm_structural_analysis
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<p align = "justify">
    This function creates the samples and evaluates the limit state functions in structural reliability problems.
</p>

```python
results_about_data, failure_prob_list, beta_list = sampling_algorithm_structural_analysis(setup)
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
        <td>Setup settings.</td>
        <td>Dictionary</td>
    </tr>
    <tr>
        <td><code>number of samples</code></td>
        <td>Number of samples (key in setup dictionary)</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>numerical model</code></td>
        <td>Numerical model settings (key in setup dictionary). See examples in <a href="#models">Table 1</a></td>
        <td>Dictionary</td>
    </tr>
    <tr>
        <td><code>variables settings</code></td>
        <td>Variables settings (key in setup dictionary). This variable is a list of dictionaries. See examples in <a href="#variables">Table 2</a></td>
        <td>List</td>
    </tr>
    <tr>
        <td><code>number of state limit functions or constraints</code></td>
        <td>Number of state limit functions or constraints</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>none_variable</code></td>
        <td>None variable. User can use this variable in the objective function (key in setup dictionary)</td>
        <td>None, List, Float, Dictionary, String, or any</td>
    </tr>
    <tr>
        <td><code>objective function</code></td>
        <td>Objective function. The PAREpy user defines this function (key in setup dictionary)</td>
        <td>Python function</td>
    </tr>
    <tr>
        <td><code>name simulation</code></td>
        <td>Output filename (key in setup dictionary)</td>
        <td>String or None</td>
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
        <td>Results about reliability analysis</td>
        <td>DataFrame</td>
    </tr>
    <tr>
        <td><code>failure_prob_list</code></td>
        <td>Failure probability list</td>
        <td>List</td>
    </tr>
    <tr>
        <td><code>beta_list</code></td>
        <td>Beta list</td>
        <td>List</td>
    </tr>
</table>

<p align="justify">
To use the sample algorithm, you must choose the algorithm and variable types and correctly fill in the <code>'numerical model'</code> and <code>'variables settings'</code> keys. See the following examples and <a href="https://wmpjrufg.github.io/PAREPY/framework_distributions_.html" target="_blank">distributions</a>.
</p>

<p align="justify" id="models"></p>
<p align="left"><b>Table 1.</b> <code>'numerical model'</code> key.</p>
<center>
    <table style = "width:100%">
        <thead>
            <tr>
            <th>Type</th>
            <th>Sintax</th>
            </tr>
        </thead>
        <tr>
            <td>Crude Monte Carlo</td>
            <td><code>'numerical model': {'model sampling': 'mcs'}</code></td>
        </tr>
        <tr>
            <td>Latin Hypercube</td>
            <td><code>'numerical model': {'model sampling': 'lhs'}</code></td>
        </tr>
        <tr>
            <td>Stochastic - Crude Monte Carlo considering five time steps</td>
            <td><ul><li><code>'numerical model': {'model sampling': 'mcs-time', 'time steps': 5}</code><sup>1,2</sup></li><li>and <code>'none variable': {'time analysis': list(np.linspace(0, 50, num=5, endpoint=True))}</code><sup>1,2</sup></li></ul></td>
        </tr>
        <tr>
            <td>Stochastic - Latin Hypercube considering five time steps</td>
            <td><ul><li><code>'numerical model': {'model sampling': 'lhs-time', 'time steps': 5}</code><sup>1,2</sup></li><li>and <code>'none variable': {'time analysis': list(np.linspace(0, 50, num=5, endpoint=True))}</code><sup>1,2</sup></li></ul></td>
        </tr>
    </table>
</center>

{: .important }
>¹When applying a stochastic procedure, use a list in ```'none variables'``` with the same length as ```'time steps'```. In this example, we use five time steps between 0 and 50 years. In this case, a user should import the **Numpy** library to use ```np. linspace```. Another library can be used to create a list.

{: .important }
>²When applying a stochastic procedure, use the following code on top of the objective function:    

```python
id_analysis = int(x[-1])
time_step = none_variable['time analysis']
t_i = time_step[id_analysis] 
```

<p align="justify" id="variables"></p>
<p align="left"><b>Table 2.</b> <code>'variable settings'</code> key. Dictionary details.</p>
<center>
    <table style = "width:100%">
        <thead>
            <tr>
            <th>Key</th>
            <th>Description</th>
            <th>Example</th>
            </tr>
        </thead>
        <tr>
            <td><code>'type'</code></td>
            <td>Type of the distribution</td>
            <td><code>'type': 'normal',</code></td>
        </tr>
        <tr>
            <td><code>'parameters'</code></td>
            <td>Parameters of the distribution. See the <a href="https://wmpjrufg.github.io/PAREPY/framework_distributions_.html" target="_blank" rel="noopener noreferrer">parameters </a>for each distribution</td>
            <td><code>'parameters': {'mean': 40.3, 'sigma': 4.64},</code></td>
        </tr>
        <tr>
            <td><code>'stochastic variable'</code></td>
            <td>Stochastic process (<code>'True' or 'False'</code>). Use <code>'True'</code> when you wish apply stochastic process</td>
            <td><code>'stochastic variable': False</code></td>
        </tr>
    </table>
</center>

<p align="justify">
More details about the reliability method are shown in examples <a href="#example1">1</a> and <a href="#example2">2</a>.
</p>

<p align="justify" id="example1"></p>

Example 1
{: .label .label-blue }

<p align="justify">
    <i>
        Consider the simply supported beam show in example 5.1 Nowak and Collins <a href="#ref1">[1]</a>. The beam is subjected to a concentrated live load \(p\) and a uniformly distributed dead load \(w\). Assume \(\boldsymbol{P}\) (concentrated live load), \(\boldsymbol{W}\) (uniformly distributed dead load) and the yield stress, \(\boldsymbol{F_y}\), are random quantities; the length \(l\) and the plastic setion modulus \(z\) are assumed to be precisely know (deterministic). The distribution parameters for \(\boldsymbol{P}, \boldsymbol{W}\) and \(\boldsymbol{F_y}\) are given bellow:
    </i>
</p>

<table style = "width:100%; text-align: center;">
    <tr>
        <th style="width: 25%;">Variable</th>
        <th style="width: 25%;">Distribution</th>
        <th style="width: 25%;">Mean</th>
        <th style="width: 25%; text-align: justify;">Coefficient of Variation (COV)</th>
    </tr>
    <tr>
        <td style="width: 25%;">Yield stress \(\left(\boldsymbol{F_y}\right)\)</td>
        <td style="width: 25%;">Normal</td>
        <td style="width: 25%;">40.3</td>
        <td style="width: 25%;">0.115</td>
    </tr>
    <tr>
        <td style="width: 25%;">Live load \(\left(\boldsymbol{P}\right)\)</td>
        <td style="width: 25%;">Gumbel max.</td>
        <td style="width: 25%;">10.2</td>
        <td style="width: 25%;">0.110</td>
    </tr>
    <tr>
        <td style="width: 25%;">Dead load \(\left(\boldsymbol{W}\right)\)</td>
        <td style="width: 25%;">Log-normal</td>
        <td style="width: 25%;">0.25</td>
        <td style="width: 25%;">0.100</td>
    </tr>
</table>

<p align="justify">
The limit state function for beam bending can be expressed as:
</p>

<table style = "width:100%">
    <tr>
        <td style="width: 90%;">\[ \boldsymbol{R} = 80 \cdot \boldsymbol{F_y} \]</td>
        <td style="width: 10%;"><p align = "right" id = "eq1">(1)</p></td>
    </tr>
    <tr>
        <td style="width: 90%;">\[ \boldsymbol{S} = 54 \cdot \boldsymbol{P} + 5832 \cdot \boldsymbol{W} \]</td>
        <td style="width: 10%;"><p align = "right" id = "eq2">(2)</p></td>
    </tr>
    <tr>
        <td style="width: 90%;">\[ \boldsymbol{G} = \boldsymbol{R} - \boldsymbol{S} \begin{cases}
\leq 0 & \text{failure}\\ 
> 0 & \text{safe}
\end{cases} \]
        </td>
        <td style="width: 10%;"><p align = "right" id = "eq3">(3)</p></td>
    </tr>
</table>

of_file.py
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

your_problem.ipynb
{: .label .label-red }

```python
# Libraries
from parepy_toolbox import sampling_algorithm_structural_analysis
from obj_function import nowak_collins_example

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
             'name simulation': 'nowak_collins_example',
        }

# Call algorithm
results, pf, beta = sampling_algorithm_structural_analysis(setup)
```

<h3>Post-processing</h3>

<p align="justify">
    The results of the sampling simulation need to be evaluated. This section demonstrates how to print, plot, and show these results. The results of the sampling simulation need to be assessed. This section demonstrates how to print, plot, and show these results. <b>Consider Example 1</b> to show examples of post-processing.
</p>

<h4>Show results - all samples</h4>

<p align="justify">
    What are the columns' names in the results of Example 1?
</p>

```bash
+-----+---------+----------+----------+---------+---------+------------+-------+
|     |     X_0 |      X_1 |      X_2 |     R_0 |     S_0 |        G_0 |   I_0 |
|-----+---------+----------+----------+---------+---------+------------+-------|
|   0 | 33.9686 | 10.9885  | 0.289494 | 2717.48 | 2281.71 |  435.773   |     0 |
|   1 | 41.6526 |  8.38575 | 0.242897 | 3332.21 | 1869.4  | 1462.8     |     0 |
|   2 | 52.0513 |  9.42486 | 0.195777 | 4164.1  | 1650.72 | 2513.39    |     0 |
|   3 | 37.6799 |  9.91397 | 0.254184 | 3014.39 | 2017.75 |  996.637   |     0 |
|   4 | 31.1943 |  8.96956 | 0.250925 | 2495.54 | 1947.75 |  547.791   |     0 |
|   5 | 36.3056 | 10.1379  | 0.242374 | 2904.45 | 1960.97 |  943.48    |     0 | 
...
| 997 | 41.493  |  9.64558 | 0.255638 | 3319.44 | 2011.74 | 1307.69    |     0 |
| 998 | 55.8406 |  9.76341 | 0.284005 | 4467.25 | 2183.54 | 2283.7     |     0 |
| 999 | 37.2467 | 10.9352  | 0.242343 | 2979.73 | 2003.85 |  975.886   |     0 |
+-----+---------+----------+----------+---------+---------+------------+-------+
```

<ul>
    <li><code>X_</code>: Random variables;</li>
    <li><code>R_</code>: First return in objective function (User defined);</li>
    <li><code>S_</code>: Second return in objective function (User defined);</li>
    <li><code>G_</code>: Second return in objective function (User defined);</li>
    <li><code>I_</code>: Indicator function (PAREpy generate).</li>
</ul>

```python
# Show results in notebook file (simply use the DataFrame's variable name in code cell) 
results

# or 
# Show results in python file (using print function)
print(results)
```

<p align="justify">
    This problem presents one state limit function. How do we show columns that results respect a relation \(\boldsymbol{G} \geq 0 \)?
</p>

```python
# Libraries
import pandas as pd

# Analysis already realized
sorted_positive = results[results['G_0'] >= 0].sort_values(by='G_0', ascending=True)

# Show results in notebook file (simply use the DataFrame's variable name in code cell) 
sorted_positive.head(3)

# or 
# Show results in python file (using print function)
print(sorted_positive.head(3))
```

{: .important } 
>
> PAREpy is using the same ID sequence in your data frames. Therefore, any column starts with zero.

<h4>Plot results - all samples</h4>

<p align="justify">
    How do we plot \(\boldsymbol{G}_0\) histogram?
</p>

```python
# Libraries
import matplotlib.pyplot as plt

# Plot histogram of G_0
plt.hist(results['G_0'], bins=50, alpha=0.7, color='blue')
plt.xlabel("Constraint Value (G_0)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

<p align="justify">
    How do we plot \(\boldsymbol{S}_0\) and \(\boldsymbol{R}_0\) in unique histogram?
</p>

```python
# Libraries
import matplotlib.pyplot as plt

# Plot histograms - R_0 and S_0
plt.hist(results['R_0'], bins=50, alpha=0.5, color='green', label='Resistance R_0')
plt.hist(results['S_0'], bins=50, alpha=0.5, color='orange', label='Demand S_0')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

<h4>Show \(p_f\) and \(\beta\) results</h4>

<p align="justify">
    Show \(p_f\) results in list format.
</p>

```python
# Acess pf results
pf_list = pf.values.flatten().tolist()
print(pf_list)
```

<p align="justify">
    Show \(\beta\) results in list format.
</p>

```python
# Acess beta results
beta_list = beta.values.flatten().tolist()
print(beta_list)
```

<p align="justify">
    How do we print \(p_f\) and \(\beta\) together?
</p>

```python
pf_list = pf.values.flatten().tolist()
beta_list = beta.values.flatten().tolist()
for i, (p, b) in enumerate(zip(pf_list, beta_list)):
    print(f"State Limite function (g): {i}, pf: {p:.6f}, beta: {b:.6f}")
```

<p align="justify" id="example2"></p>
Example 2
{: .label .label-blue }

<p align="justify">
<i>
    Consider the simply supported beam show in example 5.1 Nowak and Collins <a href="#ref1">[1]</a>. The beam is subjected to a concentrated live load \(p\) and a uniformly distributed dead load \(w\). Assume \(\boldsymbol{P}\) (concentrated live load), \(\boldsymbol{W}\) (uniformly distributed dead load) and the yield stress, \(\boldsymbol{F_y}\), are random quantities; the length \(l\) and the plastic setion modulus \(z\) are assumed to be precisely know (deterministic). The distribution parameters for \(\boldsymbol{P}, \boldsymbol{W}\) and \(\boldsymbol{F_y}\) are given bellow:
</i>
</p>

<table style = "width:100%; text-align: center;">
    <tr>
        <th style="width: 25%;">Variable</th>
        <th style="width: 25%;">Distribution</th>
        <th style="width: 25%;">Mean</th>
        <th style="width: 25%; text-align: justify;">Coefficient of Variation (COV)</th>
    </tr>
    <tr>
        <td style="width: 25%;">Yield stress \(\left(\boldsymbol{F_y}\right)\)</td>
        <td style="width: 25%;">Normal</td>
        <td style="width: 25%;">40.3</td>
        <td style="width: 25%;">0.115</td>
    </tr>
    <tr>
        <td style="width: 25%;">Live load¹ \(\left(\boldsymbol{P}\right)\)</td>
        <td style="width: 25%;">Gumbel max.</td>
        <td style="width: 25%;">10.2</td>
        <td style="width: 25%;">0.110</td>
    </tr>
    <tr>
        <td style="width: 25%;">Dead load \(\left(\boldsymbol{W}\right)\)</td>
        <td style="width: 25%;">Log-normal</td>
        <td style="width: 25%;">0.25</td>
        <td style="width: 25%;">0.100</td>
    </tr>
    <tr>
        <td style = "text-align: left;" colspan="4">¹Stochastic random variable</td>
    </tr>
</table>

<p align="justify">
The limit state function for beam bending can be expressed as:
</p>

<table style = "width:100%">
    <tr>
        <td style="width: 90%;">\[ \boldsymbol{R} = 80 \cdot \boldsymbol{F_y} \cdot D\]</td>
        <td style="width: 10%;"><p align = "right" id = "eq1">(1)</p></td>
    </tr>
    <tr>
        <td style="width: 90%;">\[ \boldsymbol{S} = 54 \cdot \boldsymbol{P} + 5832 \cdot \boldsymbol{W} \]</td>
        <td style="width: 10%;"><p align = "right" id = "eq2">(2)</p></td>
    </tr>
    <tr>
        <td style="width: 90%;">\[ \boldsymbol{G} = \boldsymbol{R} - \boldsymbol{S} \begin{cases}
\leq 0 & \text{failure}\\ 
> 0 & \text{safe}
\end{cases} \]
        </td>
        <td style="width: 10%;"><p align = "right" id = "eq3">(3)</p></td>
    </tr>
</table>

<p align="justify">
Consider equation <a href="#eq4">(4)</a> for resistance degradation \(\left(D\right)\) <a href="#ref2">[2]</a>. Use 50 years to stochastic analysis (five time steps). Assume that \(P\) load is a stochastic process. 
</p>

<table style = "width:100%">
    <tr>
        <td style="width: 90%;">\[ D(t_i) = 1 - \frac{0.2}{t_i} \cdot 0.01 \]</td>
        <td style="width: 10%;"><p align = "right" id = "eq4">(4)</p></td>
    </tr>
</table>

of_file.py
{: .label .label-red }

```python
def nowak_collins_time_example(x, none_variable):
    """Objective function for the Nowak example (tutorial).
    """
    
    # User must copy and paste this code in time reliability objective function
    ###########################################
    id_analysis = int(x[-1])
    time_step = none_variable['time analysis']
    t_i = time_step[id_analysis] 
    # t_i is a time value from your list of times entered in the 'none variable' key.
    ###########################################

    # Random variables
    f_y = x[0]
    p_load = x[1]
    w_load = x[2]
    
    # Degradation criteria
    if t_i == 0:
        degrad = 1
    else:
        degrad = 1 - (0.2 / t_i) * 1E-2

    # Capacity and demand
    capacity = 80 * f_y * degrad
    demand = 54 * p_load + 5832 * w_load

    # State limit function
    constraint = capacity - demand

    return [capacity], [demand], [constraint]
```

your_problem.ipynb
{: .label .label-red }

```python
# Libraries
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

from parepy_toolbox import sampling_algorithm_structural_analysis
from obj_function import nowak_collins_time_example

# Statement random variables
f = {
        'type': 'normal', 
        'parameters': {'mean': 40.3, 'sigma': 4.64}, 
        'stochastic variable': False, 
    }

p = {
        'type': 'gumbel max',
        'parameters': {'mean': 10.2, 'sigma': 1.12}, 
        'stochastic variable': True, 
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
             'numerical model': {'model sampling': 'mcs-time', 'time steps': 5}, 
             'variables settings': var, 
             'number of state limit functions or constraints': 1, 
             'none variable': {'time analysis': list(np.linspace(0, 50, num=5, endpoint=True))},
             'objective function': nowak_collins_time_example,
             'name simulation': 'nowak_collins_time_example',
        }

# Call algorithm
results, pf, beta = sampling_algorithm_structural_analysis(setup)
```

<h3>Post-processing</h3>

<h4>Show results - all samples</h4>

<p align="justify">
    What are the columns' names in the results of Example 2?
</p>

```bash
+-----+-----------+-----------+-----------+-----------+-----------+-----------+------------+------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
|     |   X_0_t=0 |   X_0_t=1 |   X_1_t=0 |   X_1_t=1 |   X_2_t=0 |   X_2_t=1 |   STEP_t_0 |   STEP_t_1 |   R_0_t=0 |   R_0_t=1 |   S_0_t=0 |   S_0_t=1 |   G_0_t=0 |   G_0_t=1 |   I_0_t=0 |   I_0_t=1 |
|-----+-----------+-----------+-----------+-----------+-----------+-----------+------------+------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------|
|   0 |   36.5333 |   33.3011 |   9.14011 |   9.14011 |  0.228967 |  0.228967 |          0 |          1 |   2922.66 |   2663.98 |   1828.9  |   1828.9  | 1093.76   |  835.076  |         0 |         0 |
|   1 |   43.268  |   41.0214 |   9.38534 |   9.38534 |  0.191533 |  0.191533 |          0 |          1 |   3461.44 |   3281.58 |   1623.83 |   1623.83 | 1837.61   | 1657.75   |         0 |         0 |
|   2 |   46.7621 |   45.58   |   8.56684 |   8.56684 |  0.229873 |  0.229873 |          0 |          1 |   3740.96 |   3646.25 |   1803.23 |   1803.23 | 1937.73   | 1843.02   |         0 |         0 |
|   3 |   40.9082 |   42.5734 |  11.2148  |  11.2148  |  0.270366 |  0.270366 |          0 |          1 |   3272.66 |   3405.74 |   2182.37 |   2182.37 | 1090.28   | 1223.36   |         0 |         0 |
...
| 997 |   38.9421 |   42.0164 |  10.7914  |  10.7914  |  0.224753 |  0.224753 |          0 |          1 |   3115.36 |   3361.18 |   1893.49 |   1893.49 | 1221.87   | 1467.69   |         0 |         0 |
| 998 |   45.5127 |   38.2171 |   9.66603 |   9.66603 |  0.261417 |  0.261417 |          0 |          1 |   3641.02 |   3057.25 |   2046.55 |   2046.55 | 1594.47   | 1010.69   |         0 |         0 |
| 999 |   36.0261 |   41.376  |   8.56939 |   8.56939 |  0.24753  |  0.24753  |          0 |          1 |   2882.09 |   3309.95 |   1906.34 |   1906.34 |  975.745  | 1403.61   |         0 |         0 |
+-----+-----------+-----------+-----------+-----------+-----------+-----------+------------+------------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+-----------+
```

<ul>
    <li><code>X_i_t</code>: Random variables in specific time step;</li>
    <li><code>STEP_t_</code>: Time step ID;</li>
    <li><code>R_i_t</code>: First return in objective function (User defined) -  in specific time step;</li>
    <li><code>S_i_t</code>: Second return in objective function (User defined) -  in specific time step;</li>
    <li><code>G_i_t</code>: Second return in objective function (User defined) -  in specific time step;</li>
    <li><code>I_i_t</code>: Indicator function (PAREpy generate) -  in specific time step.</li>
</ul>

<h4>Show \(p_f\) and \(\beta\) results</h4>

<p align="justify">
    Show \(p_f\) results in list format. To view results about all time steps in \(G_0\) state limit function folliwing code:
</p>

```python
# Acess pf results
pf_list = pf['G_0'].tolist()
print(pf_list)
```

<p align="justify">
    Show \(\beta\) results in list format.
</p>

```python
# Acess beta results
beta_list = beta['G_0'].tolist()
print(beta_list)
```

<p align="justify">
    How do we print \(p_f\) and \(\beta\) together?
</p>

```python
pf_list = pf['G_0'].tolist()
beta_list = beta['G_0'].tolist()
for i, (p, b) in enumerate(zip(pf_list, beta_list)):
    print(f"Time step (id={i}, time={setup['none variable']['time analysis'][i]}), pf: {p:.6f}, beta: {b:.6f}")
```

<h1>Reference list</h1>

<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Reference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><p align = "center" id = "ref1">[1]</p></td>
            <td><p align = "left"><a href="https://doi.org/10.1007/s00521-016-2328-2" target="_blank" rel="noopener noreferrer">Nowak AS, Collins KR. Reliability of Structures. 2nd edition. CRC Press; 2012.</a></p></td>
        </tr>
        <tr>
            <td><p align = "center" id = "ref2">[2]</p></td>
            <td><p align = "left"><a href="https://doi.org/10.1007/s00521-016-2328-2" target="_blank" rel="noopener noreferrer">Beck AT. Confiabilidade e segurança das estruturas. Elsevier; 2019. ISBN 978-85-352-8895-7</a></p></td>
        </tr>
    </tbody>
</table>