---
layout: home
parent: Quick Start
nav_order: 1
has_children: false
has_toc: false
title: Structural reliability
---

<!--Don't delete this script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

Example 1
{: .label .label-blue }

<p align="justify">
Consider the simply supported beam show in example 5.1 Nowak and Collins <a href="#ref1">[1]</a>. The beam is subjected to a concentrated live load \(p\) and a uniformly distributed dead load \(w\). The loads are random variables. Assume \(\boldsymbol{P},\; \boldsymbol{W}\) and the yield stress, \(\boldsymbol{F_y}\), are random quantities; the length \(l\) and the plastic setion modulus \(z\) are assumed to be precisely know (deterministic). The distribution parameters for \(\boldsymbol{P}, \boldsymbol{W}\) and \(\boldsymbol{F_y}\) are given bellow:
</p>

<table style = "width:100%; text-align: center;">
    <tr>
        <th style="width: 25%;">Variable</th>
        <th style="width: 25%;">Distribution</th>
        <th style="width: 25%;">Mean</th>
        <th style="width: 25%;">coefficient of variation (COV)</th>
    </tr>
    <tr>
        <td style="width: 25%;">Live load \(\left(\boldsymbol{F_y}\right)\)</td>
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
        <td style="width: 90%;">\[ \boldsymbol{G} = \boldsymbol{S} - \boldsymbol{R} \begin{cases}
\leq 0 & \text{safe}\\ 
> 0 & \text{failure}
\end{cases} \]</td>
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
    capacity = 80*f_y
    demand = 54*p_load + 5832*w_load

    # State limit function
    constraint = demand - capacity

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
from obj_function import nowak_collins_example

# Dataset
f = {'type': 'normal', 'loc': 40.3, 'scale': 4.64, 'seed': None}
p = {'type': 'gumbel max', 'loc': 10.2, 'scale': 1.12, 'seed': None}
w = {'type': 'lognormal', 'loc': 0.25, 'scale': 0.025, 'seed': None}
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
             'type process': 'auto',
             'name simulation': 'nowak_collins_example',
        }

# Call algorithm
results, pf, beta = sampling_algorithm_structural_analysis(setup)
```

```bash
PARE^py Report: 

- Output file name: nowak_example_MCS_20240505-135418.txt
- Processing time (s): 7.355956315994263  (serial kernel)
```

{: .note }
> See more details in [sampling_algorithm_structural_analysis](https://wmpjrufg.github.io/PAREPY/framework_samplingAlgStructuralAnalysis.html)


<h1><b>REFERENCE LIST</b></h1>

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
    </tbody>
</table>