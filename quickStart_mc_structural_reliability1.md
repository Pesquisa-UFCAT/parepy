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

Example 1 - CRUDE MONTE CARLO
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
import pandas as pd
pd.set_option('display.max_columns', None)

from parepy_toolbox import sampling_algorithm_structural_analysis
from obj_function import nowak_collins_example

# Dataset
f = {'type': 'normal', 
    'parameters': {'mean': 40.3, 'sigma': 4.64}, 
    'stochastic variable': False, 
    }

p = {'type': 'gumbel max',
    'parameters': {'mean': 10.2, 'sigma': 1.12}, 
    'stochastic variable': False, 
    }

w = {'type': 'lognormal',
    'parameters': {'mean': 0.25, 'sigma': 0.025}, 
    'stochastic variable': False, 
    }

var = [f, p, w]

# PAREpy setup
setup = {
             'number of samples': 70000, 
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

<!-- <h1>View results</h1>
<p align="justify">
Following instructions to see all results:
</p>

```python
print...bla bla bla
``` -->

Pós-Processamento dos Resultados
{: .label .label-red }

<p align="justify"> Após a simulação de Monte Carlo, é importante analisar os resultados para tomar decisões fundamentadas. Esta seção demonstra como processar os resultados da simulação armazenados no DataFrame <code>results</code>. As principais operações incluem visualizar a distribuição dos valores da função de estado limite (\(\boldsymbol{G}\)), identificar os pontos mais prováveis próximos à falha (\(G \approx 0\)) e visualizar variáveis aleatórias relacionadas (\(\boldsymbol{R}\) e \(\boldsymbol{S}\)) </p>


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
    </tbody>
</table>
