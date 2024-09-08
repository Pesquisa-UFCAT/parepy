---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 4
has_children: false
has_toc: false
title: calc_pf_beta
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h3>calc_pf_beta</h3>
<p align = "justify">
    Calculates the values of pf and beta from the columns of a DataFrame that start with 'I_' (Indicator function). If a .txt file path is passed, this function evaluates pf and beta values too.
</p>

```python
df_pf, df_beta = calc_pf_beta(df_or_path)
```

Input variables
{: .label .label-yellow }

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead>
    <tr>
        <td><code>df_or_path</code></td>
        <td>The DataFrame containing the columns with boolean values about indicator function, or a path to a .txt file</td>
        <td>DataFrame or String</td>
    </tr>
    <tr>
        <td><code>numerical_model</code></td>
        <td>Containing the numerical model parameters</td>
        <td>Dictionary</td>
    </tr>
    <tr>
        <td><code>n_constraints</code></td>
        <td>Number of state limit functions or constraints</td>
        <td>Integer</td>
    </tr>
</table>

Output variables
{: .label .label-yellow }

<table style = "width:100%">
   <thead>
     <tr>
       <th>Name</th>
       <th>Description</th>
       <th>Type</th>
     </tr>
   </thead>
   <tr>
       <td><code>df_pf</code></td>
       <td>DataFrame containing the values for probability of failure for each 'I_' column</td>
       <td>DataFrame</td>
   </tr>
   <tr>
       <td><code>df_beta</code></td>
       <td>DataFrame containing the values for beta for each 'I_' column</td>
       <td>DataFrame</td>
   </tr>
</table>
out 
Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, the calc_pf_beta function processes a DataFrame with seven columns: (a) random variable 0, random variable 1, random variable 2, state limit function 0, state limit function 1, indicator function 0 ('I_0'), indicator function 1 ('I_1'). Use this function to obtain the probability of failure and the reliability index.</i>
</p>

```python
# pip install tabulate or pip install --upgrade tabulate # external library (visit: https://pypi.org/project/tabulate/)
from tabulate import tabulate
import pandas as pd
from parepy_toolbox import calc_pf_beta

data =  {
            'X_0': [43.519326, 40.184658, 46.269007, 36.370403, 40.089100, 45.000000, 40.000000],
            'X_1': [11.222943, 11.044150, 10.586153, 9.523268, 9.728168, 10.000000, 10.000000],
            'X_2': [0.189671, 0.247242, 0.238284, 0.276446, 0.260700, 0.250000, 0.250000],
            'G_0': [200, 225, -10, 300, 325, -50, 0.01],
            'G_1': [-2, -3, -10, 300, 325, 50, 0.5],
            'I_0': [0, 0, 1, 0, 0, 1, 0],
            'I_1': [1, 1, 1, 0, 0, 0, 0]
        }
df = pd.DataFrame(data)
pf_df, beta_df = calc_pf_beta(serial_df, {'model sampling': 'mcs'}, 2)
print(f'pf:\n{tabulate(pf_df, headers="keys", tablefmt="pretty", showindex=False)}')
print(f'ϐ:\n{tabulate(beta_df, headers="keys", tablefmt="pretty", showindex=False)}')
``` 
```bash
pf:
+--------------------+---------------------+
|        I_0         |         I_1         |
+--------------------+---------------------+
| 0.2857142857142857 | 0.42857142857142855 |
+--------------------+---------------------+
ϐ:
+-------------------+---------------------+
|        I_0        |         I_1         |
+-------------------+---------------------+
| 0.565948821932863 | 0.18001236979270438 |
+-------------------+---------------------+
``` 

Example 2
{: .label .label-blue }

<p align = "justify">
    <i>In this example, the calc_pf_beta function processes a txt DataFrame with seven columns: (a) random variable 0, random variable 1, random variable 2, state limit function 0, state limit function 1, indicator function 0 ('I_0'), indicator function 1 ('I_1'). Use this function to obtain the probability of failure and the reliability index.</i>
</p>


#### example `.txt`
```bash
X_0	X_1	X_2	G_0	G_1	I_0	I_1
43.519326	11.222943	0.189671	200.0	-2.0	0	1
40.184658	11.04415	0.247242	225.0	-3.0	0	1
46.269007	10.586153	0.238284	-10.0	-10.0	1	1
36.370403	9.523268	0.276446	300.0	300.0	0	0
40.0891	9.728168	0.2607	325.0	325.0	0	0
45.0	10.0	0.25	-50.0	50.0	1	0
40.0	10.0	0.25	0.01	0.5	0	0
```

```python
# pip install tabulate or pip install --upgrade tabulate # external library (visit: https://pypi.org/project/tabulate/)
from tabulate import tabulate
import pandas as pd
from parepy_toolbox import calc_pf_beta

df = pd.read_csv('./example.txt', sep='\t')
pf_df, beta_df = calc_pf_beta(serial_df, {'model sampling': 'mcs'}, 2)
print(f'pf:\n{tabulate(pf_df, headers="keys", tablefmt="pretty", showindex=False)}')
print(f'ϐ:\n{tabulate(beta_df, headers="keys", tablefmt="pretty", showindex=False)}')
``` 
```bash
pf:
+--------------------+---------------------+
|        I_0         |         I_1         |
+--------------------+---------------------+
| 0.2857142857142857 | 0.42857142857142855 |
+--------------------+---------------------+
ϐ:
+-------------------+---------------------+
|        I_0        |         I_1         |
+-------------------+---------------------+
| 0.565948821932863 | 0.18001236979270438 |
+-------------------+---------------------+
``` 