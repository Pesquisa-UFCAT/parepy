---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 6
has_children: false
has_toc: false
title: calc_pf_beta
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->


<h3>Calculation of Probability of Failure (pf) and Beta</h3>
<br>
<p align = "justify">
    This function calculates the probability of failure (pf) and beta for each column in the DataFrame that starts with 'I_' (Indicator function). The function returns two DataFrames: one with the mean values of pf and another with the corresponding beta values.
</p>

```python
calc_pf_beta(df)
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
        <td><code>df</code></td>
        <td>DataFrame containing the columns with boolean values related to the indicator function (columns starting with 'I_').</td>
        <td>DataFrame</td>
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
       <td><code>pf_df</code></td>
       <td>DataFrame containing the mean values for pf (probability of failure) for each 'I_' column.</td>
       <td>DataFrame</td>
   </tr>
   <tr>
       <td><code>beta_df</code></td>
       <td>DataFrame containing the corresponding beta values calculated from the pf values.</td>
       <td>DataFrame</td>
   </tr>
</table>

<h4><i>Example Usage</i></h4>
<p align = "justify" id = "pf-beta-example"></p>

MODEL PARAMETERS
{: .label .label-red }

<h6><i>DataFrame Example</i></h6>

```python
data = {
    'X_0': [43.519326, 40.184658, 46.269007, 36.370403, 40.089100],
    'X_1': [11.222943, 11.044150, 10.586153, 9.523268, 9.728168],
    'R_1': [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
    'I_0': [0.0, 0.0, 0.0, 0.0, 0.0],
    'I_1': [0.0, 0.0, 0.0, 0.0, 0.0]
}

df = pd.DataFrame(data)
```

<table style = "width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead>
    <tr>
        <td><code>'I_1', 'I_2''</code></td>
        <td>Columns containing boolean values related to the indicator function.</td>
        <td>Integer (0 or 1)</td>
    </tr>
</table>

VARIABLES SETTINGS
{: .label .label-red }

```python
pf_df, beta_df = calc_pf_beta(df)
```

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, the <code>calc_pf_beta</code> function processes a DataFrame with three indicator columns ('I_1', 'I_2'). It returns two DataFrames: one containing the mean values of probability of failure (pf) for each column, and another containing the corresponding beta values.</i>
</p>
```

```python
data = {
    'X_0': [43.519326, 40.184658, 46.269007, 36.370403, 40.089100],
    'X_1': [11.222943, 11.044150, 10.586153, 9.523268, 9.728168],
    'X_2': [0.189671, 0.247242, 0.238284, 0.276446, 0.260700],
    'I_0': [0.0, 0.0, 0.0, 0.0, 0.0],
    'I_1': [0.0, 0.0, 0.0, 0.0, 0.0]
}

serial_df = pd.DataFrame(data)

pf_df, beta_df, result_df = calc_pf_beta(serial_df)

# Exibindo os resultados
print(f'PF:\n{tabulate(pf_df, headers="keys", tablefmt="pretty", showindex=False)}')
print(f'Beta:\n{tabulate(beta_df, headers="keys", tablefmt="pretty", showindex=False)}')
```
```
PF:
+---------+---------+
|   I_0   |   I_1   |
+---------+---------+
| 0.00188 | 0.00188 |
+---------+---------+
Beta:
+--------------------+--------------------+
|        I_0         |        I_1         |
+--------------------+--------------------+
| 2.8976248913337237 | 2.8976248913337237 |
+--------------------+--------------------+
```