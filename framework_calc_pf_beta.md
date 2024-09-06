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

<h3>PF and Beta Calculation</h3>
<br>
<p align = "justify">
    This function calculates the mean values of <i>pf</i> (probability of failure) and <i>beta</i> based on the columns of a DataFrame that start with the prefix <code>'I_'</code>. It returns two DataFrames: one with the calculated mean values of <i>pf</i> and another with the same values for <i>beta</i>.
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
        <td>DataFrame containing the columns to be processed. Only columns starting with 'I_' will be considered.</td>
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
       <td>DataFrame containing the mean values of <i>pf</i>.</td>
       <td>DataFrame</td>
   </tr>
   <tr>
       <td><code>beta_df</code></td>
       <td>DataFrame containing the same values as <code>pf_df</code>, representing <i>beta</i> values.</td>
       <td>DataFrame</td>
   </tr>
</table>

<h4><i>Example Usage</i></h4>
<p align = "justify" id = "pf-beta-example"></p>

VARIABLES SETTINGS
{: .label .label-red }

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
        <td><code>'df'</code></td>
        <td>DataFrame with columns to process. Only columns starting with 'I_' are used.</td>
        <td>DataFrame</td>
    </tr>
</table>

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, the <code>calc_pf_beta</code> function processes the DataFrame to calculate the mean values of <i>pf</i> and <i>beta</i> from the columns starting with 'I_'. Both results are returned in separate DataFrames.</i>
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
[0.0019, 0.0019]
PF:
+--------+--------+
|   0    |   1    |
+--------+--------+
| 0.0019 | 0.0019 |
+--------+--------+
Beta:
+--------+--------+
|   0    |   1    |
+--------+--------+
| 0.0019 | 0.0019 |
+--------+--------+
```