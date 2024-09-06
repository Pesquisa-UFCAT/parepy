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

<h4><i>Calculate exemple</i></h4>
<p align = "justify" id = "pf-beta-example"></p>

VARIABLES SETTINGS
{: .label .label-red }

<p align = "justify">
    Dataframe serial input exemple:
</p>
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X_0</th>
      <th>X_1</th>
      <th>X_2</th>
      <th>R_0</th>
      <th>R_1</th>
      <th>S_0</th>
      <th>S_1</th>
      <th>G_0</th>
      <th>G_1</th>
      <th>I_0</th>
      <th>I_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43.519326</td>
      <td>11.222943</td>
      <td>0.189671</td>
      <td>3481.546080</td>
      <td>NaN</td>
      <td>1712.201638</td>
      <td>1712.201638</td>
      <td>1769.344442</td>
      <td>1769.344442</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40.184658</td>
      <td>11.044150</td>
      <td>0.247242</td>
      <td>3214.772659</td>
      <td>NaN</td>
      <td>2038.301401</td>
      <td>2038.301401</td>
      <td>1176.471258</td>
      <td>1176.471258</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46.269007</td>
      <td>10.586153</td>
      <td>0.238284</td>
      <td>3701.520547</td>
      <td>NaN</td>
      <td>1961.325563</td>
      <td>1961.325563</td>
      <td>1740.194984</td>
      <td>1740.194984</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36.370403</td>
      <td>9.523268</td>
      <td>0.276446</td>
      <td>2909.632213</td>
      <td>NaN</td>
      <td>2126.486910</td>
      <td>2126.486910</td>
      <td>783.145303</td>
      <td>783.145303</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40.089100</td>
      <td>9.728168</td>
      <td>0.260700</td>
      <td>3207.127976</td>
      <td>NaN</td>
      <td>2045.722536</td>
      <td>2045.722536</td>
      <td>1161.405440</td>
      <td>1161.405440</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>49995</th>
      <td>50.100028</td>
      <td>12.170102</td>
      <td>0.285213</td>
      <td>4008.002218</td>
      <td>NaN</td>
      <td>2320.549910</td>
      <td>2320.549910</td>
      <td>1687.452308</td>
      <td>1687.452308</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49996</th>
      <td>36.185830</td>
      <td>9.693708</td>
      <td>0.249899</td>
      <td>2894.866377</td>
      <td>NaN</td>
      <td>1980.870768</td>
      <td>1980.870768</td>
      <td>913.995609</td>
      <td>913.995609</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49997</th>
      <td>38.931085</td>
      <td>9.926531</td>
      <td>0.262616</td>
      <td>3114.486770</td>
      <td>NaN</td>
      <td>2067.608947</td>
      <td>2067.608947</td>
      <td>1046.877823</td>
      <td>1046.877823</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49998</th>
      <td>41.673405</td>
      <td>11.669636</td>
      <td>0.242565</td>
      <td>3333.872403</td>
      <td>NaN</td>
      <td>2044.799300</td>
      <td>2044.799300</td>
      <td>1289.073102</td>
      <td>1289.073102</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49999</th>
      <td>40.430469</td>
      <td>12.503527</td>
      <td>0.273984</td>
      <td>3234.437544</td>
      <td>NaN</td>
      <td>2273.064551</td>
      <td>2273.064551</td>
      <td>961.372994</td>
      <td>961.372994</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>50000 rows × 11 columns</p>
</div>

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
# Exemple 
pf_df, beta_df, result_df = calc_pf_beta(serial_df)

# 
print(f'PF:\n{tabulate(pf_df, headers="keys", tablefmt="pretty", showindex=False)}')
print(f'Beta:\n{tabulate(beta_df, headers="keys", tablefmt="pretty", showindex=False)}')
```
```
[0.00212, 0.00212]
PF:
+---------+---------+
|    0    |    1    |
+---------+---------+
| 0.00212 | 0.00212 |
+---------+---------+
Beta:
+---------+---------+
|    0    |    1    |
+---------+---------+
| 0.00212 | 0.00212 |
+---------+---------+
```
