---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 2
has_children: false
has_toc: false
title: fbf
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h3>fbf</h3>
<br>
<p align = "justify">
    This function processes the provided data based on the specified algorithm, modifying the results according to the imposed conditions.
</p>

```python
results_about_data = fbf(algorithm, n_constraints, time_analysis, results_about_data)
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
        <td><code>algorithm</code></td>
        <td>Name of the algorithm used.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>n_constraints</code></td>
        <td>Number of constraints analyzed.</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>time_analysis</code></td>
        <td>Time period for analysis.</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>results_about_data</code></td>
        <td>DataFrame containing the results to be processed.</td>
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
       <td><code>results_about_data</code></td>
       <td>Updated DataFrame after processing.</td>
       <td>DataFrame</td>
   </tr>
</table>

<h4><i>Crude Monte Carlo - Time Analysis</i></h4>
<p align = "justify" id = "mcs-time"></p>

MODEL PARAMETERS
{: .label .label-red }

<h6><i>Crude Monte Carlo Time Analysis</i></h6>

```python
algorithm = 'MCS-TIME'
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
        <td><code>'algorithm'</code></td>
        <td>Algorithm used for processing</td>
        <td>String</td>
    </tr>
</table>

VARIABLES SETTINGS
{: .label .label-red }

```python
n_constraints = 3
time_analysis = 5
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
        <td><code>'n_constraints'</code></td>
        <td>Number of constraints to analyze</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>'time_analysis'</code></td>
        <td>Time period for analysis</td>
        <td>Integer</td>
    </tr>
</table>

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, we use the <code>fbf</code> function to process a DataFrame based on the "MCS-TIME" algorithm, analyzing 3 constraints over 5 time periods. The DataFrame is updated accordingly.</i>
</p>

```python
import pandas as pd
import numpy as np

# Example DataFrame setup
data = {
    'I_0_t=0': [1, 0, 0, 0, 0, 0, 1, 0],
    'I_0_t=1': [1, 1, 1, 1, 1, 1, 1, 1],
    'I_0_t=2': [0, 0, 0, 0, 0, 0, 1, 0],
    'I_1_t=0': [1, 0, 1, 0, 0, 0, 0, 0],
    'I_1_t=1': [0, 0, 1, 1, 0, 0, 0, 0],
    'I_1_t=2': [1, 1, 0, 0, 0, 0, 0, 0]
}
df = pd.DataFrame(data)

# Function call
processed_df = fbf('MCS-TIME', 3, 5, df)

# Output details
print(processed_df)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>I_0_t=0</th>
      <th>I_0_t=1</th>
      <th>I_0_t=2</th>
      <th>I_1_t=0</th>
      <th>I_1_t=1</th>
      <th>I_1_t=2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>