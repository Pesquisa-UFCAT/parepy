<!-- ---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 7
has_children: false
has_toc: false
title: concat
--- -->

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<p align="justify">
    This function concatenates DataFrames generated from `.txt` files in a folder and calculates the probability of failure and the Beta index for each limit state function (constraint) based on the specified model.
</p>

```python
concatenated_df, pf_time, beta_time = concat_sampling_algorithm_structural_analysis(setup)
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
        <td>Dictionary containing the settings for the function.</td>
        <td>Dictionary</td>
    </tr>
    <tr>
        <td><code>setup['model']</code></td>
        <td>The model used for analysis.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>setup['n_constraints']</code></td>
        <td>Number of limit state functions (constraints).</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>setup['folder_path']</code></td>
        <td>Path to the folder containing the files to be concatenated.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>setup['name_simulation']</code></td>
        <td>Name of the simulation to include in the output file name.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>setup['algorithm']</code></td>
        <td>Name of the algorithm used.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>setup['steps']</code></td>
        <td>Number of time steps for analysis, required if the model is 'MCS-TIME'.</td>
        <td>Integer</td>
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
       <td><code>concatenated_df</code></td>
       <td>Concatenated DataFrame containing all the data from the read files.</td>
       <td>DataFrame</td>
   </tr>
   <tr>
       <td><code>pf_time</code></td>
       <td>DataFrame containing the probability of failure calculated for each limit state function.</td>
       <td>DataFrame</td>
   </tr>
   <tr>
       <td><code>beta_time</code></td>
       <td>DataFrame containing the Beta index calculated for each limit state function.</td>
       <td>DataFrame</td>
   </tr>
</table>

<h4><i>Crude Monte Carlo - Time Analysis</i></h4>
<p align="justify" id="mcs-time"></p>

MODEL PARAMETERS
{: .label .label-red }

<h6><i>Crude Monte Carlo Time Analysis</i></h6>

```python
setup = {
    'model': 'MCS-TIME',
    'n_constraints': 3,
    'folder_path': '/path/to/txt/files',
    'name_simulation': 'simulation_name',
    'algorithm': 'MCS-TIME',
    'steps': 5
}
```

<table style="width:100%">
    <thead>
      <tr>
        <th>Name</th>
        <th>Description</th>
        <th>Type</th>
      </tr>
    </thead>
    <tr>
        <td><code>'setup'</code></td>
        <td>Dictionary containing the settings for the function.</td>
        <td>Dictionary</td>
    </tr>
    <tr>
        <td><code>'model'</code></td>
        <td>The model used for analysis.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'n_constraints'</code></td>
        <td>Number of limit state functions (constraints).</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>'folder_path'</code></td>
        <td>Path to the folder containing the files to be concatenated.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'name_simulation'</code></td>
        <td>Name of the simulation to include in the output file name.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'algorithm'</code></td>
        <td>Name of the algorithm used.</td>
        <td>String</td>
    </tr>
    <tr>
        <td><code>'steps'</code></td>
        <td>Number of time steps for analysis, required if the model is 'MCS-TIME'.</td>
        <td>Integer</td>
    </tr>
</table>

Example 1
{: .label .label-blue }

<p align="justify"> <i>
In this example, we use the <code>concat_sampling_algorithm_structural_analysis</code> function to concatenate files and calculate the probability of failure and Beta index for each limit state function using the "MCS" model. We analyze 3 constraints. The results are stored in three DataFrames: the concatenated DataFrame, the probability of failure DataFrame, and the Beta index DataFrame.
</i> </p>

```python
import pandas as pd
import os
from datetime import datetime

# Example setup dictionary
setup = {
    'model': 'MCS',
    'n_constraints': 3,
    'name_simulation': 'example_simulation',
    'algorithm': 'MCS',
    'folder_path': '/path/to/txt/files'
}

# Function call
concatenated_df, pf_time, beta_time = concat_sampling_algorithm_structural_analysis(setup)

# Output details
print(concatenated_df)
print(pf_time)
print(beta_time)
```
```bash
PARE^py Report: 

- Output file name: example_simulation_MCS_20240830-161110.txt
- Processing time (s): 0.07264947891235352
```

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
      <td>41.065376</td>
      <td>9.819842</td>
      <td>0.280362</td>
      <td>3285.230101</td>
      <td>3285.230101</td>
      <td>2165.343850</td>
      <td>2165.343850</td>
      <td>1119.886252</td>
      <td>1119.886252</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.856929</td>
      <td>9.798346</td>
      <td>0.223964</td>
      <td>3588.554293</td>
      <td>3588.554293</td>
      <td>1835.265853</td>
      <td>1835.265853</td>
      <td>1753.288440</td>
      <td>1753.288440</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41.134156</td>
      <td>9.736492</td>
      <td>0.269275</td>
      <td>3290.732458</td>
      <td>3290.732458</td>
      <td>2096.180245</td>
      <td>2096.180245</td>
      <td>1194.552213</td>
      <td>1194.552213</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.993198</td>
      <td>10.806235</td>
      <td>0.255234</td>
      <td>2879.455859</td>
      <td>2879.455859</td>
      <td>2072.062759</td>
      <td>2072.062759</td>
      <td>807.393100</td>
      <td>807.393100</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36.692144</td>
      <td>11.566328</td>
      <td>0.255177</td>
      <td>2935.371505</td>
      <td>2935.371505</td>
      <td>2112.771093</td>
      <td>2112.771093</td>
      <td>822.600412</td>
      <td>822.600412</td>
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
      <th>2995</th>
      <td>51.616671</td>
      <td>12.606109</td>
      <td>0.315159</td>
      <td>4129.333684</td>
      <td>4129.333684</td>
      <td>2518.734708</td>
      <td>2518.734708</td>
      <td>1610.598976</td>
      <td>1610.598976</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2996</th>
      <td>35.130283</td>
      <td>10.292136</td>
      <td>0.231028</td>
      <td>2810.422635</td>
      <td>2810.422635</td>
      <td>1903.130386</td>
      <td>1903.130386</td>
      <td>907.292250</td>
      <td>907.292250</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2997</th>
      <td>41.845206</td>
      <td>10.058740</td>
      <td>0.239903</td>
      <td>3347.616476</td>
      <td>3347.616476</td>
      <td>1942.286697</td>
      <td>1942.286697</td>
      <td>1405.329779</td>
      <td>1405.329779</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2998</th>
      <td>45.761574</td>
      <td>11.662165</td>
      <td>0.256711</td>
      <td>3660.925900</td>
      <td>3660.925900</td>
      <td>2126.895243</td>
      <td>2126.895243</td>
      <td>1534.030657</td>
      <td>1534.030657</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2999</th>
      <td>41.700261</td>
      <td>13.753616</td>
      <td>0.279727</td>
      <td>3336.020843</td>
      <td>3336.020843</td>
      <td>2374.060529</td>
      <td>2374.060529</td>
      <td>961.960314</td>
      <td>961.960314</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3000 rows × 11 columns</p>
</div>


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
      <th>I_0</th>
      <th>I_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001667</td>
      <td>0.001667</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>I_0</th>
      <th>I_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.935199</td>
      <td>2.935199</td>
    </tr>
  </tbody>
</table>
</div>

