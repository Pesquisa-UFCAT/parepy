---
layout: home
parent: distributions
grand_parent: Framework
nav_order: 2
has_children: true
has_toc: true
title: normal_sampling
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h3>Normal Sampling</h3>
<p align="justify">
    This function generates random samples from a Normal distribution with a specified mean \(\mu\) and standard deviation \(\sigma\).
</p>

```python
u = normal_sampling(parameters, method, n_samples, seed)
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
        <td><code>parameters</code></td>
        <td>
            <p align="justify">
            Dictionary of parameters for the normal distribution. Keys:
            <ul>
                <li><code>'mean'</code>: Mean [float]</li>
                <li><code>'sigma'</code>: Standard deviation [float]</li>
            </ul>
            </p>
        </td>
        <td>dictionary</td>
    </tr>
    <tr>
        <td><code>method</code></td>
        <td>
            <p align="justify">Sampling method. Supports the following values:
            <ul>
                <li><code>'mcs'</code>: Crude Monte Carlo Sampling</li>
                <li><code>'lhs'</code>: Latin Hypercube Sampling</li>
            </ul>
            </p>
        </td>
        <td>string</td>
    </tr>
    <tr>
        <td><code>n_samples</code></td>
        <td>Number of samples</td>
        <td>integer</td>
    </tr>
    <tr>
        <td><code>seed</code></td>
        <td>Seed for random number generation. Use <code>None</code> for a random seed</td>
        <td>integer or None</td>
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
       <td><code>u</code></td>
       <td>Random samples</td>
       <td>list</td>
   </tr>
</table>

Example 1
{: .label .label-blue }

<p align="justify">
    <i>
        In this example, we will use the <code>normal_sampling</code> function from the <code>parepy_toolbox</code> to generate two sets of random samples (\(n=400\)) following a normal distribution. The first set is sampled using the Monte Carlo Sampling (MCS) method, and the second using the Latin Hypercube Sampling (LHS) method. Mean and standard deviation is defined as \([10, 2]\). The results are visualized using histograms with Kernel Density Estimates (KDE) plotted (using matplotlib lib) side-by-side for comparison.
    </i>
</p>

```python
# Library
import matplotlib.pyplot as plt

from parepy_toolbox import normal_sampling

# Sampling
n = 400
x = normal_sampling({'mean': 10, 'sigma': 2}, 'mcs', n)
y = normal_sampling({'mean': 10, 'sigma': 2}, 'lhs', n)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# First plot: Histogram and KDE for data1
sns.histplot(x, kde=True, bins=30, color='blue', ax=axes[0], alpha=0.6, edgecolor='black')
axes[0].set_title('MCS Sampling')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Densidade')

# Second plot: Histogram and KDE for data2
sns.histplot(y, kde=True, bins=30, color='green', ax=axes[1], alpha=0.6, edgecolor='black')
axes[1].set_title('LHS Sampling')
axes[1].set_xlabel('Valores')
axes[1].set_ylabel('Densidade')

# Ajust and show plot
plt.tight_layout()
plt.show()
```

<center>
    <img src="assets/images/normal_sampling_figure_1.png" height="auto">
    <p align="center"><b>Figure 1.</b> Uniform variable example.</p>
</center>

Example 2
{: .label .label-blue }

<p align="justify">
    <i>
    In this example, we will use the <code>normal_sampling</code> function from the <code>parepy_toolbox</code> to generate two sets of random samples (\(n=3\)) following a normal distribution. Using the Monte Carlo algorithm and the specific seed (<code>seed=25</code>), we uniformly sampling generate 3 times and compare results.
    </i>
</p>

```python
from parepy_toolbox import normal_sampling

# Sampling
n = 3
x0 = normal_sampling({'mean': 10, 'sigma': 2}, 'mcs', n, 25)
x1 = normal_sampling({'mean': 10, 'sigma': 2}, 'mcs', n, 25)
x2 = normal_sampling({'mean': 10, 'sigma': 2}, 'mcs', n, 25)
print(x0, '\n', x1, '\n', x2)
```

```bash
[11.607212332320078, 10.003120351710036, 12.16598464462817] 
[11.607212332320078, 10.003120351710036, 12.16598464462817] 
[11.607212332320078, 10.003120351710036, 12.16598464462817]
```

{: .important }
> Note that using the seed 25 by 3 times, we can generate the same values in a random variable.
