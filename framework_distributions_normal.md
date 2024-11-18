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
    This function generates random samples from a normal distribution with a specified mean (<code>mu</code>) and standard deviation (<code>sigma</code>), using the specified sampling method.
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
                <li><code>'mean'</code>: Mean of the normal distribution [Float]</li>
                <li><code>'sigma'</code>: Standard deviation of the normal distribution [Float]</li>
            </ul>
            </p>
        </td>
        <td>Dictionary</td>
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
        <td>String</td>
    </tr>
    <tr>
        <td><code>n_samples</code></td>
        <td>Number of samples to generate</td>
        <td>Integer</td>
    </tr>
    <tr>
        <td><code>seed</code></td>
        <td>Seed for random number generation. Use <code>None</code> for a random seed</td>
        <td>Integer or None</td>
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
       <td>Generated random samples from a normal distribution</td>
       <td>List</td>
   </tr>
</table>

<p align="justify" id="methods"></p>
<center>
    <p align="center"><b>Table 1.</b> Sampling methods (<code>method</code> key).</p>
    <table style="width:100%">
        <thead>
        <tr>
            <th>Method</th>
            <th>Sintax and description</th>
            <th>Example</th>
        </tr>
        </thead>
        <tr>
            <td>Crude Monte Carlo Sampling</td>
            <td>
                <ul>
                    <li>Generates random samples uniformly distributed between 0 and 1</li>
                    <li>Uses the Box-Muller transform to convert uniform samples into normal samples</li>
                    <li>Uses <code>crude_sampling_zero_one</code> function</li>
                </ul>
            </td>
            <td><code>method = 'mcs'</code></td>
        </tr>
        <tr>
            <td>Latin Hypercube Sampling</td>
            <td>
                <ul>
                    <li>Divides the domain into equal intervals and samples randomly within each interval</li>
                    <li>Uses the Box-Muller transform to convert uniform samples into normal samples</li>
                    <li>Uses <code>lhs_sampling_zero_one</code> function</li>
                </ul>
            </td>
            <td><code>method = 'lhs'</code></td>
        </tr>
    </table>
</center>

Example 1
{: .label .label-blue }

<p align="justify">
    <i>In this example, we use the <code>normal_sampling</code> function to generate two sets of random samples \((n=400)\) following a Normal distribution with a mean \(\mu = 10\) and a standard deviation \(\sigma = 2\). The first set is sampled using the Monte Carlo Sampling (MCS) method, while the second set is generated using the Latin Hypercube Sampling (LHS) method. Both sets are visualized using histograms with Kernel Density Estimates (KDE) plotted side-by-side for comparison. The plots help to illustrate how the sampling methods influence the distribution of the generated data.</i>
</p>

```python
# Sampling
n = 400
x = normal_sampling({'mean': 10, 'sigma': 2}, 'mcs', n)
y = normal_sampling({'mean': 10, 'sigma': 2}, 'lhs', n)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# First plot: Histogram and KDE for data1
sns.histplot(x, kde=True, bins=30, color='blue', ax=axes[0], alpha=0.6, edgecolor='black')
axes[0].set_title('MCS Sampling')
axes[0].set_xlabel('Valores')
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
    <img src="assets/images/sampling_figure_3.svg" width="50%" height="auto">
    <p align="center"><b>Figure 3.</b> Random variable example.</p>
</center>