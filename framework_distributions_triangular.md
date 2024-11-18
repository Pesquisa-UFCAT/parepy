---
layout: home
parent: distributions
grand_parent: Framework
nav_order: 7
has_children: true
has_toc: true
title: triangular_sampling
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h3>Triangular Sampling</h3>

<p align="justify">
    This function generates random samples from a triangular distribution, defined by its minimum (<code>a</code>), mode (<code>c</code>), and maximum (<code>b</code>) values. The distribution is used to model scenarios where the exact shape of the distribution is unknown, but estimates for its bounds and most likely value are available.
</p>

```python
u = triangular_sampling(parameters, method, n_samples, seed)
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
            Dictionary of parameters for the triangular distribution. Keys:
            <ul>
                <li><code>'min'</code>: Minimum value of the distribution [Float]</li>
                <li><code>'mode'</code>: Mode (most likely value) of the distribution [Float]</li>
                <li><code>'max'</code>: Maximum value of the distribution [Float]</li>
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
       <td>Generated random samples from a triangular distribution</td>
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
            <th>Syntax and Description</th>
            <th>Example</th>
        </tr>
        </thead>
        <tr>
            <td>Crude Monte Carlo Sampling</td>
            <td>
                <ul>
                    <li>Generates random samples uniformly distributed between 0 and 1</li>
                    <li>Transforms the uniform samples into triangular samples using the inverse CDF</li>
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
                    <li>Transforms the uniform samples into triangular samples using the inverse CDF</li>
                    <li>Uses <code>lhs_sampling_zero_one</code> function</li>
                </ul>
            </td>
            <td><code>method = 'lhs'</code></td>
        </tr>
    </table>
</center>
<p align="justify">
    The triangular distribution is defined by three parameters:
    <ul>
        <li><code>a</code>: Minimum value</li>
        <li><code>c</code>: Mode (most likely value)</li>
        <li><code>b</code>: Maximum value</li>
    </ul>
</p>
<p align="justify">
    The inverse CDF method is used to generate samples from the triangular distribution:
    <ul>
        <li><code>criteria = (c - a) / (b - a)</code>: Determines the split point in the probability distribution.</li>
        <li>If <code>u_aux[i] &lt; criteria</code>: The sample is calculated as <code>a + np.sqrt(u_aux[i] * (b - a) * (c - a))</code>.</li>
        <li>If <code>u_aux[i] &ge; criteria</code>: The sample is calculated as <code>b - np.sqrt((1 - u_aux[i]) * (b - a) * (b - c))</code>.</li>
    </ul>
</p>

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, we will use the <code>sampling</code> function to generate a set of random samples \((n=1,000,000)\) following a Normal distribution with mean \(\mu = 7\) and standard deviation \(\sigma = 3\). Use "seed without control" in your setup.</i>
</p>

<center>
    <img src="assets/images/sampling_figure_3.svg" width="50%" height="auto">
    <p align="center"><b>Figure 3.</b> Random variable example.</p>
</center>