---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 5
has_children: false
has_toc: false
title: pf
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h3>Probability of Failure Equation (PF Equation)</h3>
<br>
<p align = "justify">
    This function calculates the probability of failure (\(P_f\)) for a given reliability index (\(\beta\)) using a standard normal cumulative distribution function. The calculation is performed by integrating the probability density function (PDF) of a standard normal distribution.
</p>

```python
probability_failure = pf_equation(beta)
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
        <td><code>beta</code></td>
        <td>Reliability index, representing the number of standard deviations away from the mean.</td>
        <td>Float</td>
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
       <td><code>probability_failure</code></td>
       <td>Calculated probability of failure.</td>
       <td>Float</td>
   </tr>
</table>

<h4><i>Integration Process</i></h4>
<p align = "justify" id = "integration-process"></p>

MODEL PARAMETERS
{: .label .label-red }

<h6><i>Probability of Failure Calculation</i></h6>

```python
beta = 3.5
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
        <td><code>'beta'</code></td>
        <td>Reliability index for the calculation</td>
        <td>Float</td>
    </tr>
</table>

VARIABLES SETTINGS
{: .label .label-red }

```python
def integrand(x):
    return 1/sqrt(2*np.pi) * np.exp(-x**2/2)

def integral_x(x):
    integral, _ = quad(integrand, 0, x)
    return 1 - (0.5 + integral)
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
        <td><code>'integrand(x)'</code></td>
        <td>Function representing the PDF of a standard normal distribution.</td>
        <td>Function</td>
    </tr>
    <tr>
        <td><code>'integral_x(x)'</code></td>
        <td>Function to integrate the PDF from 0 to the given \(\beta\).</td>
        <td>Function</td>
    </tr>
</table>

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, we use the <code>pf_equation</code> function to calculate the probability of failure for a reliability index of 0.3. The function returns the probability as a floating-point value.</i>
</p>

```python
from parepy_toolbox.common_library import pf_equation

beta = 0.3
pf = pf_equation(beta)
print(pf)
```
```bash
0.3820885778110473
``` 