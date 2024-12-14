---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 1
has_children: false
has_toc: false
title: pf_equation
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<p align = "justify">
  This function calculates the probability of failure for a given reliability index using a standard normal cumulative distribution function. It performs the calculation by integrating the probability density function (PDF) of a standard normal distribution.
</p>

```python
pf_value = pf_equation(beta)
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
        <td>Reliability index</td>
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
       <td><code>pf_value</code></td>
       <td>Probability of failure</td>
       <td>Float</td>
   </tr>
</table>

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>Use <code>pf_equation</code> function to calculate the probability of failure for a reliability index of 3.5.</i>
</p>

```python
# Library
from parepy_toolbox import pf_equation

# Calc pf
beta = 3.5
pf = pf_equation(beta)
print(f"Probability of failure {pf:.5e}")
``` 

<p align = "justify">
    <i>Output details.</i>
</p>

```bash
Probability of failure 2.32629e-04
``` 