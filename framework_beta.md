---
layout: home
parent: common_library
grand_parent: Framework
nav_order: 4
has_children: false
has_toc: false
title: beta
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->


<h3>Beta Equation</h3>
<br>
<p align = "justify">
    This function calculates the beta value for a given probability of failure (pf). The calculation involves complex mathematical operations and considers specific conditions where the probability of failure is greater than 0.5, resulting in a beta value of "minus infinity".
</p>

```python
beta_value = beta_equation(pf)
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
        <td><code>pf</code></td>
        <td>Probability of failure.</td>
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
       <td><code>beta_value</code></td>
       <td>Calculated beta value, or "minus infinity" if the probability of failure exceeds 0.5.</td>
       <td>Float or String</td>
   </tr>
</table>

<h4><i>Beta Equation Example</i></h4>
<p align = "justify" id = "beta-example"></p>

MODEL PARAMETERS
{: .label .label-red }

<h6><i>Beta Equation for a Given Probability of Failure</i></h6>

```python
pf = 0.3820885778
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
        <td><code>'pf'</code></td>
        <td>Probability of failure used for the beta calculation</td>
        <td>Float</td>
    </tr>
</table>

VARIABLES SETTINGS
{: .label .label-red }

```python
beta_value = beta_equation(pf)
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
        <td><code>'beta_value'</code></td>
        <td>Calculated beta value based on the given probability of failure</td>
        <td>Float or String</td>
    </tr>
</table>

Example 1
{: .label .label-blue }

<p align = "justify">
    <i>In this example, the <code>beta_equation</code> function calculates the beta value for a probability of failure (pf) of 0.3820885778. The result is returned as a floating-point number, representing the beta value associated with this failure probability.</i>
</p>

```python
from parepy_toolbox.common_library import beta_equation

pf = 0.3820885778
beta = beta_equation(pf)
print(beta)
``` 
```bash
0.3000000000289662
``` 