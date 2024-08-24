---
layout: home
nav_order: 2
has_children: true
has_toc: true
title: Quick Start
---

<!--Don't delete this script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h1><b>REQUIREMENTS AND INSTALL</b></h1>

<p align="justify">To use the framework in an <b>Python</b> environment, use the following command:</p>

```python
pip install parepy-toolbox
# or pip install --upgrade parepy-toolbox
```

<h1><b>FILES STRUCTURE</b></h1>

<p align="justify">Let's use the example of building a problem in PAREpy using Jupyter Notebook or <b>Python</b> file. Therefore, the basic file structure that you must assemble to use the library must be as follows:</p>

```bash
 .
 .
 .
 └── problem_directory
       └── of_file.py
       └── your_problem.ipynb # or your_problem.py
       └── file 0
       └── file 1
       └── file 2
       ...
       └── file n-1
       └── file n
```

<p align="justify">The <code>of_file.py</code> file should contain the objective function of the problem. The <code>your_problem</code> file is the file that will contain the call to the main function and other settings necessary for the use of the algorithm.
</p>

<h2><code>of_file.py</code></h2>

<p align="justify">
<code>of_file.py</code> is a Python function, and the user needs to define it for PAREpy to work. <code>of_file.py</code> has a fixed structure that must be respected, as described below:
</p>

```python
def my_function(x, none_variable):
    # put your block of code
    return r, s, g
```

<p align="justify">
<code>of_file.py</code> has two parameters: 
</p>

<ol>
      <li><code>x</code> (type list): list of design random variables. PAREpy generates this values;</li>
      <li><code>none_variable</code> (type None, list, float, dictionary, str or any): The user can define this variable. The user can input any value in this variable when calling the framework's main function.</li>
</ol>

<p align="justify">
<code>of_file.py</code> has three returns: 
</p>

<ol>
      <li><code>r</code> (type list): list of values. In structural problems, we recommend putting the capacity in this variable;</li>
      <li><code>s</code> (type list): list of values. In structural problems, we recommend putting the demand in this variable;</li>
      <li><code>g</code> (type list): State limit function \(\mathbf{G} = \mathbf{R} - \mathbf{S}\).</li>
</ol>

<p align="justify">
To demostrate how to create a object function we use Beck <a href="#ref1">[1]</a> example. The State Limit Function is given by:
</p>

<table style = "width:100%">
    <tr>
        <td style="width: 90%;">\[ \mathbf{G} = \mathbf{R}_d - \mathbf{D} - \mathbf{L} \]</td>
        <td style="width: 10%;"><p align = "right" id = "eq1">(1)</p></td>
    </tr>
</table>

```python
def example_function(x, none_variable):
    """Beck example"""

    # random variables statement  
    r_d = x[0]
    d = x[1]
    l = x[2]

    # state limite function
    r = r_d
    s = d + l
    g = r - s

    return [r], [s], [g]

# or

def example_function(x, none_variable):
    """Beck example"""

    # random variables statement  
    r_d = x[0]
    d = x[1]
    l = x[2]

    # state limite function
    g = r_d - d - l

    return [r_d], [d+l], [g]

# or

def example_function(x, none_variable):
    """Beck example"""

    # random variables statement  
    r_d = x[0]
    d = x[1]
    l = x[2]

    # state limite function
    r = [r_d]
    s = [d + l]
    g = [r - s]

    return r, s, g
```

<p align="justify">
Using two state limit functions:
</p>

<table style = "width:100%">
    <tr>
        <td style="width: 90%;">\[ \mathbf{G}_0 = \mathbf{R}_d - \mathbf{D} - \mathbf{L} \]</td>
        <td style="width: 10%;"><p align = "right" id = "eq1">(1)</p></td>
    </tr>
    <tr>
        <td style="width: 90%;">\[ \mathbf{G}_1 = \mathbf{\sigma _y} \cdot \mathbf{W} - \mathbf{M} \]</td>
        <td style="width: 10%;"><p align = "right" id = "eq2">(2)</p></td>
    </tr>
</table>

```python
def example_function(x, none_variable):
    """Beck example"""

    # random variables statement g_0
    r_d = x[0]
    d = x[1]
    l = x[2]

    # random variables statement g_1
    sigma_y = x[0]
    w = x[1]
    m = x[2]

    # state limite function g_0
    r_0 = r_d
    s_0 = d + l
    g_0 = r_0 - s_0

    # state limite function g_1
    r_1 = sigma_y * w
    s_1 = m
    g_1 = r_1 - s_1


    return [r_0, r_1], [s_0, s_1], [g_0, g_1]
```

<h1><b>REFERENCE LIST</b></h1>

<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Reference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><p align = "center" id = "ref1">[1]</p></td>
            <td><p align = "left"><a href="https://doi.org/10.1007/s00521-016-2328-2" target="_blank" rel="noopener noreferrer">Beck AT. Confiabilidade e segurança das estruturas. Elsevier; 2019. ISBN 978-85-352-8895-7</a></p></td>
        </tr>
    </tbody>
</table>