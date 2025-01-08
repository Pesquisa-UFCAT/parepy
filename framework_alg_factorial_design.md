---
layout: home
parent: algorithms
grand_parent: Framework
nav_order: 5
has_children: false
has_toc: false
title: generate_factorial_design
---

<!--Don't delete this script-->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h3>generate_factorial_design</h3>

<p align="justify">
  Generates a complete factorial design based on the input dictionary of variable levels. The function computes all possible combinations of the provided levels for each variable and returns them in a structured data frame.
</p>

```python
df = generate_factorial_design(level_dict)
```

Input Variables
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
        <td><code>level_dict</code></td>
        <td>
            A dictionary where keys represent variable names, and values are lists, arrays, or sequences representing the levels of each variable
        </td>
        <td>Dictionary</td>
    </tr>
</table>

---

Output Variables
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
       <td><code>df</code></td>
       <td>
           A dictionary containing all possible combinations of the levels provided in the input dictionary
       </td>
       <td>Dictionary</td>
   </tr>
</table>

---

Example 1
{: .label .label-blue }

<p align="justify">
  <i>
    This example demonstrates how to generate a full factorial design for a given set of levels. Consider four variables \(i\),\(j\),\(k\) and \(l\) to assemble full factorial design. The \(i\) variable have a range \([0, 10]\) with 3 levels, the \(j\) variable have a range \([0, 15]\) with 4 levels, The \(k\) variable have a level \([5, 15]\) and the \(l\) variable have a levels \([0, 9, 10, 11, 12]\).
  </i>
</p>

your_problem.ipynb
{: .label .label-red }

```python
import numpy as np
from parepy_toolbox import generate_factorial_design

# Input Levels
setup = {
    'i (mm)': np.linspace(0, 10, 3),
    'j (mm)': np.linspace(0, 15, 4),
    'k (mm)': [5, 15],               
    'l (mm)': [0, 9, 10, 11, 12],
}

# Generate Factorial Design
df = generate_factorial_design(setup)

# Print Results
df
```

<p align="justify">
  Output details:
</p>

```bash
+----+----------+----------+----------+----------+
|    |   i (mm) |   j (mm) |   k (mm) |   l (mm) |
|----+----------+----------+----------+----------|
|  0 |        0 |        0 |        5 |        0 |
|  1 |        0 |        0 |        5 |        9 |
|  2 |        0 |        0 |        5 |       10 |
|  3 |        0 |        0 |       15 |        0 |
|  4 |        0 |        0 |       15 |        9 |
...
| 69 |       10 |       15 |       15 |        0 |
| 70 |       10 |       15 |       15 |        9 |
| 71 |       10 |       15 |       15 |       10 |
+----+----------+----------+----------+----------+
```
