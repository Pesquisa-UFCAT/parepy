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

### Function Documentation: `generate_factorial_design`

<p align="justify">
    Generates a full factorial design based on the input dictionary of variable levels. The function computes all possible combinations of the provided levels for each variable and returns them in a structured DataFrame.
</p>

```python
df = generate_factorial_design(level_dict)
```

#### Input Variables
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
            A dictionary where keys represent variable names, and values are lists, arrays, or sequences representing the levels of each variable.
        </td>
        <td>Dictionary</td>
    </tr>
</table>

---

#### Output Variables
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
           A dictionary containing all possible combinations of the levels provided in the input dictionary. Each column corresponds to a variable defined in <code>level_dict</code>. And each row represents one combination of the factorial design.
       </td>
       <td>Dictionary</td>
   </tr>
</table>

---

EXAMPLE
{: .label .label-blue }

This example demonstrates how to generate a full factorial design for a given set of levels.

YOUR_PROBLEM.IPYNB
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
print(df)
```

**Example Output**:

```bash
+----+----------+----------+----------+----------+
|    |   i (mm) |   j (mm) |   k (mm) |   l (mm) |
|----+----------+----------+----------+----------|
|  0 |        0 |        0 |        5 |        0 |
|  1 |        0 |        0 |        5 |        9 |
|  2 |        0 |        0 |        5 |       10 |
|  3 |        0 |        0 |       15 |        0 |
|  4 |        0 |        0 |       15 |        9 |
|  5 |        0 |        0 |       15 |       10 |
|  6 |        0 |        5 |        5 |        0 |
|  7 |        0 |        5 |        5 |        9 |
|  8 |        0 |        5 |        5 |       10 |
|  9 |        0 |        5 |       15 |        0 |
| 10 |        0 |        5 |       15 |        9 |
| 11 |        0 |        5 |       15 |       10 |
| 12 |        0 |       10 |        5 |        0 |
| 13 |        0 |       10 |        5 |        9 |
| 14 |        0 |       10 |        5 |       10 |
| 15 |        0 |       10 |       15 |        0 |
| 16 |        0 |       10 |       15 |        9 |
| 17 |        0 |       10 |       15 |       10 |
| 18 |        0 |       15 |        5 |        0 |
| 19 |        0 |       15 |        5 |        9 |
| 20 |        0 |       15 |        5 |       10 |
| 21 |        0 |       15 |       15 |        0 |
...
| 69 |       10 |       15 |       15 |        0 |
| 70 |       10 |       15 |       15 |        9 |
| 71 |       10 |       15 |       15 |       10 |
+----+----------+----------+----------+----------+
```
