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
from tabulate import tabulate

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
print(tabulate(df, headers='keys', tablefmt='psql'))
```

**Example Output**:

```bash
+-----+----------+----------+----------+----------+
|     |   i (mm) |   j (mm) |   k (mm) |   l (mm) |
|-----+----------+----------+----------+----------|
|   0 |        0 |        0 |        5 |        0 |
|   1 |        0 |        0 |        5 |        9 |
|   2 |        0 |        0 |        5 |       10 |
|   3 |        0 |        0 |        5 |       11 |
|   4 |        0 |        0 |        5 |       12 |
|   5 |        0 |        0 |       15 |        0 |
|   6 |        0 |        0 |       15 |        9 |
|   7 |        0 |        0 |       15 |       10 |
|   8 |        0 |        0 |       15 |       11 |
|   9 |        0 |        0 |       15 |       12 |
|  10 |        0 |        5 |        5 |        0 |
|  11 |        0 |        5 |        5 |        9 |
|  12 |        0 |        5 |        5 |       10 |
|  13 |        0 |        5 |        5 |       11 |
|  14 |        0 |        5 |        5 |       12 |
|  15 |        0 |        5 |       15 |        0 |
|  16 |        0 |        5 |       15 |        9 |
|  17 |        0 |        5 |       15 |       10 |
|  18 |        0 |        5 |       15 |       11 |
|  19 |        0 |        5 |       15 |       12 |
|  20 |        0 |       10 |        5 |        0 |
|  21 |        0 |       10 |        5 |        9 |
|  22 |        0 |       10 |        5 |       10 |
|  23 |        0 |       10 |        5 |       11 |
|  24 |        0 |       10 |        5 |       12 |
|  25 |        0 |       10 |       15 |        0 |
|  26 |        0 |       10 |       15 |        9 |
|  27 |        0 |       10 |       15 |       10 |
|  28 |        0 |       10 |       15 |       11 |
|  29 |        0 |       10 |       15 |       12 |
|  30 |        0 |       15 |        5 |        0 |
|  31 |        0 |       15 |        5 |        9 |
|  32 |        0 |       15 |        5 |       10 |
|  33 |        0 |       15 |        5 |       11 |
|  34 |        0 |       15 |        5 |       12 |
|  35 |        0 |       15 |       15 |        0 |
|  36 |        0 |       15 |       15 |        9 |
|  37 |        0 |       15 |       15 |       10 |
|  38 |        0 |       15 |       15 |       11 |
|  39 |        0 |       15 |       15 |       12 |
|  40 |        5 |        0 |        5 |        0 |
|  41 |        5 |        0 |        5 |        9 |
|  42 |        5 |        0 |        5 |       10 |
|  43 |        5 |        0 |        5 |       11 |
|  44 |        5 |        0 |        5 |       12 |
|  45 |        5 |        0 |       15 |        0 |
|  46 |        5 |        0 |       15 |        9 |
|  47 |        5 |        0 |       15 |       10 |
|  48 |        5 |        0 |       15 |       11 |
|  49 |        5 |        0 |       15 |       12 |
|  50 |        5 |        5 |        5 |        0 |
|  51 |        5 |        5 |        5 |        9 |
|  52 |        5 |        5 |        5 |       10 |
|  53 |        5 |        5 |        5 |       11 |
|  54 |        5 |        5 |        5 |       12 |
|  55 |        5 |        5 |       15 |        0 |
|  56 |        5 |        5 |       15 |        9 |
|  57 |        5 |        5 |       15 |       10 |
|  58 |        5 |        5 |       15 |       11 |
|  59 |        5 |        5 |       15 |       12 |
|  60 |        5 |       10 |        5 |        0 |
|  61 |        5 |       10 |        5 |        9 |
|  62 |        5 |       10 |        5 |       10 |
|  63 |        5 |       10 |        5 |       11 |
|  64 |        5 |       10 |        5 |       12 |
|  65 |        5 |       10 |       15 |        0 |
|  66 |        5 |       10 |       15 |        9 |
|  67 |        5 |       10 |       15 |       10 |
|  68 |        5 |       10 |       15 |       11 |
|  69 |        5 |       10 |       15 |       12 |
|  70 |        5 |       15 |        5 |        0 |
|  71 |        5 |       15 |        5 |        9 |
|  72 |        5 |       15 |        5 |       10 |
|  73 |        5 |       15 |        5 |       11 |
|  74 |        5 |       15 |        5 |       12 |
|  75 |        5 |       15 |       15 |        0 |
|  76 |        5 |       15 |       15 |        9 |
|  77 |        5 |       15 |       15 |       10 |
|  78 |        5 |       15 |       15 |       11 |
|  79 |        5 |       15 |       15 |       12 |
|  80 |       10 |        0 |        5 |        0 |
|  81 |       10 |        0 |        5 |        9 |
|  82 |       10 |        0 |        5 |       10 |
|  83 |       10 |        0 |        5 |       11 |
|  84 |       10 |        0 |        5 |       12 |
|  85 |       10 |        0 |       15 |        0 |
|  86 |       10 |        0 |       15 |        9 |
|  87 |       10 |        0 |       15 |       10 |
|  88 |       10 |        0 |       15 |       11 |
|  89 |       10 |        0 |       15 |       12 |
|  90 |       10 |        5 |        5 |        0 |
|  91 |       10 |        5 |        5 |        9 |
|  92 |       10 |        5 |        5 |       10 |
|  93 |       10 |        5 |        5 |       11 |
|  94 |       10 |        5 |        5 |       12 |
|  95 |       10 |        5 |       15 |        0 |
|  96 |       10 |        5 |       15 |        9 |
|  97 |       10 |        5 |       15 |       10 |
|  98 |       10 |        5 |       15 |       11 |
|  99 |       10 |        5 |       15 |       12 |
| 100 |       10 |       10 |        5 |        0 |
| 101 |       10 |       10 |        5 |        9 |
| 102 |       10 |       10 |        5 |       10 |
| 103 |       10 |       10 |        5 |       11 |
| 104 |       10 |       10 |        5 |       12 |
| 105 |       10 |       10 |       15 |        0 |
| 106 |       10 |       10 |       15 |        9 |
| 107 |       10 |       10 |       15 |       10 |
| 108 |       10 |       10 |       15 |       11 |
| 109 |       10 |       10 |       15 |       12 |
| 110 |       10 |       15 |        5 |        0 |
| 111 |       10 |       15 |        5 |        9 |
| 112 |       10 |       15 |        5 |       10 |
| 113 |       10 |       15 |        5 |       11 |
| 114 |       10 |       15 |        5 |       12 |
| 115 |       10 |       15 |       15 |        0 |
| 116 |       10 |       15 |       15 |        9 |
| 117 |       10 |       15 |       15 |       10 |
| 118 |       10 |       15 |       15 |       11 |
| 119 |       10 |       15 |       15 |       12 |
+-----+----------+----------+----------+----------+
```
