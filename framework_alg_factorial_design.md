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
import itertools 
import numpy as np
import pandas as pd
from tabulate import tabulate

# Function Definition
def generate_factorial_design(level_dict):
    combinations = list(itertools.product(*level_dict.values()))
    df = pd.DataFrame(combinations, columns=level_dict.keys())
    return df

# Input Levels
setup = {
    'i (mm)': np.linspace(0, 10, 3),
    'j (mm)': np.linspace(0, 15, 4),
    'k (mm)': [5, 15],               
    'l (mm)': np.linspace(0, 20, 5),
}

# Generate Factorial Design
df = generate_factorial_design(setup)

# Print Results
print(tabulate(df, headers='keys', tablefmt='psql'))
```

**Example Output**:

```bash
```