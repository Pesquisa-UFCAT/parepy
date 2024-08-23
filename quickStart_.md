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

More details will be shown in other sections!
</p>

