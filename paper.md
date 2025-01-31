---
title: 'PAREpy: A Probabilistic Approach to Reliability Engineering in Python'
tags:
  - Python
  - Reliability
  - Monte Carlo
  - Sampling
  - Structures
authors:
  - name: Prof. PhD Wanderlei Malaquias Pereira Junior
    equal-contrib: true
    affiliation: 1

  - name: Msc Murilo Carneiro Rodrigues
    equal-contrib: true
    affiliation: 1

  - name: Msc Matheus Henrique Morato Moraes
    equal-contrib: true
    affiliation: 3

  - name: Prof. PhD Daniel de Lima Araújo
    equal-contrib: true 
    affiliation: 1

  - name: Prof. PhD André Teófilo Beck
    equal-contrib: true
    affiliation: 2

  - name: Prof. PhD André Luis Christoforo
    equal-contrib: true
    affiliation: 3

  - name: Prof. PhD Iuri Fazolin
    equal-contrib: true
    affiliation: 1

  - name: Prof. PhD Marcos Luiz Henrique
    equal-contrib: true
    affiliation: 4

  - name: Prof. Marcos Napoleão Rabelo
    equal-contrib: true
    affiliation: 1

  - name: Prof. PhD Fran Sergio Lobato
    equal-contrib: true
    affiliation: 5

  - name: Disc. Luiz Henrique Ferreira Rezio
    equal-contrib: true
    affiliation: 1

affiliations:
  - name: Engineering College, Federal University of Catalão, Brazil.
    index: 1
  - name: Department of Structures, São Carlos School of Engineering, Brazil.
    index: 2
  - name: Federal University of São Carlos (UFSCar), Brazil.
    index: 3
  - name: University Federal of Pernambuco, Campus of the Agreste, Brazil.
    index: 4
  - name: School of Chemical Engineering, Federal University of Uberlândia, Center for Exact Sciences and Technology, Brazil.
    index: 5
  - name: Polytechnic Institute, State University of Rio de Janeiro, Brazil.
    index: 6
date: 10 January 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Introduction

The PAREpy (Probabilistic Approach to Reliability Engineering) framework is a library for applying probabilistic concepts to analyze a system containing random variables. The platform is built in Python and can be used in any environment that supports this programming language.  
  
Tools often offer a certain complexity when building reliability analyses. This framework intends to be a faster library for building reliability problems. This version, it is able to assemble structural reliability problems using sampling methods and derivative methods.  
  
The study of structural reliability is concerned with the calculation and prediction of the probability of limit state violation for an engineered structural system at any stage  during its life. In particular, the study of structural safety is concerned with the violation of the ultimate or safety limit states for the structure. More generally, the study of  structural reliability is concerned with the violation of performance measures [@melchers_structural_2018].  
  
PAREPy offers the following functions:
- `pf_equation`: This function calculates the probability of failure ($p_f$) for a given reliability index ($\beta$) using a standard normal cumulative distribution function. The calculation is performed by integrating the probability density function (PDF) of a standard normal distribution.
- `beta_equation`: This function calculates the reliability index value for a given probability of failure ($p_f$).
- `calc_pf_beta_sampling`: Calculates the values of probability of failure or reliability index from the columns of a DataFrame that start with `I_` (Indicator function). If a `.txt` file path is passed, this function evaluates $p_f$ and $\beta$ values too. You can used this function when you have a dataset with sampling results.
- `convergence_probability_failure`: This function calculates the convergence rate of a given column in a data frame. This function is used to check the convergence of the probability of failure or reliability index.
- `sampling_algorithm_structural_analysis`: This function creates the samples and evaluates the limit state functions in structural reliability problems.
- `concatenates_txt_files_sampling_algorithm_structural_analysis`
  
The documentation is available on the [PAREpy web site](https://wmpjrufg.github.io/PAREPY/). There, users can find some examples, learning and see application examples.  
  
# Reliability Overview

Various factors influence the behavior of structures, most of which cannot be fully controlled. The different sources of uncertainty affecting these factors result in variability, making structural safety an inherently non-deterministic problem.  

In most cases, engineers need to check a structure's structural safety. Structural safety can be defined as an attribute that allows us to classify structures as safe or unsafe. A structure is safe when its ability (primarily concerning the possibility of causing harm to people) is higher than a previously accepted value as a minimum admissible and unsafe otherwise [@jacinto_segurancestrutural_2023].  

To check Structural safety, we should introduce the state limit function concept. Through this function, methods like FORM (First Order Reliability Method), SORM (Second Order Reliability Method), and Sampling methods can evaluate a quantitative measurement of structural safety. 

If a structure or part of a structure exceeding a specific limit cannot perform a required performance, this particular limit is called a limit state. So, the function that defines this limit can be called the state limit function [@grandhi_structural_1999].

The basic reliability problem is represented by the multiple integral of equation (1), where $p_f$ represents the failure probability of the structure, $\boldsymbol{X}$ is the n-dimensional vector representing the random variables of the system, $f_x(\boldsymbol{x})$ represents the joint probability density function over the failure domain, and $G(\boldsymbol{X})$ is the limit state equation. $G(\boldsymbol{X}) \leq 0$ represents a failure condition. 

$$
p_f = P(G(\boldsymbol{X}) \leq 0) = \int \dots \int\limits_{G(\boldsymbol{X}) \leq 0} f_{\mathbf{X}}(\boldsymbol{x}) \, d\mathbf{x} \tag{1}
$$

# Quick Start

In this paper, we present the stable version 1.3.3 about PAREPy. Sampling was implemented in this version: (a) Monte Carlo Method and (b) Latin Hyper Cube Method.

## Install

To use the framework in a Python environment, use the following command:

```python
pip install parepy-toolbox
# or pip install --upgrade parepy-toolbox
```

## Files Structure

Let's use the example of building a problem in PAREpy using Jupyter Notebook file or a Python file. The basic file structure that you must assemble to use the library must be as follows:

```bash
 .
 .
 .
 └── problem_directory
       ├── of_file.py
       ├── your_problem.ipynb # or your_problem.py
       ├── file 0
       ├── file 1
       ├── file 2
       ...
       ├── file n-1
       └── file n
```

The `of_file.py` file should contain the objective function (state limit function). The `your_problem.ipynb` file contain the call the main algorithm.

### `of_file.py`

`of_file.py` is a Python function, and the user needs to define it for PAREpy to work. `of_file.py` has a fixed structure that must be respected, as described below:

```python
def my_function(x, none_variable):
    # add your code
    return r, s, g
```

### Parameters of `of_file.py`

- **`x`** (type `list`): A list of design random variables $\left(\boldsymbol{X}\right)$. PAREpy generates these values.
- **`none_variable`** (type `None`, `list`, `float`, `dict`, `str`, or `any`): The user can define this variable. The user can input any value into this variable when calling the framework's main function.

### Returns of `of_file.py`

- `r` (type `list`): A list of values. In structural problems, we recommend putting the capacity in this variable.
- `s` (type `list`): A list of values. In structural problems, we recommend putting the demand in this variable.
- `g` (type `list`): State limit function $\mathbf{G}(\boldsymbol{X}) = \boldsymbol{R} - \boldsymbol{S}$.

> **Important**  
> The lists `r`, `s`, and `g` must have the same size and will be defined in the main function setup. List `g` must always be the last value in the output tuple

# Examples and capabilities

## Example 1: Creating an Objective Function

To demonstrate how to create an objective function, we use Beck [1] as an example. The State Limit Function is given by:

$$\mathbf{G}(\boldsymbol{X}) = \boldsymbol{R}_d - \boldsymbol{D} - \boldsymbol{L} \tag{2}$$

Here are some implementations of the example function:

```python
def example_function(x, none_variable):
    """Beck example"""

    # random variables statement  
    r_d = x[0]
    d = x[1]
    l = x[2]

    # state limit function
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

    # state limit function
    g = r_d - d - l

    return [r_d], [d+l], [g]

# or

def example_function(x, none_variable):
    """Beck example"""

    # random variables statement  
    r_d = x[0]
    d = x[1]
    l = x[2]

    # state limit function
    r = [r_d]
    s = [d + l]
    g = [r - s]

    return r, s, g
```

## Exemple 2: using Multiple State Limit Functions

For example, consider two State Limit Functions:

$$\mathbf{G}_0 = \boldsymbol{R}_d - \boldsymbol{D} - \boldsymbol{L} \tag{3}$$

$$\mathbf{G}_1 = \boldsymbol{\sigma_y} \cdot \boldsymbol{W} - \boldsymbol{M} \tag{4}$$

Here’s how to implement this:

```python
def example_function(x, none_variable):
    """Beck example"""

    # random variables statement g_0
    r_d = x[0]
    d = x[1]
    l = x[2]

    # random variables statement g_1
    sigma_y = x[3]
    w = x[4]
    m = x[5]

    # state limit function g_0
    r_0 = r_d
    s_0 = d + l
    g_0 = r_0 - s_0

    # state limit function g_1
    r_1 = sigma_y * w
    s_1 = m
    g_1 = r_1 - s_1

    return [r_0, r_1], [s_0, s_1], [g_0, g_1]
```
  
## Concluding remarks
PAREpy represents a contribution to structural reliability engineering by offering an accessible, efficient, and extensible platform for probabilistic analysis. Its use in education and research will help advance the understanding and application of reliability concepts, supporting the development of safer and more robust structural designs.  

The framework simplifies the complexity often associated with reliability analyses by offering intuitive functions for probability of failure estimation, reliability index calculation, and convergence assessment. Moreover, its modular structure allows users to define custom objective functions and integrate them seamlessly into their reliability studies.  
  
Future developments of PAREpy aim to expand its capabilities by incorporating additional reliability methods, such as Subset Simulation and Importance Sampling, along with enhanced visualization tools for reliability analysis. Additionally, the platform intends to evolve by including predefined problems and functionalities, making it easier for users to apply the framework to common reliability scenarios. Furthermore, deterministic methods like FORM (First Order Reliability Method) and SORM (Second Order Reliability Method) will be integrated, allowing for a more comprehensive approach to structural reliability analysis.
