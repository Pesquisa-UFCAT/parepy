---
title: PAREpy - A Probabilistic Approach to Reliability Engineering in Python

tags:
  - Python
  - Reliability
  - Monte Carlo
  - Sampling
  - Structures

authors:
  - name: Wanderlei Malaquias Pereira Junior
    equal-contrib: true
    affiliation: 1
  - name: Wanderlei Malaquias Pereira Junior
    equal-contrib: true
    affiliation: 1

[Prof. PhD Wanderlei Malaquias Pereira Junior](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4460682U0)$^1$, 
[Msc Murilo Carneiro Rodrigues](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K1774858Z8)$^1$,
[Msc Matheus Henrique Morato Moraes](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K8214592P6)$^3$,
[Prof. PhD Daniel de Lima Araújo](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4768864J3)$^1$,
[Prof. PhD André Teófilo Beck](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4790835Y8)$^2$,
[Prof. PhD André Luis Christoforo](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4706301Z5)$^3$,
[Prof. PhD Iuri Fazolin](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K8085097P2)$^1$,
[Prof. PhD Marcos Luiz Henrique](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4798716D4) $^4$,
[Prof. Marcos Napoleão Rabelo](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4702141E9)$^1$,
[Prof. PhD Fran Sergio Lobato](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4169590P3)$^5$,
[Prof. PhD Gustavo Barbosa Libotte](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4930703A6)$^6$.

**1.** Engineering College, Federal University of Catalão, Brazil. **2.** Department of Structures, São Carlos School of Engineering, Brazil. **3.** Federal University of São Carlos (UFSCar), Brazil. **4.** University Federal of Pernambuco, Campus of the Agreste, Brazil. **5.** School of Chemical Engineering, Federal University of Uberlândia, Center for Exact Sciences and Technology, Brazil. **6.** Polytechnic Institute, State University of Rio de Janeiro, Brazil.  
---

## Introduction and motivation
The `PAREpy` (Probabilistic Approach to Reliability Engineering) framework is a library for applying probabilistic concepts to analyze a system containing random variables. The platform is built in Python and can be used in any environment that supports this programming language.  

Tools often offer a certain complexity when building reliability analyses. This framework intends to be a faster library for building reliability problems. This version, it is able to assemble structural reliability problems using sampling methods and derivative methods.  

The study of structural reliability is concerned with the calculation and prediction of the probability of limit state violation for an engineered structural system at any stage  during its life. In particular, the study of structural safety is concerned with the violation of the ultimate or safety limit states for the structure. More generally, the study of  structural reliability is concerned with the violation of performance measures [@melchers_structural_2018].  

PAREPy offers the following functions:
- `pf_equation`: This function calculates the probability of failure ($p_f$) for a given reliability index ($\beta$) using a standard normal cumulative distribution function. The calculation is performed by integrating the probability density function (PDF) of a standard normal distribution.
- `beta_equation`: This function calculates the reliability index value for a given probability of failure ($p_f$).
- `calc_pf_beta_sampling`: Calculates the values of probability of failure or reliability index from the columns of a DataFrame that start with `I_` (Indicator function). If a `.txt` file path is passed, this function evaluates $p_f$ and $\beta$ values too. You can used this function when you have a dataset with sampling results.
- `convergence_probability_failure`: This function calculates the convergence rate of a given column in a data frame. This function is used to check the convergence of the probability of failure or reliability index.
- `sampling_algorithm_structural_analysis`: This function creates the samples and evaluates the limit state functions in structural reliability problems.
- `concatenates_txt_files_sampling_algorithm_structural_analysis`

The documentation is available at the [PAREpy web site](https://wmpjrufg.github.io/PAREPY/). There, users can find some examples, learning and see application examples.  


## Quick Start

### Install

To use the framework in a **Python** environment, use the following command:

```bash
pip install parepy-toolbox
# or pip install --upgrade parepy-toolbox
```

### Files Structure

Let's use the example of building a problem in PAREpy using **Jupyter Notebook** file or a **Python** file. The basic file structure that you must assemble to use the library must be as follows:

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

The `of_file.py` file should contain the objective function of the problem. The `your_problem` file is the file that will contain the call to the main function and other settings necessary for the use of the algorithm.

### `of_file.py`

`of_file.py` is a Python function, and the user needs to define it for PAREpy to work. `of_file.py` has a fixed structure that must be respected, as described below:

```python
def my_function(x, none_variable):
    # add your code
    return r, s, g
```

### Parameters of `of_file.py`

- **`x`** (type `list`): A list of design random variables. PAREpy generates these values.
- **`none_variable`** (type `None`, `list`, `float`, `dict`, `str`, or any): The user can define this variable. The user can input any value into this variable when calling the framework's main function.

### Returns of `of_file.py`

- **`r`** (type `list`): A list of values. In structural problems, we recommend putting the capacity in this variable.
- **`s`** (type `list`): A list of values. In structural problems, we recommend putting the demand in this variable.
- **`g`** (type `list`): State limit function \( \mathbf{G} = \mathbf{R} - \mathbf{S} \).

> **Important**  
> The lists `r`, `s`, and `g` must have the same size and will be defined in the main function setup.

## Examples and capabilities

### Example 1: Creating an Objective Function

To demonstrate how to create an objective function, we use Beck [1] as an example. The State Limit Function is given by:

$\mathbf{G} = \mathbf{R}_d - \mathbf{D} - \mathbf{L} $

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

$\mathbf{G}_0 = \mathbf{R}_d - \mathbf{D} - \mathbf{L}$

$\mathbf{G}_1 = \mathbf{\sigma_y} \cdot \mathbf{W} - \mathbf{M}$

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

## Documentation
The documentation abou this app can be see in wmpjrufg.github.io/PAREPY/.

## Contributions
A.M.K. was reponsible for conceptualization, methodology, software, testing and validation, writing of manuscript, and visualization; A.D. was responsible for testing software and results in training machine learning models; A.M.B., W.F.R., Z-K.L. were responsible for funding acquisition, review, and editing. Z-K.L. was also supervising the work.
