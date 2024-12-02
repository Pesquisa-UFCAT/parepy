# PAREpy: A Probabilistic Approach to Reliability Engineering in Python

[Prof. PhD Wanderlei Malaquias Pereira Junior](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4460682U0), 
[Prof. PhD Daniel de Lima Araújo](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4768864J3),
[Prof. PhD André Teófilo Beck](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4790835Y8),
[Prof. PhD André Luis Christoforo](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4706301Z5),
[Prof. PhD Iuri Fazolin](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K8085097P2),
[Prof. PhD Marcos Luiz Henrique](https://sigaa.ufpe.br/sigaa/public/docente/portal.jsf?siape=2324067),
[Prof. Marcos Napoleão Rabelo](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4702141E9),
[Prof. PhD Fran Sergio Lobato](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K4169590P3),
[Msc Murilo Carneiro Rodrigues](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K1774858Z8),
[Msc Matheus Henrique Morato Moraes](https://buscatextual.cnpq.br/buscatextual/visualizacv.do?metodo=apresentar&id=K8214592P6).

## Sumary
PAREpy é uma software baseado em Python para análise de confiabilidade de sistemas de engenharia, considerando variáveis aleatórias e incertezas. O software é baseado em métodos probabilísticos e estatísticos, como o método de amostragem de Monte Carlo, e o Hípercubo Latino. O software é desenvolvido para ser uma ferramenta de código aberto e fácil de usar para engenheiros e pesquisadores que desejam realizar análises de confiabilidade em seus sistemas de engenharia. 

## Current Modules
O software atualmente possui os seguintes módulos:
- **sampling**: este algoritmo gera um conjunto de amostras aleatórias de acordo com o tipo de distribuição de probabilidade especificada.
- **pf_equation**: esta função calcula a probabilidade de falha (pf) para um dado índice de confiabilidade (ϐ) usando uma função de distribuição cumulativa normal padrão. O cálculo é realizado integrando a função de densidade de probabilidade (PDF) de uma distribuição normal padrão.
- **beta_equation**: esta função calcula o valor do índice de confiabilidade para uma determinada probabilidade de falha (pf).
- **calc_pf_beta**: calcula os valores de probabilidade de falha ou índice de confiabilidade das colunas de um DataFrame que começam com 'I_' (função Indicadora). Se um caminho de arquivo .txt for passado, esta função avalia os valores pf e β também.
- **convergence_probability_failure**: esta função calcula a taxa de convergência de uma coluna dada em um data frame. Desta forma, é possível verificar a convergência da probabilidade de falha ou índice de confiabilidade.

## Quick Start


### Install

To use the framework in a **Python** environment, use the following command:

```bash
pip install parepy-toolbox
# or pip install --upgrade parepy-toolbox
```

### Files Structure

Let's use the example of building a problem in PAREpy using Jupyter Notebook or a **Python** file. The basic file structure that you must assemble to use the library must be as follows:

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

## Example 1: Creating an Objective Function

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
