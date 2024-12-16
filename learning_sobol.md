---
layout: home
parent: Learning
nav_order: 6
has_children: true
has_toc: true
title: Sobol Indices
---

<!--Don't delete this script-->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h1>Sobol Indices</h1>

<p align="justify">A análise de sensibilidade baseada em variância ou índices de Sobol, é uma forma de análise de sensibilidade global que visam detectar as variáveis/parâmetros de entrada mais influentes em modelos de computador complexos. Nessas estruturas, os índices de Sobol são utilizados para quantificar como a variabilidade de uma entrada afeta a variabilidade de uma saída de um modelo, bem como para determinar interações entre as suas variáveis. A análise de Sobol é especialmente útil em modelos com múltiplos parâmetros, onde é necessário compreender quais variáveis contribuem mais para a incerteza do modelo.</p>

<h2>Procedimento de Cálculo dos Índices de Sensibilidade</h2>

<p align="justify">O procedimento numérico baseado em Monte Carlo para calcular os índices de sensibilidade de primeira ordem e os índices de efeito total para um modelo com \(k\) fatores de entrada são discutidos a seguir:</p>

1. **Geração da Amostra Base**:  O procedimento começa com a geração de uma matriz \((N, 2k)\) de números aleatórios, onde \(k\) é o número de entradas e \(N\) é o tamanho da amostra base. Em termos práticos, \(N\) pode variar de algumas centenas a alguns milhares. A matriz é dividida em duas partes, \(A\) e \(B\), cada uma com \(N\) amostras e \(k\) variáveis. 

$$
A = 
\begin{bmatrix}
x_1^{(1)} & x_2^{(1)} & \cdots & x_i^{(1)} & \cdots & x_k^{(1)} \\
x_1^{(2)} & x_2^{(2)} & \cdots & x_i^{(2)} & \cdots & x_k^{(2)} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
x_1^{(N-1)} & x_2^{(N-1)} & \cdots & x_i^{(N-1)} & \cdots & x_k^{(N-1)} \\
x_1^{(N)} & x_2^{(N)} & \cdots & x_i^{(N)} & \cdots & x_k^{(N)}
\end{bmatrix}
$$

$$
B =
\begin{bmatrix}
x_{k+1}^{(1)} & x_{k+2}^{(1)} & \cdots & x_{k+i}^{(1)} & \cdots & x_{2k}^{(1)} \\
x_{k+1}^{(2)} & x_{k+2}^{(2)} & \cdots & x_{k+i}^{(2)} & \cdots & x_{2k}^{(2)} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
x_{k+1}^{(N-1)} & x_{k+2}^{(N-1)} & \cdots & x_{k+i}^{(N-1)} & \cdots & x_{2k}^{(N-1)} \\
x_{k+1}^{(N)} & x_{k+2}^{(N)} & \cdots & x_{k+i}^{(N)} & \cdots & x_{2k}^{(N)}
\end{bmatrix}
$$

2. **Criação da Matriz C**: Para cada variável \(X_i\), é gerada uma matriz combinada \(C^{(i)}\) onde todas as colunas são copiadas de \(B\), exceto a \(i\)-ésima, que é copiada de \(A\).

$$
C^{(i)} = 
\begin{bmatrix}
x_{k+1}^{(1)} & x_{k+2}^{(1)} & \cdots & x_{i}^{(1)} & \cdots & x_{2k}^{(1)} \\
x_{k+1}^{(2)} & x_{k+2}^{(2)} & \cdots & x_{i}^{(2)} & \cdots & x_{2k}^{(2)} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
x_{k+1}^{(N-1)} & x_{k+2}^{(N-1)} & \cdots & x_{i}^{(N-1)} & \cdots & x_{2k}^{(N-1)} \\
x_{k+1}^{(N)} & x_{k+2}^{(N)} & \cdots & x_{i}^{(N)} & \cdots & x_{2k}^{(N)}
\end{bmatrix}
$$

3. **Cálculo da Saída do Modelo**: Calcule a saída do modelo para todos os valores de entrada nas matrizes de amostra \(A\), \(B\) e \(C_i\), obtendo três vetores de saídas do modelo de dimensão \(N \times 1\):

$$
y_A = f(A), \quad y_B = f(B), \quad y_{C_i} = f(C_i)
$$

4. **Cálculo dos Índices de Sensibilidade**: Os índices de sensibilidade de primeira ordem \(S_i\) e de efeito total \(S_{T_i}\) são calculados para cada variável \(X_i\) usando as saídas do modelo:

$$
S_i = \frac{\mathrm{V}[E(Y|X_i)]}{\mathrm{V}(Y)} = \frac{y_A \cdot y_{C_i} - f_0^2}{y_A \cdot y_A - f_0^2} = \frac{ \sum_{j=1}^{N} y_A^{(j)} y_{C_i}^{(j)} - f_0^2 } { \sum_{j=1}^{N} y_A^{(j)2} - f_0^2 }
$$

onde

$$
f_0^2 = \left( \frac{1}{N} \sum_{j=1}^{N} y_A^{(j)} \right)^2
$$

é a média, e o símbolo (·) denota o produto escalar de dois vetores.

5. **Cálculo dos Índices de Efeito Total**: Os índices de efeito total são calculados como:

$$
S_{T_i} = 1 - \frac{\mathrm{V}[E(Y|X_{\sim i})]}{\mathrm{V}(Y)} = 1 - \frac{y_B \cdot y_{C_i} - f_0^2}{y_A \cdot y_A - f_0^2} = 1 - \frac{\frac{1}{N} \sum_{j=1}^{N} y_B^{(j)} y_{C_i}^{(j)} - f_0^2 } {\frac{1}{N} \sum_{j=1}^{N} y_A^{(j)2} - f_0^2 }
$$

<h2>Exemplo</h2>

<p align="justify">Considere um modelo simples onde a saída \(f(X)\) é uma função de duas variáveis \(X_1\) e \(X_2\), definidas como:</p>

$$
f(X) = X_1^2 + X_2^2
$$

Onde \(X_1\) e \(X_2\) são as variáveis de entrada.

### **Passos do Cálculo**

1. **Geração da Amostra Base**

Suponha que a amostra base tenha \(N = 100\) pontos e o número de variáveis de entrada seja \(k = 2\) (neste caso, \(X_1\) e \(X_2\)). Logo, temos uma matriz \((N, 2k)\) com 100 amostras e 4 colunas.

Aqui está um exemplo de como as matrizes \(A\) e \(B\) poderiam se parecer, onde cada coluna corresponde a uma variável \(X_1\), \(X_2\) e suas respectivas cópias:

Matriz \(A\) (com valores aleatórios):

$$
A = \begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.6 \\
0.4 & 0.7 \\
\vdots & \vdots \\
0.9 & 0.8 \\
\end{bmatrix}
$$

Matriz \(B\) (com valores aleatórios):

$$
B = \begin{bmatrix}
0.5 & 0.8 \\
0.7 & 0.9 \\
0.2 & 0.1 \\
\vdots & \vdots \\
0.3 & 0.4 \\
\end{bmatrix}
$$

2. **Criação da Matriz \(C^{(i)}\)**

Para cada variável \(X_i\), criamos uma matriz combinada \(C^{(i)}\) onde todas as colunas de \(B\) são copiadas, exceto a coluna \(i\)-ésima, que vem de \(A\). Por exemplo, para \(i = 1\):

$$
C^{(1)} = \begin{bmatrix}
0.1 & 0.8 \\
0.2 & 0.9 \\
0.4 & 0.1 \\
\vdots & \vdots \\
0.9 & 0.4 \\
\end{bmatrix}
$$

Para \(i = 2\), a matriz \(C^{(2)}\) seria:

$$
C^{(2)} = \begin{bmatrix}
0.5 & 0.3 \\
0.7 & 0.6 \\
0.2 & 0.7 \\
\vdots & \vdots \\
0.3 & 0.8 \\
\end{bmatrix}
$$

3. **Cálculo da Saída do Modelo**

Agora, calculamos a saída do modelo para todas as entradas das matrizes \(A\), \(B\) e \(C_i\). Neste caso, como temos um modelo simples de soma dos quadrados das entradas, temos:

- \(y_A = f(A) = X_1^2 + X_2^2\) com os valores das amostras de \(A\),
- \(y_B = f(B) = X_1^2 + X_2^2\) com os valores das amostras de \(B\),
- \(y_{C_1} = f(C^{(1)}) = X_1^2 + X_2^2\) com as variáveis \(X_1\) e \(X_2\) alteradas conforme \(C^{(1)}\),
- \(y_{C_2} = f(C^{(2)}) = X_1^2 + X_2^2\) com as variáveis \(X_1\) e \(X_2\) alteradas conforme \(C^{(2)}\).

Para cada linha da matriz \(A\), podemos calcular os valores de \(y_A\), \(y_B\) e \(y_{C_i}\), por exemplo:

$$
y_A^{(1)} = 0.1^2 + 0.3^2 = 0.01 + 0.09 = 0.1
$$
$$
y_B^{(1)} = 0.5^2 + 0.8^2 = 0.25 + 0.64 = 0.89
$$
$$
y_{C_1}^{(1)} = 0.1^2 + 0.8^2 = 0.01 + 0.64 = 0.65
$$
$$
y_{C_2}^{(1)} = 0.5^2 + 0.3^2 = 0.25 + 0.09 = 0.34
$$

E assim por diante para todas as amostras.

4. **Cálculo dos Índices de Sensibilidade**

Agora, podemos calcular os índices de sensibilidade de primeira ordem \(S_1\) e \(S_2\). Vamos usar a fórmula de \(S_i\) para a variável \(X_1\):

$$
S_1 = \frac{y_A \cdot y_{C_1} - f_0^2}{y_A \cdot y_A - f_0^2}
$$

Onde \(f_0^2\) é a média quadrática das saídas de \(y_A\):

$$
f_0^2 = \left( \frac{1}{N} \sum_{j=1}^{N} y_A^{(j)} \right)^2
$$

A soma de \(y_A\) seria realizada para todas as \(N\) amostras, e então calcularíamos \(f_0^2\).

Supondo que \(f_0^2 = 0.5\) (apenas um exemplo), podemos calcular \(S_1\).

Além disso, o índice de efeito total \(S_{T_1}\) é dado por:

$$
S_{T_1} = 1 - \frac{y_B \cdot y_{C_1} - f_0^2}{y_A \cdot y_A - f_0^2}
$$

Esse procedimento seria repetido para todas as variáveis de entrada \(X_1\) e \(X_2\).
