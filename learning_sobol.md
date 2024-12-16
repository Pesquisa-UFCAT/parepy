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

<p align="justify">Os índices de Sobol são agora uma ferramenta comum para métodos de sensibilidade global que visam detectar as variáveis/parâmetros de entrada mais influentes em modelos de computador complexos. Nessas estruturas, os índices de Sobol são utilizados para quantificar como a variabilidade de uma entrada afeta a variabilidade de uma saída de um modelo, bem como para determinar interações entre as suas variáveis. A análise de Sobol é especialmente útil em modelos com múltiplos parâmetros, onde é necessário compreender quais variáveis contribuem mais para a incerteza do modelo.</p>

<p align="justify">A análise de Sobol baseia-se na decomposição da variância do modelo em contribuições atribuídas às variáveis de entrada e suas combinações. Para um modelo \(f(X)\), onde \(X = (X_1, X_2, ..., X_k)\) representa um conjunto de variáveis de entrada, a decomposição da variância é dada por:</p>

$$
f(X) = f_0 + \sum_{i=1}^k f_i(X_i) + \sum_{1 \leq i < j \leq k} f_{ij}(X_i, X_j) + \ldots + f_{1,2,\ldots,k}(X_1, X_2, \ldots, X_k),
$$

<p align="justify">onde \(f_0\) é o valor médio de \(f(X)\), \(f_i(X_i)\) é a contribuição individual da variável \(X_i\), \(f_{ij}(X_i, X_j)\) é a contribuição conjunta das variáveis \(X_i\) e \(X_j\), e assim por diante. Esta decomposição permite isolar os efeitos individuais e interativos das variáveis de entrada.</p>

<h2>Índices de Sobol</h2>

<p align="justify">Os índices de Sobol são baseados na decomposição da variância do modelo. Dois índices principais são utilizados:</p>

- **Índice de Primeira Ordem (\(S_i\))**: Mede a contribuição direta da variável \(X_i\) para a variância do modelo, ignorando interações com outras variáveis. É definido como:</p>

<!-- $$
A = 
\begin{bmatrix}
x_1^{(1)} & x_2^{(1)} & \cdots & x_i^{(1)} & \cdots & x_k^{(1)} \\
x_1^{(2)} & x_2^{(2)} & \cdots & x_i^{(2)} & \cdots & x_k^{(2)} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
x_1^{(N-1)} & x_2^{(N-1)} & \cdots & x_i^{(N-1)} & \cdots & x_k^{(N-1)} \\
x_1^{(N)} & x_2^{(N)} & \cdots & x_i^{(N)} & \cdots & x_k^{(N)}
\end{bmatrix}
$$ -->

$$
S_i = \frac{\mathrm{Var}[f_i(X_i)]}{\mathrm{Var}[f(X)]}.
$$

- **Índice de Ordem Total (\(S_{Ti}\))**: Mede a contribuição total da variável \(X_i\), incluindo seus efeitos diretos e suas interações com outras variáveis. É definido como:</p>

$$
S_{Ti} = 1 - \frac{\mathrm{Var}[f_{\sim i}(X_{\sim i})]}{\mathrm{Var}[f(X)]},
$$

<p align="justify">onde \(f_{\sim i}(X_{\sim i})\) é a função do modelo sem a variável \(X_i\).</p>

<p align="justify">A soma de todos os índices de primeira ordem (\(S_i\)) e suas interações é igual a 1, garantindo que toda a variabilidade do modelo seja atribuída às entradas.</p>

<h2>Método de Cálculo</h2>

<p align="justify">Os índices de Sobol são calculados usando amostragem de Monte Carlo. A ideia é gerar duas matrizes de amostras independentes \(A\) e \(B\) de \(N\) amostras cada. Para cada variável \(X_i\), é gerada uma matriz combinada \(C^{(i)}\) onde todas as colunas são copiadas de \(B\), exceto a \(i\)-ésima, que é copiada de \(A\). O modelo \(f(X)\) é avaliado para \(A\), \(B\) e \(C^{(i)}\), e os índices de Sobol são estimados como:</p>

$$
S_i = \frac{\sum_{j=1}^N f(A_j) \cdot f(C^{(i)}_j) - f_0^2}{\mathrm{Var}[f(X)]},
$$

$$
S_{Ti} = \frac{\mathrm{Var}[f(X)] - \sum_{j=1}^N f(B_j) \cdot f(C^{(i)}_j) + f_0^2}{\mathrm{Var}[f(X)]}.
$$

<h2>Exemplo</h2>

<p align="justify">Considere um modelo simples onde a saída \(f(X)\) é uma função de duas variáveis \(X_1\) e \(X_2\), definidas como:</p>

$$
f(X_1, X_2) = X_1^2 + 2X_1X_2 + X_2^2.
$$

<p align="justify">Suponha que \(X_1\) e \(X_2\) sigam uma distribuição uniforme no intervalo \([0, 1]\). Os índices de Sobol podem ser usados para analisar como \(X_1\) e \(X_2\) contribuem para a variabilidade de \(f(X_1, X_2)\).</p>

- **Cálculo de \(S_1\):** Mede a variabilidade devida exclusivamente a \(X_1\).
- **Cálculo de \(S_2\):** Mede a variabilidade devida exclusivamente a \(X_2\).
- **Cálculo de \(S_{T1}\):** Mede a variabilidade devida a \(X_1\), incluindo interações com \(X_2\).
- **Cálculo de \(S_{T2}\):** Mede a variabilidade devida a \(X_2\), incluindo interações com \(X_1\).</p>

<p align="justify">Os índices de Sobol fornecem uma visão clara da importância de cada variável e suas interações, auxiliando na simplificação e otimização de modelos complexos.</p>