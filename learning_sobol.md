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

<p align="justify"><strong>Geração da Amostra Base:</strong> O procedimento começa com a geração de uma matriz \((N, 2k)\) de números aleatórios, onde \(k\) é o número de entradas e \(N\) é o tamanho da amostra base. Em termos práticos, \(N\) pode variar de algumas centenas a alguns milhares. A matriz é dividida em duas partes, \(A\) e \(B\), cada uma com \(N\) amostras e \(k\) variáveis.</p>

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

<p align="justify"><strong>Criação da Matriz C:</strong> Para cada variável \(X_i\), é gerada uma matriz combinada \(C^{(i)}\) onde todas as colunas são copiadas de \(B\), exceto a \(i\)-ésima, que é copiada de \(A\).</p>

$$
C_i = 
\begin{bmatrix}
x_{k+1}^{(1)} & x_{k+2}^{(1)} & \cdots & x_{i}^{(1)} & \cdots & x_{2k}^{(1)} \\
x_{k+1}^{(2)} & x_{k+2}^{(2)} & \cdots & x_{i}^{(2)} & \cdots & x_{2k}^{(2)} \\
\vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
x_{k+1}^{(N-1)} & x_{k+2}^{(N-1)} & \cdots & x_{i}^{(N-1)} & \cdots & x_{2k}^{(N-1)} \\
x_{k+1}^{(N)} & x_{k+2}^{(N)} & \cdots & x_{i}^{(N)} & \cdots & x_{2k}^{(N)}
\end{bmatrix}
$$

<p align="justify"><strong>Cálculo da Saída do Modelo:</strong> Calcule a saída do modelo para todos os valores de entrada nas matrizes de amostra \(A\), \(B\) e \(C_i\), obtendo três vetores de saídas do modelo de dimensão \(N \times 1\): </p>

$$
y_A = f(A), \quad y_B = f(B), \quad y_{C_i} = f(C_i)
$$

<p align="justify"><strong>Cálculo dos Índices de Sensibilidade:</strong> Os índices de sensibilidade de primeira ordem \(S_i\) e de efeito total \(S_{T_i}\) são calculados para cada variável \(X_i\) usando as saídas do modelo:</p>

$$
S_i = \frac{\mathrm{V}[E(Y|X_i)]}{\mathrm{V}(Y)} = \frac{y_A \cdot y_{C_i} - f_0^2}{y_A \cdot y_A - f_0^2} = \frac{(1/N) \sum_{j=1}^{N} y_A^{(j)} y_{C_i}^{(j)} - f_0^2 } {(1/N) \sum_{j=1}^{N} y_A^{(j)2} - f_0^2 }
$$

onde

$$
f_0^2 = \left( \frac{1}{N} \sum_{j=1}^{N} y_A^{(j)} \right)^2
$$

é a média, e o símbolo (·) denota o produto escalar de dois vetores.

<p align="justify"><strong>Cálculo dos Índices de Efeito Total:</strong> Os índices de efeito total são calculados como:</p>

$$
S_{T_i} = 1 - \frac{\mathrm{V}[E(Y|X_{\sim i})]}{\mathrm{V}(Y)} = 1 - \frac{y_B \cdot y_{C_i} - f_0^2}{y_A \cdot y_A - f_0^2} = 1 - \frac{(1/N) \sum_{j=1}^{N} y_B^{(j)} y_{C_i}^{(j)} - f_0^2 } {(1/N) \sum_{j=1}^{N} y_A^{(j)2} - f_0^2 }
$$

 <h2>Interpretação dos Índices de Sobol</h2>


<h3>1. Índice de Sobol de Primeira Ordem (Si)</h3>
<p>O índice <em>S<sub>i</sub></em> indica quanto da variabilidade da saída de um modelo pode ser explicada pela variabilidade de uma única variável de entrada <em>X<sub>i</sub></em>, mantendo as outras variáveis fixas. Em termos simples, <em>S<sub>i</sub></em> reflete o <strong>efeito principal</strong> de uma variável no modelo, ou seja, quanto a variabilidade de <em>X<sub>i</sub></em> contribui para a variabilidade da saída. Se uma variável for dominante, <em>S<sub>i</sub></em> será grande, indicando que a sua variação tem um impacto considerável na resposta do modelo.</p>

<h3>2. Índice de Sobol de Efeito Conjunto (S<sub>c</sub><sub>i1, i2, ..., is</sub>)</h3>
<p>Os índices de Sobol de efeito conjunto, como <em>S<sub>c<sub>i1, i2, ..., is</sub></em></em>, medem a redução da variabilidade do modelo quando múltiplas variáveis de entrada são fixadas simultaneamente. Isso captura as <strong>interações</strong> entre as variáveis. Por exemplo, <em>S<sub>c<sub>i1, i2</sub></sub></em> indicaria o efeito conjunto de fixar as variáveis <em>X<sub>1</sub></em> e <em>X<sub>2</sub></em>, o que é útil para entender como as variáveis podem interagir para afetar a saída do modelo. Se <em>S<sub>c<sub>i1, i2</sub></sub></em> for grande, isso sugere que a interação entre essas variáveis tem um papel importante no comportamento do modelo.</p>

<h3>3. Índice de Sobol de Efeito Total (S<sub>T</sub><sub>i</sub>)</h3>
<p>O índice <em>S<sub>T<sub>i</sub></em></em> representa o efeito total de uma variável <em>X<sub>i</sub></em>, levando em consideração tanto o seu efeito direto (efeito principal) quanto as interações com outras variáveis. A relação entre <em>S<sub>T</sub></em> e <em>S<sub>i</sub></em> é importante: <em>S<sub>T<sub>i</sub></em> &ge; S<sub>i</sub></em>, sendo que a diferença <em>S<sub>T</sub><sub>i</sub> - S<sub>i</sub></em> reflete o grau de interação de <em>X<sub>i</sub></em> com outras variáveis. Se <em>S<sub>T</sub><sub>i</sub> = S<sub>i</sub></em>, isso significa que <em>X<sub>i</sub></em> não tem interações relevantes com outras variáveis e seu efeito é exclusivamente devido à sua variabilidade. Por outro lado, se <em>S<sub>T</sub><sub>i</sub></em> for muito maior que <em>S<sub>i</sub></em>, isso indica que <em>X<sub>i</sub></em> está envolvido em interações importantes, e sua contribuição para a variabilidade da saída não é completamente explicada por seu efeito direto.</p>

<h3>4. Índice de Sobol Total (S<sub>T</sub><sub>i</sub> = 0)</h3>
<p>Se <em>S<sub>T</sub><sub>i</sub> = 0</em>, isso significa que a variável <em>X<sub>i</sub></em> não tem efeito sobre a saída do modelo, nem por meio de efeitos principais nem por interações com outras variáveis. Ou seja, <em>X<sub>i</sub></em> pode ser fixada em qualquer valor dentro de seu domínio sem afetar a variabilidade da saída. Isso indica que <em>X<sub>i</sub></em> é irrelevante para o modelo e pode ser ignorada na análise.</p>

<h3>5. Soma dos Índices de Sobol (Para Modelos Aditivos e Não Aditivos)</h3>
<p>Para modelos aditivos, a soma dos índices de Sobol de primeira ordem <em>S<sub>i</sub></em> de todas as variáveis de entrada é igual a 1. Isso significa que a variabilidade da saída pode ser completamente explicada pela contribuição dos efeitos principais das variáveis de entrada. Para modelos não aditivos, a soma dos índices será menor que 1, indicando que a variabilidade da saída é parcialmente explicada por interações entre as variáveis. A diferença <em>1 - &sum; S<sub>i</sub></em> é uma medida do impacto das interações no modelo, refletindo o grau em que as variáveis interagem de forma não aditiva.</p>

<h3>6. Soma dos Índices de Sobol de Efeito Total (S<sub>T</sub><sub>i</sub>)</h3>
<p>A soma de todos os índices <em>S<sub>T</sub><sub>i</sub></em> é sempre maior que 1. Para modelos perfeitamente aditivos, a soma dos <em>S<sub>T</sub><sub>i</sub></em> será igual a 1. Caso contrário, a soma será maior que 1, indicando que o modelo contém interações entre as variáveis de entrada que aumentam a variabilidade da saída.</p>


<!-- <h2>Exemplo</h2>

<p align="justify">Considere um modelo simples onde a saída \(f(X)\) é uma função de duas variáveis \(X_1\) e \(X_2\), definidas como:</p>

$$
f(X) = X_1^2 + X_2^2
$$

Onde \(X_1\) e \(X_2\) são as variáveis de entrada.

<h3>Passos do Cálculo</h3>

<p align="justify"><strong>1. Geração da Amostra Base:</strong> Suponha que a amostra base tenha \(N = 100\) pontos e o número de variáveis de entrada seja \(k = 2\) (neste caso, \(X_1\) e \(X_2\)). Logo, temos uma matriz \((N, 2k)\) com 100 amostras e 4 colunas.</p>

<p align="justify">Aqui está um exemplo de como as matrizes \(A\) e \(B\) poderiam se parecer, onde cada coluna corresponde a uma variável \(X_1\), \(X_2\) e suas respectivas cópias:</p>

<p align="justify"><strong>Matriz \(A\)</strong> (com valores aleatórios):</p>

$$
A = \begin{bmatrix}
0.1 & 0.3 \\
0.2 & 0.6 \\
0.4 & 0.7 \\
\vdots & \vdots \\
0.9 & 0.8 \\
\end{bmatrix}
$$

<p align="justify"><strong>Matriz \(B\)</strong> (com valores aleatórios):</p>

$$
B = \begin{bmatrix}
0.5 & 0.8 \\
0.7 & 0.9 \\
0.2 & 0.1 \\
\vdots & \vdots \\
0.3 & 0.4 \\
\end{bmatrix}
$$

<p align="justify"><strong>2. Criação da Matriz \(C^{(i)}\):</strong> Para cada variável \(X_i\), criamos uma matriz combinada \(C^{(i)}\) onde todas as colunas de \(B\) são copiadas, exceto a coluna \(i\)-ésima, que vem de \(A\). Por exemplo, para \(i = 1\): </p>

$$
C^{(1)} = \begin{bmatrix}
0.1 & 0.8 \\
0.2 & 0.9 \\
0.4 & 0.1 \\
\vdots & \vdots \\
0.9 & 0.4 \\
\end{bmatrix}
$$

<p align="justify">Para \(i = 2\), a matriz \(C^{(2)}\) seria:</p>

$$
C^{(2)} = \begin{bmatrix}
0.5 & 0.3 \\
0.7 & 0.6 \\
0.2 & 0.7 \\
\vdots & \vdots \\
0.3 & 0.8 \\
\end{bmatrix}
$$

<p align="justify"><strong>3. Cálculo da Saída do Modelo:</strong> Agora, calculamos a saída do modelo para todas as entradas das matrizes \(A\), \(B\) e \(C_i\). Neste caso, como temos um modelo simples de soma dos quadrados das entradas, temos:</p>

<ul>
  <li><strong>\(y_A = f(A)\)</strong> = \(X_1^2 + X_2^2\) com os valores das amostras de \(A\)</li>
  <li><strong>\(y_B = f(B)\)</strong> = \(X_1^2 + X_2^2\) com os valores das amostras de \(B\)</li>
  <li><strong>\(y_{C_1} = f(C^{(1)})\)</strong> = \(X_1^2 + X_2^2\) com as variáveis \(X_1\) e \(X_2\) alteradas conforme \(C^{(1)}\)</li>
  <li><strong>\(y_{C_2} = f(C^{(2)})\)</strong> = \(X_1^2 + X_2^2\) com as variáveis \(X_1\) e \(X_2\) alteradas conforme \(C^{(2)}\)</li>
</ul>

<p align="justify">Para cada linha da matriz \(A\), podemos calcular os valores de \(y_A\), \(y_B\) e \(y_{C_i}\), por exemplo:</p>

$$
\begin{align*}
y_A^{(1)} & = 0.1^2 + 0.3^2 = 0.01 + 0.09 = 0.1 \\
y_B^{(1)} & = 0.5^2 + 0.8^2 = 0.25 + 0.64 = 0.89 \\
y_{C_1}^{(1)} & = 0.1^2 + 0.8^2 = 0.01 + 0.64 = 0.65 \\
y_{C_2}^{(1)} & = 0.5^2 + 0.3^2 = 0.25 + 0.09 = 0.34
\end{align*}
$$

<p align="justify">E assim por diante para todas as amostras.</p>

<p align="justify"><strong>4. Cálculo dos Índices de Sensibilidade:</strong> Agora, podemos calcular os índices de sensibilidade de primeira ordem \(S_1\) e \(S_2\). Vamos usar a fórmula de \(S_i\) para a variável \(X_1\): </p>

$$
S_1 = \frac{y_A \cdot y_{C_1} - f_0^2}{y_A \cdot y_A - f_0^2}
$$

<p align="justify">Onde \(f_0^2\) é a média quadrática das saídas de \(y_A\): </p>

$$
f_0^2 = \left( \frac{1}{N} \sum_{j=1}^{N} y_A^{(j)} \right)^2
$$

<p align="justify">A soma de \(y_A\) seria realizada para todas as \(N\) amostras, e então calcularíamos \(f_0^2\).</p>

<p align="justify">Supondo que \(f_0^2 = 0.5\) (apenas um exemplo), podemos calcular \(S_1\).</p>

<p align="justify">Além disso, o índice de efeito total \(S_{T_1}\) é dado por:</p>

$$
S_{T_1} = 1 - \frac{y_B \cdot y_{C_1} - f_0^2}{y_A \cdot y_A - f_0^2}
$$

<p align="justify">Esse procedimento seria repetido para todas as variáveis de entrada \(X_1\) e \(X_2\).</p> -->
