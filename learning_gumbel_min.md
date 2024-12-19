---
layout: home
parent: Learning
nav_order: 5
has_children: true
has_toc: true
title: Gumbel left distribution
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h1>Gumbel Min (left)</h1>

<p align="justify">A distribuição de Gumbel, também conhecida como distribuição de valores extremos do tipo I, pode ser utilizada para modelar eventos extremos relacionados aos mínimos de um conjunto de dados. Quando a cauda inferior da distribuição inicial \(X\) apresenta taxa de decrescimento exponencial, a distribuição dos mínimos de \(X\) tende assintoticamente a uma distribuição de Gumbel. Distribuições com cauda exponencial incluem normal, exponencial e gamma.</p>

<p align="justify">A função de Gumbel é tabelada na variável São parâmetros da distribuição de Gumbel para mínimos:</p>

$$
\begin{align*}
u_1 &= \text{mínimo característico ou moda de } X_1, \\
\beta &= \text{parâmetro de forma}.
\end{align*}
$$

<p align="justify">Quando representada em papel de Gumbel, \(\beta^{-1}\) é a inclinação da distribuição. A função densidade de probabilidade (PDF) da distribuição de Gumbel Min é dada por:</p>

$$
f_{X_n}(x) = \beta \exp\left[\beta(x - u_1) - \exp(\beta(x - u_1))\right], \quad \text{para } -\infty < x < \infty.
$$

<p align="justify">A função de distribuição acumulada (CDF) da distribuição de Gumbel Min é dada por:</p>

$$
F_{X_n}(x) = 1 - \exp\left[-\exp(\beta(x - u_1))\right], \quad \text{para } -\infty < x < \infty.
$$

<p align="justify">A função de Gumbel é tabelada na variável \(w = \beta(x - u_1)\). Conhecidos os parâmetros, os momentos são determinados por:</p>

$$
\begin{align*}
\mu &= u_1 - \frac{\gamma}{\beta}, \\
\sigma &= \frac{\pi}{\sqrt{6}} \cdot \frac{1}{\beta}, \\
\gamma_3 &= -1.1396 \quad \text{(coef. de simetria)}.
\end{align*}
$$

<p align="justify">Sendo \(\gamma = 0.577216\) a constante de Euler. Conhecidos os momentos, os parâmetros são determinados por: </p>

$$
\begin{align*}
u_1 &= \mu + \frac{\gamma}{\beta}, \\
\beta &= \frac{\pi}{\sqrt{6}} \cdot \frac{1}{\sigma}.
\end{align*}
$$

<h2>Exemplo</h2>

<p align="justify">Suponha que estamos interessados em modelar a <strong>temperatura mínima diária</strong> em uma cidade para entender o comportamento dos extremos de temperatura ao longo do tempo. A coleta de dados foi feita ao longo de 365 dias e, ao analisar os mínimos diários de temperatura, determinamos que a distribuição dos mínimos segue uma distribuição de Gumbel.</p>

<p align="justify">Vamos imaginar que os dados de temperatura mínima diária (em °C) coletados ao longo de um ano são aproximadamente os seguintes:</p>

$$
T_{\text{mín}} = \{15.2, 14.5, 13.8, 12.1, 16.0, 14.9, \ldots\}
$$

<p align="justify">Esses dados foram usados para calcular os parâmetros da distribuição de Gumbel. Para simplificar, vamos assumir que já obtivemos os seguintes valores para os parâmetros da distribuição:</p>

$$
\begin{align*}
u_n &= 13.0 \quad \text{°C (mínimo característico ou moda)}, \\
\beta &= 0.2 \quad \text{°C}^{-1} \quad \text{(parâmetro de forma)}.
\end{align*}
$$

<p align="justify">Com esses parâmetros, podemos calcular a função densidade de probabilidade (PDF) e a função de distribuição acumulada (CDF) da distribuição de Gumbel para a temperatura mínima diária. Vamos realizar os cálculos para entender como essas funções podem ser usadas para prever a probabilidade de eventos extremos de temperatura.</p>

<p align="justify">A PDF é dada por:</p>

$$
f_T(x) = 0.2 \exp\left[0.2(x - 13.0) - \exp(0.2(x - 13.0))\right]
$$

<p align="justify">Se quisermos calcular a probabilidade de que a temperatura mínima em um dia seja menor que 10 °C, substituímos \(x = 10\) na fórmula da PDF:</p>

$$
f_T(10) = 0.2 \exp\left[0.2(10 - 13) - \exp(0.2(10 - 13))\right]
$$

<p align="justify">A CDF é dada por:</p>

$$
F_T(x) = 1 - \exp\left[-\exp(0.2(x - 13.0))\right]
$$

<p align="justify">Se quisermos calcular a probabilidade de que a temperatura mínima em um dia seja <strong>maior que</strong> 10 °C, usamos a fórmula complementar da CDF:</p>

$$
P(T > 10) = 1 - F_T(10)
$$

<p align="justify">Substituímos \(x = 10\) na fórmula da CDF:</p>

$$
F_T(10) = 1 - \exp\left[-\exp(0.2(10 - 13))\right]
$$

<p align="justify">Interpretação dos Resultados:</p>

- A CDF nos dá a probabilidade acumulada de que a temperatura mínima seja <strong>menor ou igual</strong> a um valor específico. Por exemplo, se \(F_T(10) = 0.15\), isso significa que há 15% de chance de que a temperatura mínima de um dia seja menor ou igual a 10 °C.

- A PDF nos dá a densidade de probabilidade de que a temperatura mínima seja exatamente igual a um valor. Embora a PDF de uma variável contínua não forneça uma probabilidade exata para um valor específico, ela nos dá a "intensidade" da probabilidade em torno de um valor.