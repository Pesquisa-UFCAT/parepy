---
layout: home
parent: Learning
nav_order: 4
has_children: true
has_toc: true
title: Gumbel right distribution
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h1>Gumbel Max (right)</h1>

<p align="justify">A distribuição de Gumbel, também conhecida como distribuição de valores extremos do tipo I, é uma distribuição de probabilidade contínua amplamente utilizada para modelar eventos extremos, como máximos ou mínimos de um conjunto de dados. Quando a cauda superior da distribuição inicial \(X\) apresenta taxa de decrescimento exponencial, a distribuição dos máximos de \(X\) tende assintoticamente a uma distribuição de Gumbel. Distribuições com cauda exponencial incluem normal, exponencial e gamma.

<p align="justify">São parâmetros da distribuição de Gumbel para máximos:</p>

$$
\begin{align*}
u_n &= \text{máximo característico ou moda de } X_n, \\
\beta &= \text{parâmetro de forma}.
\end{align*}

$$

<p align="justify">A função densidade de probabilidade (PDF) da distribuição de Gumbel Right é dada por:</p>

$$
f_{X_n}(x) = \beta exp[-\beta(x - u_n) - exp(-\beta(x - u_n))], \quad \text{para } -\infty < x < \infty.
$$

<p align="justify">A função de distribuição acumulada (CDF) da distribuição de Gumbel Right é dada por:</p>
$$
F_{X_n}(x) = exp[-exp(-\beta(x - u_n))], \quad \text{para } -\infty < x < \infty.
$$

<p align="justify">Conhecido os parâmetros, os momentos são determinados por:</p>

$$
\begin{align*}
\mu = u_n + \gamma \beta \\
\sigma = \frac{\pi}{\sqrt{6}} \cdot \frac{1}{\beta} \\
\gamma_3 = 1.1396 \quad \text{(coef. de simetria)}
\end{align*}
$$

<p align="justify">Sendo \(\gamma = 0,577216\) a constante de Euler. Conhecidos os momentos, os parâmetros são determinados por: </p>

$$
\begin{align*}
u_n &= \mu - \frac{\gamma}{\beta} \\
\beta &= \frac{\pi}{\sqrt{6}} \cdot \frac{1}{\sigma}
\end{align*}
$$


<h2>Exemplo</h2>

<p align="justify">Suponha que estamos interessados em modelar a <strong>temperatura máxima diária</strong> em uma cidade para entender o comportamento dos extremos de temperatura ao longo do tempo. A coleta de dados foi feita ao longo de 365 dias e, ao analisar os máximos diários de temperatura, determinamos que a distribuição dos máximos segue uma distribuição de Gumbel.</p>

<p align="justify">Vamos imaginar que os dados de temperatura máxima diária (em °C) coletados ao longo de um ano são aproximadamente os seguintes:</p>

$$
T_{\text{máx}} = \{34.1, 35.5, 33.8, 36.2, 32.9, 34.7, \ldots\}
$$

<p align="justify">Esses dados foram usados para calcular os parâmetros da distribuição de Gumbel. Para simplificar, vamos assumir que já obtivemos os seguintes valores para os parâmetros da distribuição:</p>

$$
\begin{align*}
u_n &= 32.0 \quad \text{°C (máximo característico ou moda)}, \\
\beta &= 0.1 \quad \text{°C}^{-1} \quad \text{(parâmetro de forma)}.
\end{align*}
$$

<p align="justify">Com esses parâmetros, podemos calcular a função densidade de probabilidade (PDF) e a função de distribuição acumulada (CDF) da distribuição de Gumbel para a temperatura máxima diária. Vamos realizar os cálculos para entender como essas funções podem ser usadas para prever a probabilidade de eventos extremos de temperatura.</p>

<p align="justify"> A PDF é dada por:</p>

$$
f_T(x) = 0.1 \exp\left[-0.1(x - 32.0) - \exp(-0.1(x - 32.0))\right]
$$

<p align="justify">Se quisermos calcular a probabilidade de que a temperatura máxima em um dia seja maior que 35 °C, substituímos \(x = 35\) na fórmula da PDF:</p>

$$
f_T(35) = 0.1 \exp\left[-0.1(35 - 32) - \exp(-0.1(35 - 32))\right]
$$

<p align="justify">A CDF é dada por:</p>

$$
F_T(x) = \exp\left[-\exp(-0.1(x - 32.0))\right]
$$

<p align="justify">Se quisermos calcular a probabilidade de que a temperatura máxima em um dia seja <strong>menor que</strong> 35 °C, substituímos \(x = 35\) na fórmula da CDF:</p>

$$
F_T(35) = \exp\left[-\exp(-0.1(35 - 32))\right]
$$

#### Interpretação dos Resultados
- A **CDF** nos dá a probabilidade acumulada de que a temperatura máxima seja **menor ou igual** a um valor específico. Por exemplo, se \(F_T(35) = 0.85\), isso significa que há 85% de chance de que a temperatura máxima de um dia seja menor ou igual a 35 °C.
- A **PDF** nos dá a densidade de probabilidade de que a temperatura máxima seja exatamente igual a um valor. Embora a PDF de uma variável contínua não forneça uma probabilidade exata para um valor específico, ela nos dá a "intensidade" da probabilidade em torno de um valor.

#### Passo 5: Uso da Distribuição para Previsões
Se quisermos prever o valor do **máximo extremo** da temperatura para os próximos anos, a distribuição de Gumbel nos permitirá modelar esses extremos com base nos dados históricos. Por exemplo, podemos usar a fórmula da CDF para estimar a probabilidade de um evento extremamente quente (por exemplo, temperatura superior a 38 °C) ocorrer em um futuro próximo.
