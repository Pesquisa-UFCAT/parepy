---
layout: home
parent: Learning
nav_order: 1
has_children: true
has_toc: true
title: Uniform distribution
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h1>Uniform distribution</h1>

<p align="justify">A distribuição uniforme é um tipo de distribuição de probabilidade contínua que descreve eventos onde todos os valores dentro de um intervalo são igualmente prováveis. Assim, cada um dos \(n\) valores possíveis tem a mesma chance de ocorrer (\(1/n\)). A distribuição uniforme é caracterizada por dois parâmetros, o limite inferior (\(a\)) e o limite superior (\(b\)), que definem o intervalo de valores possíveis. A função densidade de probabilidade (PDF) da distribuição uniforme é constante dentro do intervalo \([a, b]\) e zero fora dele.</p>

<p align="justify">Matematicamente, a função densidade de probabilidade (PDF) é definida como:</p>

$$
\begin{align*}
f(x) &= \frac{1}{b-a}, \quad \text{para } a \leq x \leq b \\
f(x) &= 0, \quad \text{para } x < a \text{ ou } x > b
\end{align*}
$$

<h3 align="justify">Função de Distribuição Acumulada (CDF)</h3>

<p align="justify">A função de distribuição acumulada (CDF) da distribuição uniforme, denotada por \(F(x)\), descreve a probabilidade acumulada até um ponto \(x\). Ela é definida como:</p>

$$
F(x) =
\begin{cases}
0, & \text{se } x < a, \\
\frac{x - a}{b - a}, & \text{se } a \leq x \leq b, \\
1, & \text{se } x > b.
\end{cases}
$$

<p align="justify">Assim, a CDF é uma função que cresce linearmente no intervalo \([a, b]\), com valores que variam de 0 a 1.</p>

<h3 align="justify">Momentos</h3>

<p align="justify">Além disso, essa distribuição possui momentos que são derivados diretamente de seus limites. O momento de ordem 1, ou seja, a média (\(\mu\)), é o ponto médio do intervalo, enquanto a variância (\(\sigma^2\)) descreve a dispersão dos valores ao redor da média. Esses momentos são calculados como:</p>

$$
\begin{align*}
\mu &= \frac{a+b}{2} \\
\sigma^2 &= \frac{(b-a)^2}{12}
\end{align*}
$$

<p align="justify">Os limites \(a\) e \(b\) podem ser determinados diretamente a partir da média e do desvio padrão (\(\sigma\)), o que torna a distribuição uniforme bastante conveniente para aplicações práticas. A relação é dada por:</p>

$$
\begin{align*}
a &= \mu - \sqrt{3}\sigma \\
b &= \mu + \sqrt{3}\sigma
\end{align*}
$$

<h2>Exemplo</h2>

<p align="justify">Para exemplificar a distribuição uniforme, considere um problema de engenharia onde a resistência de um material é modelada como uma variável aleatória uniformemente distribuída entre 50 e 100 MPa. A média e o desvio padrão da resistência são calculados como:</p>

$$
\mu = \frac{50+100}{2} = 75 \text{ MPa}
$$

$$
\sigma = \sqrt{\frac{(100-50)^2}{12}} = 14.43 \text{ MPa}
$$

<p align="justify">Substituindo esses valores na relação entre os limites e os momentos, obtemos:</p>

$$
a = 75 - \sqrt{3} \times 14.43 = 60.71 \text{ MPa}
$$

$$
b = 75 + \sqrt{3} \times 14.43 = 89.29 \text{ MPa}
$$

<p align="justify">Portanto, a resistência do material é modelada como uma variável aleatória uniformemente distribuída entre 60.71 e 89.29 MPa. A CDF correspondente para a resistência seria:</p>

$$
F(x) =
\begin{cases}
0, & \text{se } x < 60.71, \\
\frac{x - 60.71}{89.29 - 60.71}, & \text{se } 60.71 \leq x \leq 89.29, \\
1, & \text{se } x > 89.29.
\end{cases}
$$