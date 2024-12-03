---
layout: home
parent: Learning
nav_order: 1
has_children: true
has_toc: true
title: Normal distribution
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h1>Distribuição Normal</h1>

<p align="justify">A distribuição normal ou gaussiana é uma das mais importantes distribuições de probabilidade na estatística e em muitas áreas da ciência. Ela descreve variáveis contínuas que apresentam comportamento simétrico ao redor de sua média. A distribuição normal é caracterizada por sua curva em formato de sino, onde os valores próximos à média são os mais prováveis, enquanto os valores extremos são cada vez menos prováveis.</p>

<p align="justify">A distribuição normal é completamente definida por dois parâmetros: a média (\(\mu\)), que indica o ponto central da distribuição, e o desvio padrão (\(\sigma\)), que mede a dispersão dos valores ao redor da média. A função densidade de probabilidade (PDF) da distribuição normal é dada pela fórmula:</p>

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad \text{para } -\infty < x < \infty.
$$

<p align="justify">onde \(\exp\) representa a função exponencial, \(\pi\) é uma constante matemática (aproximadamente 3,14159) e \(\sigma^2\) é a variância.</p>

<p align="justify">A função de distribuição acumulada (CDF) da distribuição normal, denotada por \(F(x)\), calcula a probabilidade de uma variável assumir um valor menor ou igual a \(x\). A CDF é dada por:</p>

$$
F(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left(-\frac{(t - \mu)^2}{2\sigma^2}\right) dt, \quad \text{para } -\infty < x < \infty.
$$

<p align="justify">Infelizmente, esta integral não possui uma solução analítica. Em geral, a CDF pode ser calculada numericamente, e é usada para determinar probabilidades acumuladas, como a área sob a curva da PDF até um ponto \(x\). Tais resultados são frequentemente apresentados em termos de uma distribuição normal padrão, que tem média 0 e desvio padrão 1. Qualquer variável aleatória \(X \sim N(\mu, \sigma^2)\) pode ser transformada em uma variável com distribuição normal padrão \(Y \sim N(0, 1)\) por meio da fórmula:</p>  

$$
Y = \frac{X - \mu}{\sigma}.
$$

<p align="justify">Para a variável normal padrão \(Y\), a função de densidade de probabilidade \( \phi(y) \) e a função de distribuição acumulada \( \Phi(y) \) são definidas como:</p>

$$
\phi(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right), \quad \text{para } -\infty < y < \infty,
$$

$$
\Phi(y) = \int_{-\infty}^{y} \phi(z) dz, \quad \text{para } -\infty < y < \infty.
$$

<p align="justify">A PDF \(f(x)\) para uma variável \(X \sim N(\mu, \sigma)\) pode ser expressa em termos de \(\phi(y)\) da seguinte forma:</p>

$$
f(x) = \phi\left(\frac{x - \mu}{\sigma}\right).
$$

<p align="justify">A distribuição normal é frequentemente utilizada para modelar erros ou desvios em processos produtivos ou de fabricação. Intervalos de confiança são definidos em termos do fator \(k\), que é o número de desvios padrões a partir da média. Para um intervalo de confiança de \(k\) desvios padrão, temos:</p>

$$
P(x_{\text{inf}} < x < x_{\text{sup}}) = P[\mu - k\sigma < x < \mu + k\sigma] = \int_{-k}^{k} \phi(y) dy = \Phi(k) - \Phi(-k).
$$

<p align="justify">Os limites \(x_{\text{inf}}\) e \(x_{\text{sup}}\) são utilizados como filtros em controle de qualidade de produção. Para uma confiança de 95,5%, por exemplo, componentes com dimensões \(x_i < \mu - 2\sigma\) ou \(x_i > \mu + 2\sigma\) são considerados fora de especificação. Isto ajuda a evitar que variações excessivas comprometam a qualidade do produto final.</p>

<h2>Exemplo</h2>

<p align="justify">Considere uma variável que mede a altura de adultos em uma população, que segue uma distribuição normal com média \(\mu = 170\) cm e desvio padrão \(\sigma = 10\) cm. A função densidade de probabilidade é dada por:</p>

$$
f(x) = \frac{1}{\sqrt{2\pi(10)^2}} \exp\left(-\frac{(x - 170)^2}{2(10)^2}\right).
$$

<p align="justify">Usando a regra empírica, podemos inferir que:</p>

- Aproximadamente 68% das alturas estão entre \(160\) cm e \(180\) cm (\(170 \pm 10\)).
- Aproximadamente 95% das alturas estão entre \(150\) cm e \(190\) cm (\(170 \pm 2 \cdot 10\)).
- Aproximadamente 99,7% das alturas estão entre \(140\) cm e \(200\) cm (\(170 \pm 3 \cdot 10\)).

<p align="justify">A CDF dessa distribuição pode ser usada para calcular probabilidades acumuladas, como a probabilidade de uma pessoa ter altura menor que 165 cm (\(P(X \leq 165)\)) ou maior que 185 cm (\(P(X > 185)\)).</p>
