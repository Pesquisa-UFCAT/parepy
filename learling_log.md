---
layout: home
parent: Learning
nav_order: 3
has_children: true
has_toc: true
title: Log-normal distribution
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h1>Distribuição Log-normal</h1>

<p align="justify">Em probabilidade, a distribuição log-normal é uma distribuição de probabilidade contínua de uma variável aleatória cujo logaritmo natural é normalmente distribuído. A distribuição log-normal é amplamente aplicada em situações onde os valores crescem multiplicativamente, como preços de ações, crescimento populacional e tempo de sobrevivência em processos biológicos.</p>

<p align="justify">Se uma variável \(Y\) segue uma distribuição log-normal, então a variável \(X = exp(Y)\) segue uma distribuição log-normal (\(X \sim LN( \lambda, \xi)\)). A função densidade de probabilidade (PDF) da distribuição log-normal é dada por:</p>

$$
f(x) = \frac{1}{\xi x \sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\ln(x) - \lambda)^2}{\xi}\right), \quad \text{para } 0 ≤ x ≤ \infty.
$$


<p align="justify">A função de distribuição acumulada (CDF) da distribuição log-normal, denotada por \(F(x)\), calcula a probabilidade de uma variável assumir um valor menor ou igual a \(x\). Ela é dada por:</p>

$$
F(x) = \Phi \left(\frac{\ln(x) - \lambda}{\xi}\right), \quad \text{para } 0 ≤ x ≤ \infty.
$$

<p align="justify">Essa integral, assim como no caso da distribuição normal, não possui uma solução analítica exata. Por isso, a CDF é frequentemente calculada numericamente. Uma relação útil é a transformação do logaritmo de \(X\):</p>

<p align="justify">Os momentos de uma variável log-normal são derivados diretamente de seus parâmetros. A média, variância e moda da distribuição log-normal são calculadas como:</p>

$$
\mu = \exp(\lambda + \frac{\xi^2}{2})
$$

$$
\sigma = \mu \sqrt{(\exp(\xi^2) - 1)}
$$

<p align="justify">E os parâmetros \(\lambda\) e \(\xi\) podem ser determinados diretamente a partir da média e do desvio padrão (\(\sigma\)), o que torna a distribuição log-normal bastante conveniente para aplicações práticas. A relação é dada por:</p>

$$
\lambda = \ln(\mu) - 0.5 \xi^2
$$

$$
\xi = \sqrt{\ln\left(\frac{\sigma^2}{\mu^2} + 1\right)}
$$

<p align="justify">Para coeficiente de variação \( \delta = \sigma / \mu \lesssim 0,3 \), pode-se aproximar \( \xi \approx \delta \).


<h2>Exemplo</h2>

Suponha que a média (\(\mu\)) de uma variável que segue uma distribuição log-normal seja \(10\) e o desvio padrão (\(\sigma\)) seja \(3\). Desejamos determinar os parâmetros \(\lambda\) e \(\xi\) da distribuição.

1. **Calcular o parâmetro \(\xi\):**
   
   A relação para \(\xi\) é dada por:
   \[
   \xi = \sqrt{\ln\left(\frac{\sigma^2}{\mu^2} + 1\right)}
   \]

   Substituímos os valores de \(\mu = 10\) e \(\sigma = 3\):
   \[
   \xi = \sqrt{\ln\left(\frac{3^2}{10^2} + 1\right)} = \sqrt{\ln\left(\frac{9}{100} + 1\right)} = \sqrt{\ln\left(1.09\right)} \approx \sqrt{0.086} \approx 0.293
   \]

2. **Calcular o parâmetro \(\lambda\):**
   
   A relação para \(\lambda\) é:
   \[
   \lambda = \ln(\mu) - 0.5 \xi^2
   \]

   Substituímos os valores de \(\mu = 10\) e \(\xi \approx 0.293\):
   \[
   \lambda = \ln(10) - 0.5 (0.293)^2 \approx 2.302 - 0.5 (0.086) \approx 2.302 - 0.043 \approx 2.259
   \]

Portanto, os parâmetros da distribuição log-normal são aproximadamente:
\[
\lambda \approx 2.259, \quad \xi \approx 0.293
\]

3. **Validar com a média e desvio padrão:**

   Usando as fórmulas:
   \[
   \mu = \exp(\lambda + 0.5 \xi^2), \quad \sigma = \mu \sqrt{\exp(\xi^2) - 1}
   \]

   Substituímos \(\lambda \approx 2.259\) e \(\xi \approx 0.293\) e verificamos que os valores de \(\mu\) e \(\sigma\) retornam próximos de \(10\) e \(3\), confirmando os parâmetros calculados.
