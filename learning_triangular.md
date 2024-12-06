---
layout: home
parent: Learning
nav_order: 5
has_children: true
has_toc: true
title: Triangular distribution
---

<!--Don't delete ths script-->
<script src = "https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id = "MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete ths script-->

<h1>Triangular distribution</h1>

<p align="justify">A distribuição triangular é uma distribuição de probabilidade contínua que é utilizada em situações onde há uma relação conhecida entre os dados variáveis, contudo há relativamente poucos dados disponíveis para conduzir uma análise estatística completa. A distribuição triangular é uma distribuição ideal quando os únicos dados disponíveis são os valores máximo e mínimo, e o resultado mais provável. Ela é frequentemente usada em análise de decisões empresariais.</p>

<p align="justify">A distribuição triangular é definida por três parâmetros: o valor mínimo (\(a\)), o valor máximo (\(b\)) e o valor mais provável (\(c\)), que é o pico da distribuição. A função densidade de probabilidade (PDF) da distribuição triangular é composta por duas partes lineares, formando um triângulo, e é dada pela fórmula:</p>

$$
f(x) = 
\begin{cases} 
0, & x < a \text{ ou } x > b, \\
\frac{2(x-a)}{(b-a)(c-a)}, & a \leq x \leq c, \\
\frac{2(b-x)}{(b-a)(b-c)}, & c < x \leq b.
\end{cases}
$$

<p align="justify">Essa fórmula garante que a área total sob a curva seja igual a 1.</p>

<p align="justify">A CDF da distribuição triangular é dada por:</p>

$$
F(x) = 
\begin{cases} 
0, & x < a, \\
\frac{(x-a)^2}{(b-a)(c-a)}, & a \leq x \leq c, \\
1 - \frac{(b-x)^2}{(b-a)(b-c)}, & c < x \leq b, \\
1, & x > b.
\end{cases}
$$


<h2>Exemplo</h2>

<p align="justify">Considere o tempo necessário para a conclusão de um projeto, estimado com os seguintes valores: tempo mínimo \(a = 20\) dias, tempo máximo \(b = 40\) dias, e tempo mais provável \(c = 30\) dias. A função densidade de probabilidade para essa distribuição triangular é dada por:</p>

$$
f(x) = 
\begin{cases} 
0, & x < 20 \text{ ou } x > 40, \\
\frac{2(x-20)}{(40-20)(30-20)}, & 20 \leq x \leq 30, \\
\frac{2(40-x)}{(40-20)(40-30)}, & 30 < x \leq 40.
\end{cases}
$$

<p align="justify">Com base na forma da distribuição, podemos inferir que:</p>

- O valor mais provável do tempo de conclusão é \(30\) dias.
- Os valores extremos (\(20\) e \(40\) dias) têm a menor probabilidade de ocorrência.
- A maior probabilidade de tempos ocorre entre \(25\) e \(35\) dias, próximos ao valor mais provável.

<p align="justify">A função de distribuição acumulada pode ser usada para calcular probabilidades específicas, como a probabilidade de o projeto ser concluído em menos de 25 dias (\(P(X \leq 25)\)) ou em mais de 35 dias (\(P(X > 35)\)). Essas probabilidades são úteis para avaliar riscos e planejar cenários alternativos.</p>