---
layout: home
parent: Reliability analysis of reinforced concrete frames subjected to post-construction settlements
nav_order: 7
has_children: true
has_toc: true
title: Time-dependent reliability
---

<!--Don't delete this script-->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h1>Time-dependent reliability: First barrier failure</h1>

<p align="justify">Time-variant reliability analysis of uncertain structures involves evaluating the probability that a vector random load process S(t) exceeds the uncertain or random resistance R(t) of a structure or structural component at any time during the structure's life [15]. Equation (1) is the fundamental equation for time-dependent reliability problems. Where tD is the design life of the structure. </p>

$$
\begin{align*}
p_f(t_D) = P[min g(R, S, t) \leq 0] \quad{(1)} \\
\textrm{if} \quad 0 \leq f \leq t_D
\end{align*}
$$

<p align="justify">Using the Crude Monte Carlo process with a solver, the first step is a time discretization, carried out of the time interval [\(0, t_D\)] in \(N\) instants, where \(t_D\) is the final time. The second step evaluates \(g(R, S, t)\) in discrete time. The first out-crossing event, i.e., \(g(R, S, t) \quad 0\), defines fail status in structure, where \(t_f\) corresponds to a discrete-time where the first out-crossing event occurs. Equation (2) defines the limit state equation status.</p>

$$
\begin{cases}
g(R, S, t) > 0 R(t), \quad \text{ if} \quad 0 \leq t \leq t_f \\
g(R, S, t) \leq 0, \quad \text{ if} \quad t_f \leq t \leq t_D \quad{(2)}
\end{cases}
$$

<p align="justify">Therefore, using the Monte Carlo algorithm, the probability of failure rate is estimated through Equation (3):</p>

$$
p_f(0, t_i) \approx p_f(0, t_i) = \frac{k_j}{n_{si}} \quad{(3)}
$$

<p align="justify">Where \(k_j\) is the number of barrier out-crossing in step ti and nsi is the number of samples. Figure 1 shows a simplified example of the first barrier failure.</p>

<center>
    <img src="assets/images/reliability2_001.png" height="auto">
    <p align="center"><b>Figure 1.</b> First barrier failure example.</p>
</center>

<p align="justify">The reliability index β is a geometric measure of the probability of failure [15]. The reliability index β index can be obtained numerically by solving the equation (4).</p>

$$
\beta = \Phi^{-1}(1 - p_f) \quad{(4)}
$$