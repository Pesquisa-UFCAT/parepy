---
layout: home
nav_order: 2
has_children: false
has_toc: false
title: Normal
parent: Distributions
grand_parent: Learning
---

<!--Don't delete this script-->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h1>Normal Distribution</h1>

<p align="justify">
The normal or Gaussian distribution is one of the most important probability distributions, widely used in statistics, engineering, and applied sciences. As noted by Montgomery and Runger in <a href="#ref1">[1]</a>, it is symmetric around its mean, with higher probabilities for values near the mean and decreasing probabilities for extreme values. Wasserman <a href="#ref2">[2]</a> emphasizes its relevance as a foundation for probabilistic models and statistical inference.
</p>

<h3>Definitions</h3>

<p align="justify">
The normal distribution is characterized by two parameters:
</p>

<ul>
    <li>\(\mu\): the mean, which represents the central value of the distribution.</li>
    <li>\(\sigma\): the standard deviation, which quantifies the spread of the distribution around the mean.</li>
</ul>

<p align="justify">
The variance, \(\sigma^2\), is the square of the standard deviation. The probability density function (PDF) and the cumulative distribution function (CDF) are essential for describing the normal distribution's behavior.
</p>

<h3>Probability Density Function (PDF)</h3>

<p align="justify">
The probability density function (PDF) of the normal distribution quantifies the likelihood of a random variable, \(X\), taking on a specific value within its domain. The PDF is defined in <a href="#eq1">Equation 1</a> as follows:
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right),
            \quad \text{for } -\infty < x < \infty.
            \]
        </td>
        <td style="width: 10%;">
            <p align="right" id="eq1">(1)</p>
        </td>
    </tr>
</table>

<h3>Standard Normal Distribution and Transformations</h3>

<p align="justify">
The cumulative distribution function (CDF) of the normal distribution, as shown in <a href="#eq2">Equation 2</a>, often lacks a closed-form solution. Instead, results are typically presented in terms of a standard normal distribution, which has a mean of 0 and a standard deviation of 1. Any random variable \(X \sim N(\mu, \sigma^2)\) can be transformed into a standard normal variable \(Y \sim N(0, 1)\) using the formula shown in <a href="#eq3">Equation 3</a>:
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            F(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi \sigma^2}} 
            \exp\left(-\frac{(t - \mu)^2}{2\sigma^2}\right) dt,
            \]
        </td>
        <td style="width: 10%;">
            <p align="right" id="eq2">(2)</p>
        </td>
    </tr>
</table>

<p align="justify">The transformation formula is:</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            Y = \frac{X - \mu}{\sigma}.
            \]
        </td>
        <td style="width: 10%;">
            <p align="right" id="eq3">(3)</p>
        </td>
    </tr>
</table>

<p align="justify">
For the standard normal variable \(Y\), the probability density function \(\phi(y)\) and the cumulative distribution function \(\Phi(y)\) are defined as follows:
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            \begin{align*}
            \phi(y) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{y^2}{2}\right), \quad \text{for } -\infty < y < \infty, \\
            \Phi(y) = \int_{-\infty}^{y} \phi(z) dz, \quad \text{for } -\infty < y < \infty.
            \end{align*}
            \]
        </td>
    </tr>
</table>

<p align="justify">
The PDF \(f(x)\) for a variable \(X \sim N(\mu, \sigma^2)\) can be expressed in terms of \(\phi(y)\) as follows:
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            f(x) = \phi\left(\frac{x - \mu}{\sigma}\right).
            \]
        </td>
    </tr>
</table>

<h3>Probability Intervals</h3>

<p align="justify">
Some useful results concerning a normal distribution are summarized below. These results, shown in <a href="#eq4">Equation 4</a>, define the probabilities of a normal random variable falling within specific intervals of the mean and standard deviation <a href="#ref2">[2]</a>:
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            \begin{align*}
            P(\mu - \sigma < X < \mu + \sigma) &= 0.6827, \\
            P(\mu - 2\sigma < X < \mu + 2\sigma) &= 0.9545, \\
            P(\mu - 3\sigma < X < \mu + 3\sigma) &= 0.9973.
            \end{align*}
            \]
        </td>
        <td style="width: 10%;">
            <p align="right" id="eq4">(4)</p>
        </td>
    </tr>
</table>

<p align="justify">
Additionally, due to the symmetry of \(f(x)\), we have:
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            P(X > \mu) = P(X < \mu) = 0.5.
            \]
        </td>
    </tr>
</table>

<p align="justify">
This demonstrates that for a normal distribution, approximately 68% of values fall within one standard deviation of the mean, 95% within two, and 99.7% within three, which is known as the empirical rule or the 68-95-99.7 rule <a href="#ref2">[2]</a>.
</p>

<h3>Confidence Intervals and Applications</h3>

<p align="justify">
The normal distribution is often used to model errors or deviations in manufacturing or production processes. Confidence intervals are defined in terms of the factor \(k\), which represents the number of standard deviations from the mean. For a confidence interval of \(k\) standard deviations, we have:
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            P(x_{\text{inf}} < x < x_{\text{sup}}) = P[\mu - k\sigma < x < \mu + k\sigma] = \int_{-k}^{k} \phi(y) dy = \Phi(k) - \Phi(-k).
            \]
        </td>
    </tr>
</table>

<p align="justify">
The limits \(x_{\text{inf}}\) and \(x_{\text{sup}}\) are used as filters in quality control. For a confidence level of 95.5%, for example, components with dimensions \(x_i < \mu - 2\sigma\) or \(x_i > \mu + 2\sigma\) are considered out of specification. This helps prevent excessive variations from compromising the final product's quality.
</p>

<h3>Applications in Engineering</h3>

<p align="justify">
The normal distribution finds extensive applications in engineering, particularly in the fields of reliability analysis, quality control, and structural design. As highlighted by Choi et al. <a href="#ref3">[3]</a>, it is often used in modeling uncertainties in material properties, load capacities, and environmental conditions. 
</p>

<p align="justify">
Montgomery and Runger <a href="#ref1">[1]</a> emphasize its critical role in statistical process control, where control charts based on the normal distribution help monitor manufacturing processes and detect deviations from expected performance. In structural engineering, reliability assessments use the normal distribution to estimate the probability of failure under various loading conditions.
</p>

<p align="justify">
Wasserman <a href="#ref2">[2]</a> discusses the application of the normal distribution in machine learning and data analysis, particularly in Bayesian inference, where it serves as a key component of probabilistic models. These applications illustrate the versatility of the normal distribution in solving complex, real-world engineering problems.
</p>

<h3>References</h3>

<table>
    <thead>
        <tr>
            <th>ID</th>
            <th>Reference</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><p align="center" id="ref1">[1]</p></td>
            <td><p align="left"><a href="https://www.amazon.com/Douglas-Montgomery-George-Runger-Probability/dp/B004VG3ZT2" target="_blank" rel="noopener noreferrer">Montgomery, D. C., & Runger, G. C. (2011). <i>Applied Statistics and Probability for Engineers</i>. Fifth Edition. John Wiley & Sons.</a></p></td>
        </tr>
        <tr>
            <td><p align="center" id="ref2">[2]</p></td>
            <td><p align="left"><a href="https://link.springer.com/book/10.1007/978-1-4612-2560-7" target="_blank" rel="noopener noreferrer">Wasserman, L. (2004). <i>All of Statistics: A Concise Course in Statistical Inference</i>. Springer.</a></p></td>
        </tr>
        <tr>
            <td><p align="center" id="ref3">[3]</p></td>
            <td><p align="left"><a href="https://link.springer.com/book/10.1007/978-1-84628-445-8" target="_blank" rel="noopener noreferrer">Choi, S. K., Grandhi, R. V., & Canfield, R. A. (2007). <i>Reliability-based Structural Design</i>. Springer.</a></p></td>
        </tr>
    </tbody>
</table>
