---
layout: home
parent: Learning
nav_order: 1
has_children: true
has_toc: true
title: Uniform Distributions
---

<!--Don't delete this script-->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h1>Uniform Distribution</h1>

<p align="justify">
The uniform distribution is a continuous probability distribution that describes events where all values within a given range are equally likely. Each of the <i>n</i> possible values has the same probability of occurring (<i>1/n</i>). The distribution is characterized by two parameters: the lower limit (<i>a</i>) and the upper limit (<i>b</i>), which define the range of possible outcomes. The probability density function (PDF) is constant within the interval <i>[a, b]</i> and zero outside this range <a href="#ref1">[1]</a>.
</p>

<h3>Probability Density Function (PDF)</h3>

<p align="justify">
The probability density function (PDF) of a uniform distribution provides the likelihood of any given value within the range \([a, b]\). Mathematically, the PDF is expressed in Equation (1), as defined by Ross <a href="#ref2">[2]</a>.
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            f(x) =
            \begin{cases} 
            \frac{1}{b - a}, & \text{if } a \leq x \leq b \\
            0, & \text{otherwise}
            \end{cases}
            \]
        </td>
        <td style="width: 10%;">
            <p align="right" id="eq1">(1)</p>
        </td>
    </tr>
</table>

<h3>Cumulative Distribution Function (CDF)</h3>

<p align="justify">
The cumulative distribution function (CDF) of the uniform distribution, denoted as <i>F(x)</i>, describes the cumulative probability up to a value <i>x</i>. The CDF is particularly useful in determining the probability that a random variable will fall within a specific range <a href="#ref2">[2]</a>. It is defined in Equation (2).
</p>

<table style="width:100%">
    <tr>
        <td style="width: 90%;">
            \[
            F(x) =
            \begin{cases}
            0, & \text{if } x < a, \\
            \frac{x - a}{b - a}, & \text{if } a \leq x \leq b, \\
            1, & \text{if } x > b.
            \end{cases}
            \]
        </td>
        <td style="width: 10%;">
            <p align="right" id="eq2">(2)</p>
        </td>
    </tr>
</table>

<h3>Applications in Engineering</h3>

<p align="justify">
Uniform distributions frequently appear in engineering and physical sciences, modeling situations where outcomes are equally likely across a defined range. They are also discussed extensively in the context of probability and statistics, providing a foundational understanding for more advanced distributions (<a href=\"#ref1\">[1]</a>). Examples include modeling uncertainties in initial design parameters or in simulations where random sampling from a specific interval is needed (<a href="#ref3">[3]</a>).
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
            <td><p align="left"><a href="https://www.sciencedirect.com/book/9780123948113/introduction-to-probability-and-statistics-for-engineers-and-scientists" target="_blank" rel="noopener noreferrer">Ross, S. M. (2014). <i>Introduction to Probability and Statistics for Engineers and Scientists</i>. Fifth Edition. Academic Press.</a></p></td>
        </tr>
        <tr>
            <td><p align="center" id="ref2">[2]</p></td>
            <td><p align="left"><a href="https://www.pearson.com/en-us/subject-catalog/p/first-course-in-probability-a/P200000006334/9780137504589" target="_blank" rel="noopener noreferrer">Ross, S. M. (2014). <i>A First Course in Probability</i>. Tenth Edition. Pearson Education.</a></p></td>
        </tr>
        <tr>
            <td><p align="center" id="ref3">[3]</p></td>
            <td><p align="left"><a href="https://www.sciencedirect.com/book/9780123756862/introduction-to-probability-models" target="_blank" rel="noopener noreferrer">Ross, S. M. (2010). <i>Introduction to Probability Models</i>. Tenth Edition. Academic Press.</a></p></td>
        </tr>
    </tbody>
</table>
