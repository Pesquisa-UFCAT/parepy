---
layout: home
parent: Learning
nav_order: 7
has_children: true
has_toc: true
title: Reliability analysis of reinforced concrete frames subjected to post-construction settlements
---

<!--Don't delete this script-->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<!--Don't delete this script-->

<h1>Reliability analysis of reinforced concrete frames subjected to post-construction settlements</h1>

<h2>Análise de confiabilidade de pórticos em concreto armado sujeitos a recalques pós construção</h2>
<div class="authors">
    <p align="justify">
        <b>Wanderlei Malaquias Pereira Junior</b><sup>a</sup> 
        <a href="https://orcid.org/0000-0002-7404-3666" target="_blank">https://orcid.org/0000-0002-7404-3666</a><br>
        <b>Matheus Henrique Morato de Moraes</b><sup>b</sup> 
        <a href="https://orcid.org/0000-0002-7285-1344" target="_blank">https://orcid.org/0000-0002-7285-1344</a><br>
        <b>Daniel Lima de Araújo</b><sup>c</sup> 
        <a href="https://orcid.org/0000-0002-6802-4637" target="_blank">https://orcid.org/0000-0002-6802-4637</a><br>
        <b>Mauro Alexandre Paula de Sousa</b><sup>b</sup> 
        <a href="https://orcid.org/0000-0002-5798-8553" target="_blank">https://orcid.org/0000-0002-5798-8553</a><br>
        <b>Antover Panazzolo Sarmento</b><sup>c</sup> 
        <a href="https://orcid.org/0000-0002-8533-1575" target="_blank">https://orcid.org/0000-0002-8533-1575</a><br>
        <b>André Teófilo Beck</b><sup>d</sup> 
        <a href="https://orcid.org/0000-0003-4127-5337" target="_blank">https://orcid.org/0000-0003-4127-5337</a>
    </p>
</div>
<div class="affiliations">
    <p align="justify">
        <sup>a</sup>Federal University of Catalão - UFCAT, Engineering College, Department of Civil Engineering, Catalão, Goiás, Brasil<br>
        <sup>b</sup>Federal University of São Carlos - UFSCar, Civil Engineering Graduate Program, Department of Civil Engineering, São Carlos, São Paulo, Brasil<br>
        <sup>c</sup>Federal University of Goiás - UFG, Department of Civil Engineering, Goiânia, Goiás, Brasil<br>
        <sup>d</sup>University of São Paulo - USP, Engineering School, Department of Engineering Structures, São Carlos, São Paulo, Brasil
    </p>
</div>

<p align="justify"><b>Abstract:</b> Most papers found in the literature address reliability analysis of isolated elements, like beams and columns. Yet, symmetry and regularity are often exploited in the construction of regular RC frames, resulting in the same or similar designs for all columns of a floor, and for all beams of a building. This leads to significant differences in member reliability, due to different axial load to bending moment ratios, in different parts of the structure. Moreover, load effects increase and symmetry is lost under individual support settlements. In this scenario, reliability analyses are performed, for an intact 4-story 4-bays RC frame; and considering different settlement conditions. Monte Carlo simulation is performed, considering uncertainties in dead and life loading, and steel and concrete strengths. The results show that a settlement of 5 mm, corresponding to an angular distortion of 1/1000, reduced the average reliability of the frame by only 7%, just the same, it reduced the reliability index of several cross-sections of the beams to up to 2.65,
</p>

<p align="justify"><b>Keywords:</b> reinforced concrete, reliability analysis, Monte Carlo simulation, foundation settlements.

<p align="justify"><b>Resumo:</b> A maior parte dos artigos encontrados na literatura endereça a confiabilidade de elementos isolados, como vigas e colunas. No entanto, simetria e regularidade são frequentemente exploradas na construção de pórticos regulares de concreto armado, o que resulta em projetos iguais ou semelhantes para todas as colunas de um andar, ou para todas as vigas de um prédio. Isto leva a diferenças na confiabilidade dos elementos, em função das diferentes razões entre carga axial e momento fletor, em diferentes partes da estrutura. Mais ainda, há um aumento dos esforços solicitantes e perda de simetria na presença de recalques de apoios. Neste cenário são realizadas análises de confiabilidade para um pórtico de 4 andares e 4 vãos, na situação intacta e considerando diferentes recalques de apoio. É realizada simulação de Monte Carlo, considerando incertezas nas ações permanentes e de utilização, e na resistência do aço e do concreto. 
<br><br>
Os resultados mostram que um recalque de 5 mm, correspondente a uma distorção angular de 1/1000, reduziu a confiabilidade média do pórtico em apenas 7%, contudo reduziu o índice de confiabilidade de várias seções transversais das vigas para até 2.65, valor inferior ao recomendado no Model Code 2010. Conclui-se que a metodologia utilizada neste artigo apresenta-se como uma ferramenta importante para a análise de eventos não previstos em projeto, auxiliando na tomada de decisão sobre a necessidade de intervenção nas estruturas.
<br><br>
<b>Palavras-chave:</b> concreto armado, confiabilidade estrutural, simulação de Monte Carlo, recalque de fundações. </p>

<h3>How to cite:</h3>
    <p align="justify">
        W. M. Pereira Junior, M. H. M. Moraes, D. L. Araújo, M. A. P. Sousa, A. P. Sarmento, A. T. Beck: 
        “Reliability analysis of reinforced concrete reticulated structures subjected to post-construction settlements” 
        <i>Rev. IBRACON Estrut. Mater.</i>, vol. <span class="red-text">xx</span>, no. <span class="red-text">xx</span>, 
        elocation, year, <a href="https://doi.org/" target="_blank" class="red-text">https://doi.org/...</a> <br>
        <span class="red-text">[Items in red will be filled in later by the journal staff]</span>
    </p>

<p align="justify"><b>Corresponding author:</b> First Name Surname. E-mail: <a href="mailto:email@email.com">email@email.com</a></p>
<p align="justify"><b>Financial support:</b> None.</p>
<p align="justify"><b>Conflict of interest:</b> Nothing to declare.</p>

<h2>1. Introduction</h2>

<p align="justify">In the structural engineering context, uncertainties are related to the inability to predict some characteristics of the structural system, such as loads, material properties, and assumptions of the structural model adopted [1], [2]. Despite these uncertainties, design codes employ design methodologies to obtain a resistant, safe and robust structure.
<br><br>
Although the design of structures considers these uncertainties in the design variables, it is still possible that a structure will be exposed to a condition not foreseen in the design [3]. Given these new conditions in the service situation, it is necessary to evaluate the new safety level of the structure, verifying the necessity of reinforcements or even the demolition of the system in more severe cases.
<br><br>
In terms of assessing the safety level of an existing structure, reliability theory can be employed for this purpose. Some authors have dedicated themselves to studying and evaluating the safety level of existing structures using such a theory. Works such as Facholli and Beck [4] and Beck et al. [5] have employed reliability theory to evaluate the safety level in structural element loss events. Küttenbaum et al.[6], Mankar et al. [7] and Souza et al. [8] evaluated the variations of the mechanical properties of materials and their impact on the system’s reliability in situations of structure use. In Ávilla et al. [9] reliability analysis was applied to verify the safety level of historic buildings in regions susceptible to earthquakes.
<br><br>
In terms of structural design, many engineers still design structures without considering the effects of settlement [10]. Amancio [11] states that such a condition often occurs since predicting settlements in structures is still a complex factor due to the difficulty of obtaining parameters such as soil strength and deformability. Thus, this paper aims to contribute to the soil-structure interaction theme by developing a conceptual study to verify the influence of settlements on the safety of reinforced concrete reticulated structures. Therefore, this work intends to develop an analysis methodology that can contribute to decision-making regarding the maintenance of reinforced concrete structures of multiple floors submitted to actions arising from foundation settlement.
<br><br>
This paper is divided into six sections. The first three sections introduce the initial concepts of beam design and structural reliability. Section 4 discusses the conceptual problem of a frame structure submitted to settlement conditions. Sections 5 and 6 present the results and conclusions about this research.
</p>

<h2>2. DESIGN OF BEAMS UNDER PURE BENDING</h2>

<p align="justify">This section presents the concepts related to structural analysis and the format of the limit state equation. The normative used were the Brazilian standards NBR 8681 [12] and NBR 6118 [13].
<br><br>
The rectangular beams considered in this work are subject to pure bending caused by the action of gravitational loads. Equation (1) characterizes the ultimate limit state (g) of the beam by the exhaustion of the resistant capacity of the cross-section at stage III due to normal loads, as defined by NBR 6118 [13]. 
</p>

$$
\begin{align*}
    g(f_y, f_c, D, L) = E_R \cdot M_R(f_y, f_c) - M_S(D, L, f'_c) \quad{(1)} \\
    M_R = A_S \cdot f_y \cdot (d - \frac{\lambda}{2} \cdot x) \quad{(2)} \\
    x = \frac{A_S \cdot f_y}{f_c \cdot b_w \cdot \alpha_c \cdot \lambda} \quad{(3)}
\end{align*}
$$

<p align="justify">The requesting moment $M_S$ indicates the maximum demand of bending moment on the cross-section, and in the case of this work, it was determined using a linear analysis. The resistant moment $M_R$ indicates the resistant capacity of the cross-section of the reinforced concrete (RC) beam. $A_s$ represents the steel’s area of the section, $\lambda$ and $\alpha_c$ are factors that depend on the characteristic compressive strength of concrete ($f_c$). These factors can be consulted in section 17.2.2 of NBR 6118 [13]. $f_y$ represents the yield strength of the passive reinforcement steel used. $d$ and $b_w$ represent the effective height and width of the RC section. $E_R$ represents model error variable for beam bending.
</p>