---
title: 'MACE: a Machine-learning Approach to Chemistry Emulation'
tags:
  - Python
  - astrophysics
  - chemistry
  - surrogate model
  - stellar winds
authors:
  - name: Silke Maes
    orcid: 0000-0003-4159-9964
    corresponding: true
    affiliation: 1
  - name: Frederik De Ceuster
    orcid: 0000-0001-5887-8498
    affiliation: 1
  - name: Marie Van de Sande
    orcid: 0000-0001-9298-6265
    affiliation: "2,3"
  - name: Leen Decin
    orcid: 0000-0002-5342-8612
    affiliation: "1,4"

affiliations:
 - name: Institute of Astronomy, KU Leuven, Celestijnenlaan 200D, B-3001 Leuven, Belgium
   index: 1
 - name: Leiden Observatory, Leiden University, PO Box 9513, 2300 RA Leiden, The Netherlands
   index: 2
 - name: School of Physics and Astronomy, University of Leeds, Leeds LS2 9JT, United Kingdom
   index: 3
 - name: School of Chemistry, University of Leeds, Leeds LS2 9JT, United Kingdom
   index: 4
date: 16 May 2024
bibliography: paper.bib


aas-doi: 10.3847/1538-4357/ad47a1
aas-journal: Astrophysical Journal 
---

# Summary
Astrochemistry is the study of chemical species in astrophysical objects. By combining observations and theoretical predictions, the physical conditions of the object can be analysed as well as its chemical composition and yields, since chemistry is closely coupled to the dynamics of the environment. Most often, such astrophysical environments are known to be complex and out of thermodynamic equilibrium. For that reason, the chemical evolution is usually modelled using the chemical kinetics approach, in which a set of non-linear coupled ordinary different equations (ODEs) is solved for a given network of chemical species and reactions. For large chemical networks, however, this method is computationally slow. Therefore, we developed MACE, *a Machine-learning Approach to Chemistry Emulation*, inspired on similar work in the literature (e.g., @Holdship2021; @Grassi2022; @Sulzer2023). MACE is a PyTorch module that offers a trainable surrogate model that is able to emulate chemical kinetics in a shorter computation time than its classical analogue. More specifically, it provides a machine learning architecture, consisting of an autoencoder and a trainable ODE, implemented in PyTorch and [torchode](https://github.com/martenlienen/torchode) [@torchode]. Mathematically, MACE is give by 

$$ \hat{\textbf{\textit{n}}}(t) = \mathcal{D}\Big( G \big( \mathcal{E} (\\textit{\textbf{n}}, \\textit{\textbf{p}}),t \big) \Big), $$

where $\hat{\boldsymbol{n}}(t)$ represents the predicted chemical abundances at time $t$. A schematic representation of the architecture and flow of the emulator is shown in Fig. \ref{fig:MACE}. MACE is developed in the context of circumstellar envelopes (CSEs) of evolved stars, but is flexible to be retrained on data of other astrophysical environments.

![Schematic representation of the architecture of MACE. The symbols $\boldsymbol{n}$ and $\boldsymbol{p}$ represent the chemical abundances and physical parameters, respectively. The autoencoder (in blue) consists of an encoder $\mathcal{E}$, able to map $\boldsymbol{n}$ and $\boldsymbol{p}$ to a latent space representation $\boldsymbol{z}$, and decoder $\mathcal{D}$, working the other way around. The function $g$ represents the trainable ODE, which needs to be solved for a time step $\Delta t$. (Adapted from @Maes2024.) \label{fig:MACE}](MACE.png)


# Statement of need
Astrophysical objects or environments entail 3D dynamical processes, as well as chemical and radiation processes. Thus, in order to properly model such an environment, these three constituents all need to be included in a single simulation. However, modelling each constituents separately already poses a computational challenge for a modest parameter space, let alone running a simulation with the three constituents integrated, and on an extended parameter space. Specifically on the chemistry side, it is generally established that, from a mathematical point of view, chemical kinetics is a stiff problem. When working with an extended chemical network ($\gtrsim 100$ species, connected by $\gtrsim 1000$ reactions), the system becomes computationally very slow to solve. Therefore, it is essential to speed up the solver in an alternative way, also because the solver is applied multiple times when calculating the chemical evolution of a certain environment. A fast and accurate chemistry solver would make it computationally feasible to run it alongside a 3D hydrodynamical and/or radiation model.


# Example case
In the context of circumstellar envelopes, we aim to build a comprehensive model, consisting of 3D hydrodynamics, chemistry, and radiation. This is needed since recently, complex structure (such as spiral, arcs, and bipolarity) are observed in CSEs around evolved, low- and intermediate mass stars, currently attributed to the presence of (sub)stellar companions [@Nordhaus2006;@Decin2020;@Gottlieb2022]. In order to study these stars and their CSEs, taking into account companions, accurate 3-dimensional models are needed. Currently, only hydrodynamical modelling is done for these systems in 3D (e.g., @Maes2021; @Malfait2021), though the aim is to include chemistry as well as radiation [@Maes2022]. Starting with integrating chemistry and 3D hydrodynamics, this coupling will only be computationally feasible when a faster alternative to chemical kinetics exists. 

The development of MACE is a first step towards this coupling. Using the MACE architecture, we are able to reproduce the 1D abundance profiles of 468 chemical species with an accuracy between 10 and 18, averaged over 3000 tests, for MACE models with different sets of hyperparameters, for the following error metric:

$$ {\rm error} = \frac{\log_{10}\boldsymbol{n}-\log_{10}\hat{\boldsymbol{n}}}{\log_{10}\boldsymbol{n}}, $$

which is executed element-wise and subsequently summed over the different chemical species. More details on the accuracy of the MACE models can be found in @Maes2024. Fig. \ref{fig:int4} shows the abundance profiles of seven chemical species, where the full curves indicate the MACE test of model *int4* from @Maes2024, and the dashed curves give the result for the classical model. On average, the MACE routine provides a speed-up of a factor 26, not taking into account the potential extra speed-up factor due to efficient vectorisation when coupling MACE with an SPH hydrodynamical model.


![Chemical abundance profiles from a test of trained MACE model model *int4* (full curves), compared to the classical model (dashed curves). More details in @Maes2024. \label{fig:int4}](int4_example.png){ width=7x0% }

# Code availability
The code of MACE is publicly available in a GitHub repository: [https://github.com/silkemaes/MACE](https://github.com/silkemaes/MACE).


# Acknowledgements
S.M. and L.D. acknowledge support from the Research Foundation Flanders (FWO) grant G099720N. F.D.C. is a Postdoctoral Research Fellow of the Research Foundation - Flanders (FWO), grant number 1253223N, and was previously supported for this research by a Postdoctoral Mandate (PDM) from KU Leuven, grant number PDMT2/21/066. M.V.d.S. acknowledges support from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement No 882991 and the Oort Fellowship at Leiden Observatory. L.D. also acknowledges support from KU Leuven C1 MAESTRO grant C16/17/007, KU Leuven C1 BRAVE grant C16/23/009, and KU Leuven Methusalem grant METH24/012.


# References