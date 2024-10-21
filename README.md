# MACE

Welcome to the MACE repository!

***MACE, a Machine-learning Approach to Chemistry Emulation***, by [Maes et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract), is a surrogate model for chemical kinetics. It is developed in the contexts of circumstellar envelopes (CSEs) of asymptotic giant branch (AGB) stars, i.e. evolved low-mass stars. 

During development, the chemical models of [Maes et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4654M/abstract) are used. In this paper you can also find more details about the astrochemical environment used.

MACE is implemented in Python and is trained using [PyTorch](https://pytorch.org/), together with [torchode](https://github.com/martenlienen/torchode) [(Lienen & Gunnemann, 2022)](https://openreview.net/pdf?id=uiKVKTiUYB0).

---
## Table of content
- [Installation](#inst)
- [What is MACE?](#what)
- [How to use?](#use)
- [Example case](#exmp)
- [Contact](#cont)
- [Acknowledgements](#ackn)

---
## Notes on installation <a name="inst"></a>
- MACE is currently not available as a package on ```pypi```. There is a package named ```mace```, but it is not this one.
- To use MACE, please clone the repo and install the required packages, see ```requirements.txt```:
```
git clone https://github.com/silkemaes/MACE.git
```

---
## What is MACE? <a name="what"></a>

MACE offers a surrogate model that emulates the evolution of chemical abundances over time in a dynamical physical environment. As the name states, it makes use of machine-learning techniques. More specifically, combining an *autoencoder* (blue) and a *trainable ordinary differential equation (ODE)* (red) allows to accurately emulate a chemical kinetics model.

Hence, MACE is a framework, an architecture, that can be trained for specific chemical datasets, but before using, should be made compatible with the dataset, see _[How to use?](#use)_.

The architecture of MACE is schematically given as 
![MACE architecture](MACE.png)

MACE offers a surrogate model that emulates the evolution of chemical abundances over time in a dynamical physical environment. As the name states, it makes use of machine-learning techniques. More specifically, combining an *autoencoder* (blue) and a *trainable ordinary differential equation (ODE)* (red) allows to accurately emulate a chemical kinetics model.

In formula, MACE is stated as

$$
{\hat{\boldsymbol{n}}}(t) = \mathcal{D}\Big( G \big( \mathcal{E} ({\boldsymbol{n}}, {\boldsymbol{p}}),t \big) \Big).
$$

Here, ${\hat{\boldsymbol{n}}}(t)$ are the predicted chemical abundances at a time $t$ later dan the initial state ${\boldsymbol{n_0}}$. $\mathcal{E}$ and $\mathcal{D}$ represent the autoecoder, with the encoder and decoder, respectively. The autoencoder maps the chemical space ${\boldsymbol{n_0}}$ together with the physical space ${\boldsymbol{p}}$ to a lower dimensional representation $\boldsymbol{z}$, called the latent space. The function $G$ describes the evolution in latent space such that $\boldsymbol{z}(\Delta t) = G(\boldsymbol{z}, \Delta t)=\int_0^{\Delta t} g(\boldsymbol{z}){\rm d}t$.

For more details, check out our paper: [Maes et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract).

---
## How to use?  <a name="use"></a>

The script ```routine.py``` gives the flow of training & storing a MACE architecture, and immediately applies to the specified test dataset once training is finished. As such, it returns an averaged error on the MACE model compared to the classical model. More info on the training routine can be found in the [paper](https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract). 

An annotated notebook of the routine can be found in the [documentation](https://mace-code.readthedocs.io/en/latest/use/index.html).

The script ```routine.py``` takes an input file with the needed (hyper)parameter setup. An example of such an input file can be found in input/.
```
python routine.py example
```

***Disclaimer:***

In order to train MACE with a certain chemical dataset, the ```Dataset``` class
should be made compatible with that data. Currently, the script ```src/mace/CSE_0D/dataset.py``` works only for the specific dataset used here, i.e. models from [Maes et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4654M/abstract), using the [Rate22-CSE code](https://github.com/MarieVdS/rate22_cse_code).


---
## Example case <a name="exmp"></a>

This repository contains a trained MACE model as a test case, see ```model/20240604_160152```. 

The code for loading a trained MACE model can be found in the script ```src/mace/load.py```, testing in ```src/mace/test.py```. An annotated notebook can be found in the [documentation](https://mace-code.readthedocs.io/en/latest/example/index.html).

---
## Contact <a name="cont"></a>

If any comments or issues come up, please contact me via [email](mailto:silke.maes@protonmail.com), or set up a GitHub issue.

---
## Acknowledgements <a name="ackn"></a>

The MACE architecture is free to use. Please cite our paper [Maes et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract).


  


  
