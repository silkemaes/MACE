####
MACE 
####


*Welcome to the documentation of MACE!*

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   use/index
   example/index
    

*****
About
*****

**MACE, a Machine-learning Approach to Chemistry Emulation**, by `Maes et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract>`_, is a surrogate model for chemical kinetics. It is developed in the contexts of circumstellar envelopes (CSEs) of asymptotic giant branch (AGB) stars, i.e. evolved low-mass stars.

During development, the chemical models of `Maes et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4654M/abstract>`_ are used. In this paper you can also find more details about the astrochemical environment used.

MACE is implemented in Python and uses `PyTorch <https://pytorch.org/>`_, together with `torchode <https://github.com/martenlienen/torchode>`_ (`Lienen & Gunnemann, 2022 <https://openreview.net/pdf?id=uiKVKTiUYB0>`_), to be trained.


The architecture of MACE is schematically given as 

.. image:: images/MACE.png

MACE offers a surrogate model that emulates the evolution of chemical abundances over time in a dynamical physical environment. As the name states, it makes use of machine learning techniques. More specifically, combining an *autoencoder* (blue) and a *trainable ordinary differential equation (ODE)* (red) allows to accurately emulate a chemical kinetics model.

In formula, MACE is stated as

.. math:: 

    {\hat{\boldsymbol{n}}}(t) = \mathcal{D}\Big( G \big( \mathcal{E} ({\boldsymbol{n}}, {\boldsymbol{p}}),t \big) \Big).

Here, :math:`{\hat{\boldsymbol{n}}}(t)` are the predicted chemical abundances at a time $t$ later dan the initial state :math:`{\boldsymbol{n}}`. :math:`\mathcal{E}` and :math:`\mathcal{D}` represent the autoecoder, with the encoder and decoder, respectively. The autoencoder maps the chemical space :math:`{\boldsymbol{n}}` together with the physical space :math:`{\boldsymbol{p}}` to a lower dimensional representation :math:`\boldsymbol{z}`, called the latent space. The function $G$ describes the evolution in latent space such that :math:`\boldsymbol{z}(\Delta t) = G(\boldsymbol{z}, \Delta t)=\int_0^{\Delta t} g(\boldsymbol{z}){\rm d}t`.

For more details, check out our paper: `Maes et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract>`_.


.. note::
    Required packages:
    
    * torch
    * torchode
    * numpy
    * matplotlib
    * natsort
    * tqdm
    * scipy

*****
Acknowledgements
*****

The MACE architecture is free to use. Please cite our paper `Maes et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract>`_.

*****
Contact
*****

If any comments or issues come up, please contact `Silke Maes <https://github.com/silkemaes>`_, or set up an issue on `GitHub <https://github.com/silkemaes/MACE>`_.






  
