How to use MACE
################


The script :literal:`routine.py` gives the flow of training & storing a MACE architecture, and immediately applies to the specified test dataset once training is finished. As such, it returns an averaged error on the MACE model compared to the classical model. More info on the training routine can be found in the `paper <https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract>`_. 

The script :literal:`routine.py` takes an input file with the needed (hyper)parameter setup. An example of such an input file can be found in input/.
.. code-block:: shell

    python routine.py example


***Disclaimer:***

In order to train MACE with a certain chemical dataset, the :literal:`Dataset` class
should be made compatible with that data. Currently, the script :literal:`src/mace/CSE_0D/dataset.py` works only for the specific dataset used here, i.e. models from [Maes et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023MNRAS.522.4654M/abstract), using the [Rate22-CSE code](https://github.com/MarieVdS/rate22_cse_code).

Tutorial of the routine:

.. toctree::
   :maxdepth: 1

   routine.ipynb

.. note::
    Required packages:
    
    * torch
    * torchode
    * numpy
    * matplotlib
    * natsort
    * tqdm


