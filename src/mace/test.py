'''
This script contains functions to apply a trained MACE model on a test dataset, 
i.e., test the trained model.

    - Test the model on 1 consecutive time step or on the evolution of the system.
    - Test the model on a full evolution of the system.

TODO: add plotting to test_model
TODO: check p loss during idn testing

Version 2.0
'''
from time                   import time
import numpy                as np
import matplotlib.pyplot    as plt
import torch
import math

import src.mace.CSE_0D.dataset  as ds
import src.mace.utils           as utils
import src.mace.loss            as loss
import src.mace.train           as train
from src.mace.CSE_0D.plotting   import plot_abs

## Im fairly certain I can make this outta recycled validation code
## fosho

def testModel(model, loader):
    '''
    Tests a model against given data.

    - mode:     The trained solver to be tested
    - loader:   Dataloader with test data

    Returns:
    - errors (idn, evol):   Mean unnormalized abundance errors 
    - times (au, evol):     Mean evaluation ime

    '''
    device = next(model.parameters()).device

    # Object to store errors and time (general)
    errors = {'idn':[],'evol':[]}
    times  = {'idn':[],'evol':[]}

    # Iterate over every entry
    with torch.no_grad():
        for n, p, dt in loader:
            n, p, dt = n[0].to(device), p[0].to(device), dt[0].to(device)

            # Identity replicaton
            n_i = n[:-1]
            p_i = p
            tic = time()
            n_i_hat, p_i_hat = model.autoencode(n_i, p_i) # TODO
            times['idn'].append(time()-tic)
            _,err_idn = utils.error(ds.get_abs(n_i), ds.get_abs(n_i_hat))
            errors['idn'].append(float(err_idn))

            # Evolution accuracy
            n0 = n[:1] # oh my god this stinkin bug took 2 hours
            p_seq  = p[None] # Add extra dimension (N) -> (1,N)
            dt_seq = dt[None]
            n_e = n[1:][None]
            tic=time()
            n_e_hat,_ = model.rollout(n0, p_seq, dt_seq)
            times['evol'].append(time()-tic)
            _,err_evol = utils.error(ds.get_abs(n_e[0]), ds.get_abs(n_e_hat[0]))
            errors['evol'].append(float(err_evol))

    errors = {name:(sum(val)/len(val)) for name,val in errors.items()}
    times = {name:(sum(val)/len(val)) for name,val in times.items()}

    return errors, times