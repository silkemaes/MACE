'''
This script contains functions to apply a trained MACE model on a test dataset, 
i.e., test the trained model.

    - Test the model on 1 consecutive time step or on the evolution of the system.
    - Test the model on a full evolution of the system.
'''


from time                   import time
import numpy                as np
import matplotlib.pyplot    as plt

import src.mace.CSE_0D.dataset  as ds
import src.mace.utils           as utils
from src.mace.CSE_0D.plotting   import plot_abs



def test_step(model, input, printing = True):
    '''
    Function to test a trained MACE model on 1 consecutive time step.

    Input
        - model = trained MACE model
        - input = input data to test the model
            - [0] = n: abundances
            - [1] = p: physical parameters
            - [2] = dt: time steps

    Method:
        1. pass the input through the model
        2. calculate the solving time

    Returns:
        - true abundances
        - predicted abundances
        - time steps
        - time to solve the model
    '''

    mace_time = list()
    if printing == True:
        print('>>> Testing step...')

    model.eval()
    n     = input[0]
    p     = input[1]
    dt    = input[2]

    # print(n.shape, p.shape,dt.shape)
    
    tic = time()
    n_hat, z_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 
    toc = time()

    solve_time = toc-tic
    mace_time.append(solve_time)

    ### Include initial abundances in output mace 
    n0 = n[0].view(1,-1).detach().numpy()
    n_hat = np.concatenate((n0,n_hat[0].detach().numpy()),axis=0)

    if printing == True:
        print('Solving time [s]:', solve_time)

    return n.detach().numpy(), n_hat, dt, mace_time



def test_evolution(model, input, printing = True, start_idx=0):
    '''
    Function to test the evolution of a MACE model.
    
    Input:
        - model     = trained MACE model
        - input     = input data to test the model
            - [0] = n: abundances
            - [1] = p: physical parameters
            - [2] = dt: time steps
        - start_idx = index to start the evolution, default = 0
    
    Method:
        1. pass the input through the model
        2. use the resulting predicted abundances (n_hat) as input for the next time step
        3. store the calculation time

    Returns:
        - predicted abundances
        - calculation time
    '''
    if printing == True:
        print('\n>>> Testing evolution...')

    model.eval()
    n     = input[0][start_idx]
    p     = input[1]
    dt    = input[2]

    mace_time = list()
    n_evol = list(n.view(1,1,1,-1).detach().numpy())


    tic_tot = time()

    ## first step of the evolution
    tic = time()
    n_hat, z_hat,modstatus = model(n.view(1, -1),p[start_idx].view(1, -1),dt[start_idx].view(-1))    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 
    toc = time()
    n_evol.append(n_hat.detach().numpy())
    solve_time = toc-tic
    mace_time.append(solve_time)

    
    ## subsequent steps of the evolution
    for i in range(start_idx+1,len(dt)):
        tic = time()
        n_hat,z_hat, modstatus = model(n_hat.view(1, -1),p[i].view(1, -1),dt[i].view(-1))    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches
        toc = time()
        n_evol.append(n_hat.detach().numpy())
        solve_time = toc-tic
        mace_time.append(solve_time)
    toc_tot = time()

    if printing == True:
        print('Solving time [s]:', np.array(mace_time).sum())
        print('Total   time [s]:', toc_tot-tic_tot)

    return np.array(n_evol).reshape(-1,468), np.array(mace_time)


def test_model(model, testpath, meta, specs=[], printing = True, plotting = False, save = False):
    '''
    Test the model on a test set.

    Input:
        - testpath: path to the test data
        - plotting: plot the results, default = False
    '''

    model1D, input, info = ds.get_test_data(testpath, meta)
    id = info['path'] +'_'+ info['name']

    n, n_hat, t, step_time = test_step(model, input, printing = printing)
    n_evol, evol_time  = test_evolution(model, input, start_idx=0, printing = printing)

    if printing == True:
        print('\n>>> Denormalising... ')
    n = ds.get_abs(n)
    n_hat = ds.get_abs(n_hat)
    n_evol = ds.get_abs(n_evol)

    err, err_test = utils.error(n, n_hat)
    err, err_evol = utils.error(n, n_evol)

    if plotting == True:
        print('\nErrors (following Eq. 23 of Maes et al., 2024):')
        print('      Step error:', np.round(err_test,3))
        print(' Evolution error:', np.round(err_evol,3))

        print('\n>>> Plotting...')

        if len(specs) == 0:
            print('No species specified, using a default list:')
            print('     CO, H2O, OH, C2H2, C2H, CH3C5NH+, C10H2+')
            specs = ['CO', 'H2O','OH',  'C2H2',  'C2H', 'CH3C5NH+', 'C10H2+']

        ## plotting results for the step test
        fig_step = plot_abs(model1D, n, n_hat, specs=specs, step = True)
        if save == True:
            plt.savefig(model.plotpath+'step_'+model.epoch+'_'+id+'.png', dpi=300)
            print('Step test plot saved at:', model.plotpath+'step_'+model.epoch+'_'+id+'.png')
        
        ## plotting results for the evolution test
        fig_evol = plot_abs(model1D, n, n_evol, specs=specs)
        if save == True:
            plt.savefig(model.plotpath+'evol_'+model.epoch+'_'+id+'.png', dpi=300)
            print('Evolution test plot saved at:', model.plotpath+'evol_'+model.epoch+'_'+id+'.png')

        plt.show()

    return err_test, err_evol, step_time, np.sum(evol_time)
            