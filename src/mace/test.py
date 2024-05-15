from time import time
import numpy as np
from tqdm import tqdm



def test_step(model, input):
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

    print('Solving time [s]:', solve_time)

    return n.detach().numpy(), n_hat, dt, mace_time



def test_evolution(model, input, start_idx=0):
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
    for i in tqdm(range(start_idx+1,len(dt))):
        tic = time()
        n_hat,z_hat, modstatus = model(n_hat.view(1, -1),p[i].view(1, -1),dt[i].view(-1))    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches
        toc = time()
        n_evol.append(n_hat.detach().numpy())
        solve_time = toc-tic
        mace_time.append(solve_time)
    toc_tot = time()

    print('Solving time [s]:', np.array(mace_time).sum())
    print('Total   time [s]:', toc_tot-tic_tot)

    return np.array(n_evol).reshape(-1,468), np.array(mace_time)


