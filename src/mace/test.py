from time import time
import numpy as np
from tqdm import tqdm

import mace.CSE_0D.dataset as ds

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
    print('>>> Testing model...')

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
    print('\n>>> Testing model...')


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


def test_model(model, path, meta, specs, plots_path, epoch, save = False, plotting = False):


    mods = list()


    # for i, testpath in enumerate(testpaths):
    #     print(i,testpath)
    model1D, input, info = ds.get_test_data(path, meta)

    save = True
    title = info['path'] +'_'+ info['name']
    mods.append(title)

    n, n_hat, t, comptime = test_step(model, input)
    start_idx = 0
    n_evol, mace_time  = test_evolution(model, input, start_idx=start_idx)

    print('\n>> Denormalising abundances...')
    n = ds.get_abs(n)
    n_hat = ds.get_abs(n_hat)
    n_evol = ds.get_abs(n_evol)

    print('\n>> Calculating & saving losses...')
    # print('per time step:')
    mse = loss.abs_loss(n, n_hat)
    avg_step.append(mse.mean())
    std_step.append(mse.std())
    sum_step.append(mse.sum())
    losses_step = [mse.mean(), mse.std(), mse.sum()]

    # print('    evolution:')
    mse_evol = loss.abs_loss(n, n_evol)
    avg_evol.append(mse_evol.mean())
    std_evol.append(mse_evol.std())
    sum_evol.append(mse_evol.sum())
    losses_evol = [mse_evol.mean(), mse_evol.std(), mse_evol.sum()]

    testloss = {title: losses_step,
                title: losses_evol,
                }
    
    with open(outloc+dirname+"/testloss.json", "r") as outfile:
        testloss_old = json.load(outfile)
    testloss_old.update(testloss)
    # print(testloss_old)
    with open(outloc+dirname+"/testloss.json", "w") as outfile:
        json.dump(testloss_old, outfile, indent=4)

    # plotting = False
    if plotting == True:
        print('\n>> Plotting...')

        ## --------------- Neural model - per time step ----------------- ##
        # pl.plot_compare(n, n_hat, plots_path, 'comp_timestep_'+title, alpha = 0.5, j = -1, save=True)
        pl.plot_abs( r,n, n_hat, plots_path, rho,T,'-timestep_'+title+'_'+epoch,specs_lg, specs=specs, save=save, step = True)

        ## --------------- Neural model - evolution ----------------- ##

        # pl.plot_compare(n, n_evol, plots_path, 'comp_evolution_'+title, alpha = 0.5, j = -1, save=True)
        pl.plot_abs( r,  n, n_evol, plots_path, rho, T, '-evolution_'+title+'_'+epoch,specs_lg,specs=specs, save=save)

    print('Done!')
    print('----------------------------------------\n')

    return np.array(avg_step), np.array(std_step), np.array(sum_step), np.array(avg_evol), np.array(std_evol), np.array(sum_evol), mods