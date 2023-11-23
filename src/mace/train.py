import numpy    as np
from time               import time

import torch
import torch.nn          as nn
from torch.utils.data    import DataLoader
from torch.optim         import Adam

## own scripts
import dataset  as ds
import plotting  



def mse_loss(x, x_hat):
    '''
    Return the mean squared loss per x_i.
    '''
    loss = (x-x_hat)**2
    return loss

def rel_loss(x,x_hat):
    '''
    Return the relative loss per x_i.
    The relative loss is given by ((x_hat-x_0+eps**2)/(x-x_0+eps))**2, 
        where eps makes sure we don't devide by 0.
    '''
    x_0   = x[:,0,:]
    eps = 1e-10
    loss  = ((torch.abs(x_hat-x_0)+eps**2)/(torch.abs(x-x_0)+eps))**2     ## absolute waarden nemen rond x, zodat het niet nog 0 kan worden
    return loss

def loss_function(x,x_hat, norm_mse=1, norm_rel=1,f_mse = 1, f_rel = 1):
    '''
    Get the MSE loss and the relative loss, normalised to the maximum lossvalue.
        - f_mse and f_rel are scaling factors, put to 0 if you want to exclude one of both losses.
    Returns the MSE loss per species, and the relative loss per species.
    '''
    mse = (mse_loss(x,x_hat))
    rel = (rel_loss(x,x_hat))

    mse = mse/norm_mse* f_mse
    rel = rel/norm_rel* f_rel

    return mse,rel



def train_one_epoch(data_loader, model, DEVICE, optimizer, norm_mse, norm_rel,f_mse, f_rel):
    '''
    Function to train 1 epoch.

    - data_loader   = data, torchtensor
    - model         = ML architecture to be trained
    - loss_type     = type of loss function, string
        Options:
            - 'mse'     : mean squared error loss
            - 'rel'     : relative change in abundance loss
            - 'combi'   : combination of mse & rel loss

    Method:
    1. get data
    2. push it through the model, x_hat = result
    3. calculate loss (difference between x & x_hat), according to loss function defined in loss_function()
    4. with optimiser, get the gradients and update weights using back propagation.
    
    Returns 
    - losses
    '''    
    overall_loss = 0
    overall_mse_loss = 0
    overall_rel_loss = 0
    idv_mse_loss = torch.zeros(466)
    idv_rel_loss = torch.zeros(466)
    status = 0

    for i, (n,p,t) in enumerate(data_loader):

        print('\tbatch',i+1,'/',len(data_loader),end="\r")

        # while i < 500:
        
        n = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
        p = p.to(DEVICE) 
        t = t.to(DEVICE)

        # if t[-1,-1].item() < 0.011003478870511375:    ## only use data with small dt

        n = torch.swapaxes(n,1,2)

        n_hat, modstatus = model(n[:,0,:],p,t)       

        if modstatus.item() == 4:
            status += modstatus.item()

        ## Calculate losses
        mse_loss, rel_loss  = loss_function(n,n_hat,norm_mse, norm_rel,f_mse, f_rel)

        loss = mse_loss.mean() + rel_loss.mean()

        overall_loss += loss.item()
        overall_mse_loss += mse_loss.mean().item()
        overall_rel_loss += rel_loss.mean().item()
        idv_mse_loss += mse_loss[:,-1,:].view(-1)       ## only take the loss on the final abundances 
        idv_rel_loss += rel_loss[:,-1,:].view(-1)


        ## Backpropagation
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # else:           ## else: skip this data
        #     continue

        # break

    return (overall_loss)/(i+1), overall_mse_loss/(i+1), overall_rel_loss/(i+1), idv_mse_loss/(i+1), idv_rel_loss/(i+1), status  ## save losses
            



def validate_one_epoch(test_loader, model, DEVICE, norm_mse, norm_rel,f_mse, f_rel):

    overall_loss = 0
    overall_mse_loss = 0
    overall_rel_loss = 0
    idv_mse_loss = torch.zeros(466)
    idv_rel_loss = torch.zeros(466)
    status = 0

    with torch.no_grad():
        for i, (n,p,t) in enumerate(test_loader):
            # while i < 200:
            print('\tbatch',i+1,'/',len(test_loader),end="\r")

            n     = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p     = p.to(DEVICE) 
            t     = t.to(DEVICE)

            # if t[-1,-1].item() < 0.011003478870511375:
            
            n = torch.swapaxes(n,1,2)

            n_hat, modstatus = model(n[:,0,:],p,t)         ## output van het autoecoder model

            if modstatus.item() == 4:
                status += modstatus.item()

            ## Calculate losses
            mse_loss, rel_loss  = loss_function(n,n_hat,norm_mse, norm_rel,f_mse, f_rel)

            loss = mse_loss.mean() + rel_loss.mean()

            overall_loss     += loss.item()
            overall_mse_loss += mse_loss.mean().item()
            overall_rel_loss += rel_loss.mean().item()
            idv_mse_loss += mse_loss[:,-1,:].view(-1) 
            idv_rel_loss += rel_loss[:,-1,:].view(-1) 

        # else:           ## else: skip this data
        #     continue

        return (overall_loss)/(i+1), overall_mse_loss/(i+1), overall_rel_loss/(i+1), idv_mse_loss/(i+1), idv_rel_loss/(i+1), status  ## save losseses



def train(model, lr, data_loader, test_loader, epochs, DEVICE, norm_mse, norm_rel,f_mse, f_rel, plot = False, log = True, show = True):
    optimizer = Adam(model.parameters(), lr=lr)

    ## initialise lists for statistics of training
    trainstats = dict()
    loss_train_all = []
    train_mse_loss_all = []
    train_rel_loss_all = []
    train_idv_mse_loss_all = []
    train_idv_rel_loss_all = []
    train_status_all = []

    ## initialise lists for statistics of validating
    teststats = dict()
    loss_test_all  = []
    test_mse_loss_all = []
    test_rel_loss_all = []
    test_idv_mse_loss_all = []
    test_idv_rel_loss_all = []
    test_status_all = []
    

    print('Model:         ')
    print('learning rate: '+str(lr))
    print('\n>>> Training model...')
    for epoch in range(epochs):

        ## Training
        
        model.train()
        print('')
        train_loss, train_mse_loss, train_rel_loss, train_idv_mse_loss, train_idv_rel_loss, status = train_one_epoch(data_loader, model, DEVICE, optimizer, norm_mse, norm_rel,f_mse, f_rel)
        ## save losses & status
        loss_train_all.append(train_loss)  
        train_mse_loss_all.append(train_mse_loss)
        train_rel_loss_all.append(train_rel_loss)
        train_idv_mse_loss_all.append(train_idv_mse_loss.detach().cpu().numpy())
        train_idv_rel_loss_all.append(train_idv_rel_loss.detach().cpu().numpy())
        train_status_all.append(status%4)

        
        trainstats['total_loss']        = loss_train_all
        trainstats['total_mse_loss']    = train_mse_loss_all
        trainstats['total_rel_loss']    = train_rel_loss_all
        trainstats['idv_mse_loss']      = train_idv_mse_loss_all
        trainstats['idv_rel_loss']      = train_idv_rel_loss_all
        trainstats['status']            = train_status_all

        ## Validating
        # print('\n>>> Validating model...')
        model.eval() ## zelfde als torch.no_grad

        test_loss, test_mse_loss, test_rel_loss, test_idv_mse_loss, test_idv_rel_loss, status = validate_one_epoch(test_loader, model, DEVICE, norm_mse, norm_rel,f_mse, f_rel)
        ## save losses & status
        loss_test_all.append(test_loss)  
        test_mse_loss_all.append(test_mse_loss)
        test_rel_loss_all.append(test_rel_loss)
        test_idv_mse_loss_all.append(test_idv_mse_loss.detach().cpu().numpy())
        test_idv_rel_loss_all.append(test_idv_rel_loss.detach().cpu().numpy())
        test_status_all.append(status%4)

        teststats['total_loss']        = loss_test_all
        teststats['total_mse_loss']    = test_mse_loss_all
        teststats['total_rel_loss']    = test_rel_loss_all
        teststats['idv_mse_loss']      = test_idv_mse_loss_all
        teststats['idv_rel_loss']      = test_idv_rel_loss_all
        teststats['status']            = test_status_all
        
        print("\nEpoch", epoch + 1, "complete!", "\tAverage loss train: ", train_loss, "\tAverage loss test: ", test_loss, end="\r")
    print('\n \tDONE!')

    if plot == True:
        plotting.plot_loss(trainstats, teststats, log = log, show = show)

    return trainstats, teststats




def test(model, test_loader, DEVICE, f_mse, f_rel):

    mace_time = list()
    overall_loss = 0
    idv_mse_loss = []
    idv_rel_loss = []
    print('\n>>> Testing model...')

    with torch.no_grad():
        for i, (n,p,t) in enumerate(test_loader):
            print('\tbatch',i+1,'/',len(test_loader),end="\r")

            n     = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p     = p.to(DEVICE) 
            t     = t.to(DEVICE)

            # if t[-1,-1].item() < 0.011003478870511375:
            
            n = torch.swapaxes(n,1,2)

            tic = time()
            n_hat, status = model(n[:,0,:],p,t)         ## output van het autoecoder model
            toc = time()

            if status.item() == 4:
                print('ERROR: neuralODE could not be solved!',i)
                # break

            ## Calculate losses
            mse_loss, rel_loss  = loss_function(n,n_hat,f_mse, f_rel)

            loss = mse_loss.mean() + rel_loss.mean()

            ## overall summed loss of test set
            overall_loss     += loss.item()

            ## individual losses of test set
            idv_mse_loss.append(mse_loss[:,-1,:].view(-1) .detach().cpu().numpy())
            idv_rel_loss.append(rel_loss[:,-1,:].view(-1) .detach().cpu().numpy())

            solve_time = toc-tic
            mace_time.append(solve_time)

        else:           ## else: skip this data
            mace_time.append(0)


    print('\nTest loss:',(overall_loss)/(i+1))

    return n, n_hat, t, (overall_loss)/(i+1),idv_mse_loss,idv_rel_loss, mace_time