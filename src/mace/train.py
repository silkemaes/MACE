import numpy    as np
from time               import time

import torch
import torch.nn          as nn
from torch.utils.data    import DataLoader
from torch.optim         import Adam

## own scripts
import dataset  as ds
import plotting  
# import tqdm 


def mse_loss(x, x_hat):
    '''
    Return the mean squared loss per x_i.
    '''
    loss = nn.functional.mse_loss(x_hat, x, reduction='none')
    return loss

def rel_loss(x,x_hat):
    '''
    Return the relative loss per x_i.
    The relative loss is given by ((x_hat-x_0+eps**2)/(x-x_0+eps))**2, 
        where eps makes sure we don't devide by 0.
    '''
    len   = x.shape[1]
    x_0   = x[:,0,:]
    x_hat =x_hat[:,1:,:]
    x = x[:,1:,:]
    eps = 1e-4
    loss  = ((x_hat-x_0+eps**2)/(x-x_0+eps))**2
    return loss

def combi_loss(x,x_hat, factor=1):
    mse = mse_loss(x,x_hat)/max(mse_loss(x,x_hat))
    rel = rel_loss(x,x_hat)/max(rel_loss(x,x_hat))/factor
    return mse,rel

def loss_function(x,x_hat,type, factor = 1):
    if type == 'mse':
        return mse_loss(x,x_hat)
    if type == 'rel': 
        return rel_loss(x,x_hat)
    if type == 'combi':
        return combi_loss(x,x_hat, factor)


def train_one_epoch(data_loader, model, DEVICE, optimizer, loss_type, factor = 1):
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
    status = 0

    for i, (n,p,t) in enumerate(data_loader):

        print('\tbatch',i+1,'/',len(data_loader),end="\r")
        
        n = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
        p = p.to(DEVICE) 
        t = t.to(DEVICE)

        if t[-1,-1].item() < 0.011003478870511375:    ## only use data with small dt

            n = torch.swapaxes(n,1,2)

            n_hat, modstatus = model(n[:,0,:],p,t)      
            # print(n[:,0,:])  

            if modstatus.item() == 4:
                # print('stat4')
                status += modstatus.item()

            ## Calculate losses
            loss  = loss_function(n,n_hat, loss_type, factor = factor)
            if type == 'combi':
                mse_loss = loss[0]
                rel_loss = loss[1]
                tot_loss = mse_loss + rel_loss
            else:
                tot_loss = loss

            overall_loss += tot_loss.mean.item()

            ## Backpropagation
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()
    
        else:           ## else: skip this data
            continue

    return (overall_loss)/(i+1), loss, status  ## save losses
            



def validate_one_epoch(test_loader, model, DEVICE, loss_type):

    overall_loss = 0
    # status = 0

    with torch.no_grad():
        for i, (n,p,t) in enumerate(test_loader):
            # print('\tbatch',i+1,'/',len(test_loader),end="\r")

            n     = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p     = p.to(DEVICE) 
            t     = t.to(DEVICE)

            if t[-1,-1].item() < 0.011003478870511375:
            
                n = torch.swapaxes(n,1,2)

                n_hat, status = model(n[:,0,:],p,t)         ## output van het autoecoder model

                # if status.item() == 4:
                #     status += 4

                ## Calculate losses
                loss  = loss_function(n,n_hat,loss_type)
                overall_loss += loss.item()

            else:           ## else: skip this data
                continue

            return (overall_loss)/(i+1)  ## save losses


def train(model, lr, data_loader, test_loader, epochs, DEVICE, loss_type,plot = False, log = True, show = True):
    optimizer = Adam(model.parameters(), lr=lr)

    loss_train_all = []
    loss_test_all  = []
    status_all = []

    print('Model:         ')
    print('learning rate: '+str(lr))
    print('\n>>> Training model...')
    for epoch in range(epochs):

        ## Training
        
        model.train()
        print('')
        train_loss, all_losses, status = train_one_epoch(data_loader, model, DEVICE, optimizer, loss_type)
        loss_train_all.append(train_loss)  ## save losses
        status_all.append(status%4)

        ## Validating
        # print('\n>>> Validating model...')
        model.eval() ## zelfde als torch.no_grad

        test_loss = validate_one_epoch(test_loader, model, DEVICE, loss_type)
        loss_test_all.append(test_loss)
        
        print("\nEpoch", epoch + 1, "complete!", "\tAverage loss train: ", train_loss, "\tAverage loss test: ", test_loss, end="\r")
    print('\n \tDONE!')

    if plot == True:
        plotting.plot_loss(loss_train_all, loss_test_all, log = log, show = show)

    return loss_train_all, loss_test_all, status_all


def test(model, test_loader, DEVICE, loss_type):
    overall_loss = 0
    mace_time = list()

    print('\n>>> Testing model...')

    with torch.no_grad():
        for i, (n,p,t) in enumerate(test_loader):
            print('\tbatch',i+1,'/',len(test_loader),end="\r")

            n     = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p     = p.to(DEVICE) 
            t     = t.to(DEVICE)

            if t[-1,-1].item() < 0.011003478870511375:
            
                n = torch.swapaxes(n,1,2)

                tic = time()
                n_hat, status = model(n[:,0,:],p,t)         ## output van het autoecoder model
                toc = time()

                if status.item() == 4:
                    print('ERROR: neuralODE could not be solved!',i)
                    # break

                ## Calculate losses
                loss  = loss_function(n,n_hat, loss_type)
                overall_loss += loss.item()

                solve_time = toc-tic
                mace_time.append(solve_time)

            else:           ## else: skip this data
                mace_time.append(0)

            
    loss = (overall_loss)/(i+1)
    print('\nTest loss:',loss)

    return n, n_hat, t, loss, mace_time