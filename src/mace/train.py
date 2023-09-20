import numpy    as np
# import pandas   as pd

import torch
import torch.nn          as nn
from torch.utils.data    import DataLoader
from torch.optim         import Adam

## own scripts
import dataset  as ds
import plotting  
# import tqdm 



def loss_function(x, x_hat):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    return reproduction_loss


def train_one_epoch(data_loader, model, DEVICE, optimizer):
    '''
    Function to train 1 epoch.

    - data_loader   = data, torchtensor
    - model         = ML architecture to be trained

    Method:
    1. get data
    2. push it through the model, x_hat = result
    3. calculate loss (difference between x & x_hat), according to loss function defined in loss_function()
    4. with optimiser, get the gradients and update weights using back propagation.
    
    Returns 
    - losses
    '''    
    overall_loss = 0
    count_nan = 0

    for i, (n,p,t) in enumerate(data_loader):

        print('\tbatch',i+1,'/',len(data_loader),', # nan',count_nan,end="\r")
           
        n = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
        p = p.to(DEVICE) 
        t = t.to(DEVICE)

        n = torch.swapaxes(n,1,2)

        n_hat = model(n[:,0,:],p,t)         ## output van het autoecoder model
        
        if torch.isnan(n_hat[0][-1]).any(0):
            count_nan +=1
            break

        ## Calculate losses
        loss  = loss_function(n,n_hat)
        overall_loss += loss.item()

        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    print('\n\t\t# nan:',count_nan,'/',len(data_loader))
    return (overall_loss)/(i+1)  ## save losses



def validate_one_epoch(test_loader, model, DEVICE):

    overall_loss = 0

    with torch.no_grad():
        for i, (n,p,t) in enumerate(test_loader):
            print('\tbatch',i+1,'/',len(test_loader),end="\r")

            n     = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p     = p.to(DEVICE) 
            t     = t.to(DEVICE)
            
            n = torch.swapaxes(n,1,2)

            n_hat = model(n[:,0,:],p,t)         ## output van het autoecoder model

            ## Calculate losses
            loss  = loss_function(n,n_hat)
            overall_loss += loss.item()

            return (overall_loss)/(i+1)  ## save losses


def train(model, lr, data_loader, test_loader, epochs, DEVICE, plot = False, log = True, show = True):
    optimizer = Adam(model.parameters(), lr=lr)

    loss_train_all = []
    loss_test_all  = []

    print('Model:         ')
    print('learning rate: '+str(lr))
    print('\n>>> Training model...')
    for epoch in range(epochs):

        ## Training
        model.train()

        train_loss = train_one_epoch(data_loader, model, DEVICE, optimizer)
        loss_train_all.append(train_loss)  ## save losses

        ## Validating
        model.eval() ## zelfde als torch.no_grad

        test_loss = validate_one_epoch(test_loader, model, DEVICE)
        loss_test_all.append(test_loss)
        
        print("Epoch", epoch + 1, "complete!", "\tAverage loss train: ", train_loss, "\tAverage loss test: ", test_loss, end="\r")
    print('\n \tDONE!')

    if plot == True:
        plotting.plot_loss(loss_train_all, loss_test_all, log = log, show = show)

    return loss_train_all, loss_test_all


def test(model, test_loader, DEVICE):
    overall_loss = 0

    print('\n>>> Testing model...')
    count_nan = 0
    with torch.no_grad():
        for i, (n,p,t) in enumerate(test_loader):
            print('\tbatch',i+1,'/',len(test_loader),', # nan',count_nan,end="\r")

            n     = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p     = p.to(DEVICE) 
            t     = t.to(DEVICE)
            
            n = torch.swapaxes(n,1,2)

            n_hat = model(n[:,0,:],p,t)         ## output van het autoecoder model

            if torch.isnan(n_hat[0][-1]).any(0):
                count_nan +=1
                break

            ## Calculate losses
            loss  = loss_function(n,n_hat)
            overall_loss += loss.item()

            

    loss = (overall_loss)/(i+1)
    print('\nTest loss:',loss)

    return n, n_hat, loss