
import numpy             as np
import pandas as pd

import torch
import torch.nn          as nn
from torch.utils.data    import DataLoader
from torch.optim         import Adam

## own scripts
import dataset as ds
import plotting   



def loss_function(x, x_hat):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    return reproduction_loss

'''
x = dataset in PyTorch tensor form
'''
def get_unscaled(Dataset, x, scale):
    mean, std, min, max = Dataset.get_stats()

    if scale == 'norm':
        unscale = x*(std) + mean
    if scale == 'minmax':
        unscale = x*np.abs(min-max) + min
    
    return unscale

'''
Function to compare an abundance profile with the autoencoded version.
'''
def test_abundance_profile(dir, label, model, DEVICE, kwargs, scale = 'norm'):
    physpar = pd.read_fwf(dir+'csphyspar_smooth_'+label+'.out', )
    test_fracs = ds.MyDataset(file=dir+'csfrac_smooth_'+label+'.out', fraction=1, scale = scale)
    
    rad = physpar['RADIUS']

    df_tensor = DataLoader(dataset=test_fracs, batch_size=len(test_fracs), shuffle=False ,  **kwargs)

    model.eval()

    with torch.no_grad(): 
        for x in df_tensor:
            x_hat = model(x.to(DEVICE))

    pred = 10**get_unscaled(test_fracs, x_hat, scale)
    real = 10**get_unscaled(test_fracs, x    , scale)

    return rad, real, pred


def train_one_epoch(data_loader, model, DEVICE, optimizer):
    
    overall_loss = 0
    
    for i, x in enumerate(data_loader):
           
        x     = x.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device

        x_hat = model(x)         ## output van het autoecoder model

        ## Calculate losses
        loss  = loss_function(x,x_hat)
        overall_loss += loss.item()

        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return (overall_loss)/(i+1)  ## save losses



def validate_one_epoch(test_loader, model, DEVICE):

    overall_loss = 0

    with torch.no_grad():
        for i, x in enumerate(test_loader):

            x     = x.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device

            x_hat = model(x)         ## output van het autoecoder model

            ## Calculate losses
            loss  = loss_function(x,x_hat)
            overall_loss += loss.item()

        return (overall_loss)/(i+1)  ## save losses


def Train(model, lr, data_loader, test_loader, epochs, DEVICE, plot = False, log = True):
    optimizer = Adam(model.parameters(), lr=lr)

    loss_train_all = []
    loss_test_all  = []

    print('Model:         '+model.name)
    print('learning rate: '+str(lr))
    for epoch in range(epochs):

        ## Training
        model.train()

        train_loss = train_one_epoch(data_loader, model, DEVICE, optimizer)
        loss_train_all.append(train_loss)  ## save losses

        ## Validating
        model.eval() ## zelfde als torch.no_grad

        test_loss = validate_one_epoch(test_loader, model, DEVICE)
        loss_test_all.append(test_loss)
        
        print("\tEpoch", epoch + 1, "complete!", "\tAverage loss train: ", train_loss, "\tAverage loss test: ", test_loss, end="\r")
    print('\n \tDONE!\n')

    if plot == True:
        plotting.plot_loss(loss_train_all, loss_test_all, log = log)

    return loss_train_all, loss_test_all


def Test(model, test_loader, DEVICE):
    overall_loss = 0

    with torch.no_grad():
        for i, x in enumerate(test_loader):

            x     = x.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device

            x_hat = model(x)         ## output van het autoecoder model

            ## Calculate losses
            loss  = loss_function(x,x_hat)
            overall_loss += loss.item()

    loss = (overall_loss)/(i+1)
    print('Test loss '+model.name+': ',loss)

    return x, x_hat, loss