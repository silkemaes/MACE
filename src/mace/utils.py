'''
This script contains utility functions for MACE.
'''


import os
import numpy            as np
import json
import torch
import src.mace.mace    as mace
from pathlib import Path


def makeOutputDir(path):
    '''
    Makes the output directory - if the path does not exist yet.
    Returns the path of that dir.
    '''
    if not os.path.exists(path):
        os.mkdir(path)
        
    return path

def relative_error(x,x_hat):
    '''
    Computes the relative error between x and x_hat
        The x tensor is the true value
        The x_hat tensor is the predicted value

        x contains one element more than x_hat (the initial value of the system)
    '''
    res = np.abs((x[1:]-x_hat)/x[1:])
    return res

def temp(Tstar, eps, r):
    '''
    Computes the temperature at a given radius r, 
    according to the power-law temperature profile.
    T = Tstar * (r/Rstar)^(-eps)
    '''
    Rstar = 2e13
    # r = 1e16
    T = Tstar * (r/Rstar)**(-eps)
    return T

def error(n,n_hat):
    '''
    Computes the error between the true and predicted values.

    The error is defined as 
            error = ( log10(n) - log10(n_hat) ) / log10(n) in an element-wise way. 
        See Maes et al. (2024), Eq. (23) for more information.

    Input:
        - n: true values
        - n_hat: predicted values
    
    Returns:
        - err: the error per species, for each sample
        - the summed error, averaged by the number of samples in n
    '''
    err = (np.log10(n[:])-np.log10(n_hat))[:]/np.log10(n[:][:])
    nb_samples = len(n[:,0])

    return err, np.abs(err).sum()/nb_samples


def get_absolute_residuals(real, pred):
    '''
    Computes the absolute residuals between the real and predicted values.
    Mainly used for plotting of results.
    '''

    nb_specs = np.shape(pred)[1]

    for i in range(nb_specs):  ## loop over all specs
        res = np.abs((np.array(real[:,i])-np.array(pred[:,i])))/float(max(np.array(real[:,i])))

    res = res/nb_specs

    return res

def get_files_in(path):
    '''
    Returns the files in the given path.
    '''
    files = os.listdir(path) 
    locs = []
    for file in files:
        locs.append(file)
    
    return locs


def unscale(x, min, max):
    '''
    Unscale the data from the range [0,1] to the original range [min,max]
    '''
    unscaled = x*np.abs(max-min)+min
    return unscaled


def get_specs():
    '''
    Reads the species file, using code from ChemTorch.
        chemtype: 'C' for carbon-rich,  'O' for oxygen-rich
        rate: indicates the used rate file. 16 --> Rate16, UMIST

    Returns a dictionary with the species and their index, 
        and a dictionary with the index and the species.

    '''

    parentpath = str(Path(__file__).parent)[:-15]

    loc_specs = parentpath+'data/rate16.specs'
    
    specs = np.loadtxt(loc_specs, usecols=(1), dtype=str, skiprows = 1, max_rows=466)  

    specs_dict = dict()
    idx_specs  = dict()
    for i in range(len(specs)):
        specs_dict[specs[i]] = i
        idx_specs[i] = specs[i]

    return specs_dict, idx_specs


def normalise(x,min,max):
    '''
    Normalise the data to the range [0,1]
    '''
    norm = (x - min)*(1/np.abs( min - max ))
    return norm

def generate_random_numbers(n, start, end):
    '''
    Generates n random numbers between start and end
    '''
    return np.random.randint(start, end, size=n)



def load_model(loc, meta, epoch):
    '''
    Load a MACE model.

    Option to load a specific epoch file, or the latest epoch.

        if sepr == True, the model is loaded from the epoch file, indicated with epoch

        meta; meta data from the model, needed to know the hyperparameters
    
    '''
    n_dim = 468
    cuda   = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    # model = mace.Solver(p_dim=4,z_dim = meta['z_dim'], n_dim=n_dim, nb_hidden=meta['nb_hidden'], ae_type=meta['ae_type'], DEVICE = DEVICE)

    model = mace.Solver(p_dim=4,z_dim = meta['z_dim'], n_dim=n_dim, nb_hidden=meta['nb_hidden'], ae_type=meta['ae_type'], scheme = meta['scheme'],nb_evol=meta['nb_evol'] , lr = meta['lr'], path = loc,DEVICE = DEVICE)

    if epoch >= 0:
        file = 'nn/nn'+str(epoch)+'.pt'
    else:
        file = 'nn/nn.pt'

    model.load_state_dict(torch.load(loc+file))
    
    num_params = count_parameters(model)
    print(f'The model has {num_params} trainable parameters')

    return model, num_params



def load_meta(loc):
    '''
    Load the meta file of a MACE model,
    given the output location (outloc) and the name of the directory (loc).
    '''
    ## loading meta file
    with open(loc+'/meta.json', 'r') as f:
        meta=f.read()
    meta  = json.loads(meta)

    return meta



def count_parameters(model):
    '''
    Count the number of trainable parameters in a model.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



