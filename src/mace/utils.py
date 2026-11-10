'''
This script contains utility functions for MACE.
'''


import os
import numpy            as np
import json
import torch
import src.mace.mace    as mace
from pathlib import Path
from src.mace.input             import Input


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


def get_specs(model='1DCSE'):
    '''
    Reads the species file, using code from ChemTorch.
        chemtype: 'C' for carbon-rich,  'O' for oxygen-rich
        rate: indicates the used rate file. 16 --> Rate16, UMIST

    Returns a dictionary with the species and their index, 
        and a dictionary with the index and the species.

    '''

    parentpath = str(Path(__file__).parent)[:-8]
    if model == '1DCSE':
        loc_specs = parentpath + 'rate16.specs'
        specs = np.loadtxt(loc_specs,
                           usecols=(1),
                           dtype=str,
                           skiprows=1,
                           max_rows=469)
    elif model == 'Phantom':
        loc_specs = parentpath + '/data/krome_umist.specs'
        specs = []
        with open(loc_specs, "r") as file:
            for line in file:
                if not line.startswith("#"):
                    specs.extend(line.strip('[').strip(']').replace('"', '').replace(
                        "'", "").replace("\\", "").split(',')[:-1])  # :-1 to remove last element that just contains a linebreak
    else:
        print('Model not recognized, using 1DCSE as default')
        loc_specs = parentpath + 'rate16.specs'
        specs = np.loadtxt(loc_specs,
                           usecols=(1),
                           dtype=str,
                           skiprows=1,
                           max_rows=469)
    specs_dict = dict()
    for i, sp in enumerate(specs):
        specs_dict[sp.strip()] = i

    return specs_dict


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



def overwrite_weights(model, dir, epoch=-1):
    '''
    Overwrites a MACE model's weights by the given 
    '''
    if epoch >= 0:
        infile = f'nn/nn{epoch}.pt'
    else:
        infile = 'nn/nn.pt'
    state_dict = torch.load(f'{dir}/{infile}', map_location='cpu', weights_only=True)

    model.eval()
    model.load_state_dict(state_dict) 
    model.eval()



def load_model(dir, meta, epoch=-1):
    '''
    Initializes MACE model. Tries to recover parameters from meta.json and input file.
    Overwrites MACE weights by those found in meta.json.

    - dir:      Directory where meta.json is stored
    - meta:     meta.json
    - epoch:    Epoch to load. Defaults to last weights.
    
    '''
    try:
        input = Input(meta['inputfile'], dir) #TODO unlink these variables from input
        n_dim = input.n_dim # TODO include in meta.json
    except FileNotFoundError:
        print('Loading currently relies on retracing the input file.\n',
              f'Failed to find {meta["inputfile"]}.\n',
              'n_dim should be stored in meta.json soon.')
        n_dim = 468

    p_dim = 4 # TODO include in meta.json
    z_dim = meta['z_dim']
    ae_type = meta['ae_type']
    # TODO include atol and rtol in meta.json
    
    model = mace.MACE(n_dim, p_dim, z_dim, ae_type)
    overwrite_weights(model, dir, epoch)
    num_params = count_parameters(model)
    
    return model, num_params



def load_meta(loc):
    '''
    Load the meta file of a MACE model,
    given the output location (outloc) and the name of the directory (loc).
    '''
    ## loading meta file
    with open(loc+'meta.json', 'r') as f:
        meta=f.read()
    meta  = json.loads(meta)

    return meta



def count_parameters(model):
    '''
    Count the number of trainable parameters in a model.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
