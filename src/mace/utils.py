import os
import numpy as np
import sys
import json
import torch


sys.path.append('/STER/silkem/ChemTorch/src')
import rates as rate

sys.path.insert(1, '/STER/silkem/MACE/src/mace')
from CSE_0D.loss  import Loss_analyse
import mace     as mace


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
    specs, parnt, convs = rate.read_specs_file(chemtype= 'C', rate=16)
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

def load_model(loc, meta, epoch, sepr):
    '''
    Load a MACE model.

    Option to load a specific epoch file, or the latest epoch.

        if sepr == True, the model is loaded from the epoch file, indicated with epoch

        meta; meta data from the model, needed to know the hyperparameters
    
    '''
    n_dim = 468
    cuda   = False
    DEVICE = torch.device("cuda" if cuda else "cpu")
    model = mace.Solver(p_dim=4,z_dim = meta['z_dim'], n_dim=n_dim, nb_hidden=meta['nb_hidden'], ae_type=meta['ae_type'], DEVICE = DEVICE)

    if sepr == True:
        file = 'nn/nn'+str(epoch)+'.pt'
    else:
        file = 'nn/nn.pt'

    model.load_state_dict(torch.load(loc+file))
    
    num_params = count_parameters(model)
    print(f'The model has {num_params} trainable parameters')

    return model


    
def load_all(outloc, dirname,epoch = ''):
    '''
    Load all the components of a MACE model,
    given the output location (outloc) and the name of the directory (dirname).

    Returns:
        - meta: file with meta data
        - model: torch model
        - trainloss: training loss per epoch
        - testloss: test loss per epoch
    '''
    loc   = outloc+dirname+'/'

    ## loading meta file
    with open(loc+'/meta.json', 'r') as f:
        meta=f.read()
    meta  = json.loads(meta)

    ## loading torch model
    model = load_model(loc,meta,epoch, sepr=True)

    ## loading losses

    trainloss = Loss_analyse()
    trainloss.load(loc, 'train', meta)
    testloss  = Loss_analyse()
    testloss.load(loc, 'test', meta)

    return meta, model, trainloss, testloss


def load_meta(outloc,loc):
    '''
    Load the meta file of a MACE model,
    given the output location (outloc) and the name of the directory (loc).
    '''
    ## loading meta file
    with open(outloc+loc+'/meta.json', 'r') as f:
        meta=f.read()
    meta  = json.loads(meta)

    return meta

    


def load_model_old(loc, meta, epoch, sepr):
    n_dim = 468
    cuda   = False
    DEVICE = torch.device("cuda" if cuda else "cpu")
    model = mace.Solver_old(p_dim=4,z_dim = meta['z_dim'], n_dim=n_dim, DEVICE = DEVICE)

    if sepr == True:
        file = 'nn/nn'+str(epoch)+'.pt'
    else:
        file = 'nn/nn.pt'

    model.load_state_dict(torch.load(loc+file))
    
    num_params = count_parameters(model)
    print(f'The model has {num_params} trainable parameters')

    return model

def load_all_noevol(outloc, dirname, sepr = False, epoch = ''):
    loc   = outloc+dirname+'/'

    ## loading meta file
    with open(loc+'/meta.json', 'r') as f:
        meta=f.read()
    meta  = json.loads(meta)

    ## loading torch model
    model = load_model_old(loc, meta, epoch, sepr)

    ## loading losses
    if sepr == False:
        trainloss = Loss_analyse()
        trainloss.load(loc, 'train', meta)
        testloss  = Loss_analyse()
        testloss.load(loc, 'test', meta)
        model.set_status(np.load(loc+'train/status.npy'), 'train')
        model.set_status(np.load(loc+'test/status.npy'), 'test')

    if sepr == False:
        return meta, model, trainloss, testloss
    else:
        return meta, model

def count_parameters(model):
    '''
    Count the number of trainable parameters in a model.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)