import os
import numpy as np
import sys
import json
import torch

sys.path.insert(1, '/STER/silkem/MACE/src/mace')
from CSE_0D.loss  import Loss_analyse
from neuralODE    import Solver_old, Solver

'''
Makes the output directory - if nessecary.
Returns the path of that dir.
'''
def makeOutputDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def relative_error(x,x_hat):
    res = np.abs((x[1:]-x_hat)/x[1:])
    return res


def get_absolute_residuals(real, pred):

    nb_specs = np.shape(pred)[1]

    for i in range(nb_specs):  ## loop over all specs
        res = np.abs((np.array(real[:,i])-np.array(pred[:,i])))/float(max(np.array(real[:,i])))
    res = res/nb_specs

    return res

def get_files_in(path):
    files = os.listdir(path) 
    locs = []
    for file in files:
        locs.append(file)
    
    return locs


def unscale(x, min, max):
    unscaled = x*np.abs(max-min)+min
    return unscaled


sys.path.append('/STER/silkem/ChemTorch/src')
import rates as rate

def get_specs():
    specs, parnt, convs = rate.read_specs_file('C', 16)
    specs_dict = dict()
    idx_specs  = dict()
    for i in range(len(specs)):
        specs_dict[specs[i]] = i
        idx_specs[i] = specs[i]

    return specs_dict, idx_specs


def normalise(x,min,max):
        norm = (x - min)*(1/np.abs( min - max ))
        return norm

def generate_random_numbers(n, start, end):
    return np.random.randint(start, end, size=n)

def load_model(loc, meta, epoch, sepr):
    n_dim = 468
    cuda   = False
    DEVICE = torch.device("cuda" if cuda else "cpu")
    model = Solver(p_dim=4,z_dim = meta['z_dim'], n_dim=n_dim, DEVICE = DEVICE)

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
    model = load_model(loc,meta, epoch, sepr)

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
    
def load_all(outloc, dirname, epoch = ''):
    loc   = outloc+dirname+'/'

    ## loading meta file
    with open(loc+'/meta.json', 'r') as f:
        meta=f.read()
    meta  = json.loads(meta)

    ## loading torch model
    model = load_model(loc,meta, epoch, sepr=True)

    ## loading losses

    trainloss = Loss_analyse()
    trainloss.load(loc, 'train', meta)
    testloss  = Loss_analyse()
    testloss.load(loc, 'test', meta)
    # model.set_status(np.load(loc+'train/status.npy'), 'train')
    # model.set_status(np.load(loc+'test/status.npy'), 'test')

    return meta, model, trainloss, testloss

    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)