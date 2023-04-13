import matplotlib.pyplot as plt
import matplotlib        as mpl
import sys
import os

import torch
from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})
mpl.rcParams.update({'font.size': 8})
plt.rcParams['figure.dpi'] = 200


## import own functions
sys.path.insert(1, '/lhome/silkem/MACE/MACE/src/mace')
import autoencoder  as ae
import dataset      as ds
import training     as tr
import utils


## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 100
epochs = 100

kwargs = {'num_workers': 1, 'pin_memory': True} 

## -------------------------------------------------------------------------

dir = '/lhome/silkem/MACE/MACE/train_data_C/'

train, data_loader, test_loader = ds.get_dataset(dir, batch_size, kwargs, plot = False)

## Set up training hyperparams
lrs = ['1e-5', '3e-5','1e-4', '3e-4', '1e-3', '3e-3', '1e-2', '3e-2', '1e-1', '3e-1', '1e0']                   ## learning rate

## Set up architecture hyperparams
input_dim  = train.df.shape[1]
hidden_dim = 300
latent_dim = 10
output_dim = input_dim
nb_hidden = 2
type = 'decr'

name = 'model2'

## make dir for output
path = '/lhome/silkem/MACE/MACE/ae-models/learning-rate/'+name+'/'
utils.makeOutputDir(path)

## Training model
for lr in lrs:
    model = ae.build(input_dim, hidden_dim, latent_dim,output_dim, nb_hidden, type, DEVICE)
    ae.name(model, 'Encoder','Decoder',name)
    
    loss_train_all, loss_test_all = tr.Train(model, float(lr), data_loader, test_loader, epochs, DEVICE, plot = True, show = False)

    plt.savefig(     path+'/ae-lr'+str(lr)+'.png')
    torch.save(model,path+'/ae-lr'+str(lr)+'.pl')