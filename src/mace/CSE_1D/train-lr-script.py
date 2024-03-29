import matplotlib.pyplot as plt
import matplotlib        as mpl
import sys
import time

import torch
from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})
mpl.rcParams.update({'font.size': 8})
plt.rcParams['figure.dpi'] = 200


## import own functions
sys.path.insert(1, '/lhome/silkem/MACE/MACE/src/mace')
import autoencoder  as ae
import dataset      as ds
import mace.train_1Dchem     as tr
import utils
from tqdm import tqdm 


## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 100
epochs = 10

kwargs = {'num_workers': 1, 'pin_memory': True} 

## -------------------------------------------------------------------------

dir = '/STER/silkem/MACE/data/train_data_C/'

train, data_loader, test_loader = ds.get_dataset(dir, batch_size, kwargs, plot = False)

## Set up training hyperparams
lrs = ['1e-4']#, '3e-4', '1e-3', '3e-3', '1e-2', '3e-2', '1e-1']                   ## learning rate

## Set up architecture hyperparams
input_dim  = train.df.shape[1]
hidden_dim = 300
latent_dim = 10
output_dim = input_dim
nb_hidden = 1
type = 'decr'

name = 'model4'

## make dir for output
path = '/STER/silkem/MACE/ae-models/learning-rate/'+name+'/'
utils.makeOutputDir(path)


tic = time.time() 

## Training model
for lr in tqdm(lrs):
    model = ae.build(input_dim, hidden_dim, latent_dim,output_dim, nb_hidden, type, DEVICE)
    ae.name(model, 'Encoder','Decoder',name)
    
    loss_train_all, loss_test_all = tr.Train(model, float(lr), data_loader, test_loader, epochs, DEVICE, plot = True, show = False)

    print(model.state_dict)
    plt.savefig(     path+'/ae-lr'+str(lr)+'.png')
    torch.save(model,path+'/ae-lr'+str(lr)+'.pl')

toc = time.time()

print('** ALL DONE! in [min]', round((toc-tic)/60,2))

