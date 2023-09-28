
import sys
import torch
import json
import datetime         as dt
from time             import time


## import own functions
sys.path.insert(1, '/STER/silkem/MACE/src/mace')
import dataset      as ds
import train        as tr
import neuralODE    as nODE
import utils        as utils

start = time()


## ADJUST THESE PARAMETERS FOR DIFFERENT MODELS
epochs = 50
lr = 1e-2
z_dim = 10
dirname = 'C-short-dtime'


## ---------------------------------------
name = dt.datetime.now()
path = '/STER/silkem/MACE/models/'+str(name)

utils.makeOutputDir(path)

## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1 ## if not 1, dan kan er geen tensor van gemaakt worden

kwargs = {'num_workers': 1, 'pin_memory': True} 


train, data_loader, test_loader = ds.get_data(dirname = dirname, batch_size=batch_size, kwargs=kwargs, plot = True, scale = None)
model = nODE.Solver(p_dim=4,z_dim = z_dim, n_dim=466, DEVICE = DEVICE)

tic = time()
loss_train_all, loss_test_all = tr.train(model, lr, data_loader, test_loader, epochs, DEVICE)
n_test, n_test_hat, tstep, loss = tr.test(model, test_loader, DEVICE)


torch.save(model,path+'/nn.pl')

toc = time()
train_time = toc-tic

stop = time()

overhead_time = (stop-start)-train_time

metadata = {'traindir'  : dirname,
            'lr'        : lr,
            'epochs'    : epochs,
            'z_dim'     : z_dim,
            'train_time'    : train_time,
            'overhead'      : overhead_time,
            'samples'       : len(train),
            'mins'          : train.mins,
            'maxs'          : train.maxs,
            'cutoff_abs'    : train.cutoff
}


json_object = json.dumps(metadata, indent=4)
with open(path+"/meta.json", "w") as outfile:
    outfile.write(json_object)

