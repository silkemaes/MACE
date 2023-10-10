
import sys
import torch
import json
import numpy            as np
import datetime         as dt
from time             import time
import matplotlib.pyplot as plt


## import own functions
sys.path.insert(1, '/STER/silkem/MACE/src/mace')
import dataset      as ds
import train        as tr
import neuralODE    as nODE
import utils        as utils
import plotting     as pl

start = time()
name = dt.datetime.now()
path = '/STER/silkem/MACE/models/'+str(name)

## ADJUST THESE PARAMETERS FOR DIFFERENT MODELS
epochs = 20
lr = 1.e-3
z_dim = 10
# dirname = 'C-short-dtime'
dirname = 'new'

print('------------------------------')
print('')
print('Training:')
print('Name:', name)
print('------------------------------')
print('      # epochs:', epochs)
print(' learning rate:', lr)
print('# z dimensions:', z_dim)
print('    sample dir:',dirname)
print('')

## ---------------------------------------

utils.makeOutputDir(path)

metadata = {'traindir'  : dirname,
            'lr'        : lr,
            'epochs'    : epochs,
            'z_dim'     : z_dim,
            'done'      : 'false'
}

json_object = json.dumps(metadata, indent=4)
with open(path+"/meta.json", "w") as outfile:
    outfile.write(json_object)

## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1 ## if not 1, dan kan er geen tensor van gemaakt worden

kwargs = {'num_workers': 1, 'pin_memory': True} 


train, data_loader, test_loader = ds.get_data(dirname = dirname, batch_size=batch_size, kwargs=kwargs, plot = True, scale = None)
model = nODE.Solver(p_dim=4,z_dim = z_dim, n_dim=466, DEVICE = DEVICE)

tic = time()
loss_train_all, loss_test_all, status = tr.train(model, lr, data_loader, test_loader, epochs, DEVICE, plot = True, show = False)
print('\n\tTraining done!')

toc = time()
train_time = toc-tic

stop = time()

overhead_time = (stop-start)-train_time

metadata = {'traindir'  : dirname,
            'lr'        : lr,
            'epochs'    : epochs,
            'z_dim'     : z_dim,
            'train_time': train_time,
            'overhead'  : overhead_time,
            'samples'   : len(train),
            'cutoff_abs': train.cutoff,
            'done'      : 'true'
}

min_max = np.stack((train.mins, train.maxs), axis=1)
losses = np.stack((np.array(loss_train_all), np.array(loss_test_all)), axis = 1)

## Saving all files
np.save(path+'/minmax', min_max) 
np.save(path+'/status', np.array(status))
np.save(path+'/losses', losses)
plt.savefig(path+'/mse.png')
torch.save(model.state_dict(),path+'/nn.pt')

json_object = json.dumps(metadata, indent=4)
with open(path+"/meta.json", "w") as outfile:
    outfile.write(json_object)

print('** ',name,' ALL DONE! in [min]', round((toc-tic)/60,2))



