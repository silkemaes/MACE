
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
lr = 1.e-3
tot_epochs = 80
z_dim = 10
dirname = 'C-short-dtime'
# dirname = 'new'

print('------------------------------')
print('')
print('Training:')
print('Name:', name)
print('------------------------------')
print('      # epochs:', tot_epochs)
print(' learning rate:', lr)
print('# z dimensions:', z_dim)
print('    sample dir:',dirname)
print('')

## --------------------------------------- SET UP ------------------

utils.makeOutputDir(path)

metadata = {'traindir'  : dirname,
            'lr'        : lr,
            'epochs'    : tot_epochs,
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

## Load train & test data sets 
train, test, data_loader, test_loader = ds.get_data(dirname = dirname, batch_size=batch_size, kwargs=kwargs, plot = True, scale = None)
## Make model
model = nODE.Solver(p_dim=4,z_dim = z_dim, n_dim=466, DEVICE = DEVICE)


## --------------------------------------- TRAINING ----------------

## ------------- PART 1: unnormalised losses 
norm_mse = 1
norm_rel = 1
f_mse = 1
f_rel = 1
n_epochs_un = 5
tic = time()
trainstats1, teststats1 = tr.train(model, lr, data_loader, test_loader, epochs = n_epochs_un, DEVICE= DEVICE, norm_mse= norm_mse, norm_rel=norm_rel, f_mse=f_mse, f_rel=f_rel, plot = False, log = True, show = True)
toc = time()
train_time1 = toc-tic

## ------------- PART 2: normalised losses 
norm_mse = np.mean(trainstats1['total_mse_loss'])
norm_rel = np.mean(trainstats1['total_rel_loss'])
f_mse = 1
f_rel = 1
n_epochs = 50 - n_epochs_un
tic = time()
trainstats2, teststats2 = tr.train(model, lr, data_loader, test_loader, epochs = n_epochs, DEVICE= DEVICE, norm_mse= norm_mse, norm_rel=norm_rel, f_mse=f_mse, f_rel=f_rel, plot = False, log = True, show = True)
toc = time()
train_time2 = toc-tic

## ------------- PART 3: increase losses with factor & train further
# norm_mse = np.mean(trainstats1['total_mse_loss'])
# norm_rel = np.mean(trainstats1['total_rel_loss'])
f_mse = 100
f_rel = 100
n_epochs = tot_epochs - n_epochs
tic = time()
trainstats3, teststats3 = tr.train(model, lr, data_loader, test_loader, epochs = n_epochs, DEVICE= DEVICE, norm_mse= norm_mse, norm_rel=norm_rel, f_mse=f_mse, f_rel=f_rel, plot = False, log = True, show = True)
toc = time()
train_time3 = toc-tic

print('\n\tTraining done!')

train_time = train_time1+train_time2+train_time3


## -------------- STACKING LOSSES THE DATA --------------------

stats_train = dict()
for key in trainstats1:
    arr1 = np.array([trainstats1[key]])
    arr2 = np.array([trainstats2[key]])
    arr3 = np.array([trainstats2[key]])
    stats_train[key] = np.concatenate((arr1,arr2, arr3), axis=1)[0]


## vanaf hier nog aanpassen!!

min_max = np.stack((train.mins, train.maxs), axis=1)
losses = np.stack((np.array(loss_train_all), np.array(loss_test_all)), axis = 1)

## Saving all files
np.save(path+'/minmax', min_max) 
np.save(path+'/status', np.array(status))
np.save(path+'/losses', losses)
plt.savefig(path+'/mse.png')
torch.save(model.state_dict(),path+'/nn.pt')



stop = time()

overhead_time = (stop-start)-train_time

metadata = {'traindir'  : dirname,
            'lr'        : lr,
            'epochs'    : tot_epochs,
            'z_dim'     : z_dim,
            'train_time': train_time,
            'overhead'  : overhead_time,
            'samples'   : len(train),
            'cutoff_abs': train.cutoff,
            'done'      : 'true'
}

json_object = json.dumps(metadata, indent=4)
with open(path+"/meta.json", "w") as outfile:
    outfile.write(json_object)

print('** ',name,' ALL DONE! in [min]', round((train_time + overhead_time)/60,2))



