
import sys
import torch
import json
import numpy                as np
import datetime             as dt
from time                   import time
import matplotlib.pyplot    as plt
# import matplotlib           as mpl
from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})
# mpl.rcParams.update({'font.size': 10})
# plt.rcParams['figure.dpi'] = 200


## import own functions
sys.path.insert(1, '/STER/silkem/MACE/src/mace')
import dataset      as ds
import train        as tr
import neuralODE    as nODE
import utils        as utils
import plotting     as pl
import loss

start = time()
name = dt.datetime.now()
path = '/STER/silkem/MACE/models/test_'+str(name)


## ================================================== INPUT ========
## ADJUST THESE PARAMETERS FOR DIFFERENT MODELS

lr = 1.e-3
tot_epochs = 10
nb_epochs  = 5
z_dim = 10
dt_fract = 0.2
# dirname = 'C-short-dtime'
# dirname = 'new'
dirname = 'easy-mace2'

## ================================================== INPUT ========

print('------------------------------')
print('')
print('Training:')
print('Name:', name)
print('------------------------------')
print('      # epochs:', tot_epochs)
print(' learning rate:', lr)
print('# z dimensions:', z_dim)
print('    sample dir:', dirname)
print('')

## --------------------------------------- SET UP ------------------

utils.makeOutputDir(path)
utils.makeOutputDir(path+'/nn')

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
train, test, data_loader, test_loader = ds.get_data(dirname = dirname,dt_fract=dt_fract, batch_size=batch_size, kwargs=kwargs)
## Make model
model = nODE.Solver(p_dim=4,z_dim = z_dim, n_dim=466, DEVICE = DEVICE)


## --------------------------------------- TRAINING ----------------

## ------------- PART 1: unnormalised losses 
norm = {'mse' : 1,
        'rel' : 1,
        'evo' : 1}

fract = {'mse' : 1, 
         'rel' : 1,
         'evo' : 1}


## Make loss objects
trainloss = loss.Loss(norm, fract)
testloss  = loss.Loss(norm, fract)

tic = time()
trainloss, testloss = tr.train(model, lr, data_loader, test_loader,path, end_epochs = 2, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, plot = False, log = True, show = True)
toc = time()
train_time1 = toc-tic

## ------------- PART 2: normalised losses, but reinitialise model
model = nODE.Solver(p_dim=4,z_dim = z_dim, n_dim=466, DEVICE = DEVICE)

norm = {'mse' : 1,
        'rel' : 1,
        'evo' : 1}

trainloss.change_norm({'mse' :np.mean(trainloss.get_loss('mse')),
                       'rel' :np.mean(trainloss.get_loss('rel')),
                       'evo' :np.mean(trainloss.get_loss('evo'))})  
testloss.change_norm({'mse' :np.mean(testloss.get_loss('mse')),
                        'rel' :np.mean(testloss.get_loss('rel')),
                        'evo' :np.mean(testloss.get_loss('evo'))})

tic = time()
trainloss, testloss = tr.train(model, lr, data_loader, test_loader,path, end_epochs = nb_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_epochs=2, plot = False, log = True, show = True)
toc = time()
train_time2 = toc-tic

## ------------- PART 3: increase losses with factor & train further
fract = {'mse' : 100, 
         'rel' : 100,
         'evo' : 100}
trainloss.change_fract(fract)
testloss.change_fract(fract)

tic = time()
trainloss, testloss = tr.train(model, lr, data_loader, test_loader, path, end_epochs = tot_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_epochs=5, plot = False, log = True, show = False)
toc = time()
train_time3 = toc-tic

print('\n\tTraining done!')

train_time = train_time1+train_time2+train_time3


## -------------- STACKING LOSSES THE DATA --------------------
trainpath = path+'/train'
testpath  = path+'/test'

trainloss.save(trainpath)
testloss.save(testpath)



min_max = np.stack((train.mins, train.maxs), axis=1)
np.save(path+'/minmax', min_max) 

torch.save(model.state_dict(),path+'/nn/nn.pt')

fig_loss = pl.plot_loss(trainloss, testloss, show = True)
plt.savefig(path+'/loss.png')

stop = time()

overhead_time = (stop-start)-train_time

metadata = {'traindir'  : dirname,
            'lr'        : lr,
            'epochs'    : tot_epochs,
            'z_dim'     : z_dim,
            'dt_fract'  : train.dt_fract,
            'tmax'      : train.tmax,
            'train_time': train_time,
            'overhead'  : overhead_time,
            'samples'   : len(train),
            'cutoff_abs': train.cutoff,
            'done'      : 'true',
            'norm'      : norm,
            'fract'     : fract
}

json_object = json.dumps(metadata, indent=4)
with open(path+"/meta.json", "w") as outfile:
    outfile.write(json_object)

print('** ',name,' ALL DONE! in [min]', round((train_time + overhead_time)/60,2))



