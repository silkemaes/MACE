
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
import CSE_0D.dataset    as ds
import CSE_0D.train      as tr
import neuralODE         as nODE
import utils             as utils
import CSE_0D.plotting   as pl
import CSE_0D.loss       as loss


torch.multiprocessing.set_sharing_strategy('file_system')

start = time()
name = dt.datetime.now()
path = '/STER/silkem/MACE/models/CSE_0D/'+str(name)

dt_fracts = {1 : 0.597,
             2 : 0.482,
             3 : 0.374,
             4 : 0.377,
             5 : 0.33,
             8 : 0.213,
             10: 0.147,
             12: 0.154,
             20: 0.099,
             25: 0.076,
             32: 0.062,
             48: 0.043,
             64: 0.033,
             128: 0.017
}


## ================================================== INPUT ========
## ADJUST THESE PARAMETERS FOR DIFFERENT MODELS

## READ INPUT FILE
arg = sys.argv[1]

inFile = '/STER/silkem/MACE/input/'+arg+'.txt'

with open(inFile, 'a') as file:
    file.write('\nName = '+str(name)+'\n')

with open(inFile,'r') as f:
    file = f.readlines()
    lines = []
    for line in file:
        lines.append(line.split())

inputfile = {}
for i in range(len(lines)):
    if not len(lines[i]) == 0 and len(lines[i]) > 2:
        # print(test[i])
        inputfile[lines[i][0]] = lines[i][2]
    elif not len(lines[i]) == 0 and len(lines[i]) <= 2:
        print('You forgot to give an input for '+lines[i][0])

## SET PARAMETERS
lr          = float(inputfile['lr'])
tot_epochs  = int(inputfile['tot_epochs'])
nb_epochs   = int(inputfile['nb_epochs'])
ini_epochs  = 5
losstype    = inputfile['losstype']
z_dim       = int(inputfile['z_dim'])
dt_fract    = dt_fracts[z_dim]
batch_size  = 1
nb_samples  = int(inputfile['nb_samples'])
n_dim       = 468

print(lr, tot_epochs, nb_epochs, ini_epochs, losstype, z_dim, dt_fract, batch_size, nb_samples, n_dim)

## ================================================== INPUT ========

print('------------------------------')
print('')
print('Training:')
print('Name:', name)
print('------------------------------')
print('      # epochs:', tot_epochs)
print(' learning rate:', lr)
print('# z dimensions:', z_dim)
print('     # samples:', nb_samples)
print('     loss type:', losstype)
print('')

## --------------------------------------- SET UP ------------------

utils.makeOutputDir(path)
utils.makeOutputDir(path+'/nn')

metadata = {'nb_samples'  : nb_samples,
            'lr'        : lr,
            'epochs'    : tot_epochs,
            'z_dim'     : z_dim,
            'done'      : 'false',
            'losstype'  : losstype,
            'inputfile' : arg
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
train, test, data_loader, test_loader = ds.get_data(dt_fract=dt_fract,nb_samples=nb_samples, batch_size=batch_size, kwargs=kwargs)

## Load model
outloc  = '/STER/silkem/MACE/models/CSE_0D/'
dirname = '2023-12-13 18:19:22.522359'  ## lr = 1.e-4       GOOD MODEL
meta, model = utils.load_all(outloc, dirname, sepr = True, epoch = 7) # type: ignore


## --------------------------------------- TRAINING ----------------

## ------------- PART 1: unnormalised losses ----------------
norm = {'mse' : 1,
        'rel' : 1,
        'evo' : 1,
        'idn' : 1}

fract = {'mse' : 1, 
         'rel' : 1,
         'evo' : 1,
         'idn' : 1}


## Make loss objects
trainloss = loss.Loss(norm, fract)
testloss  = loss.Loss(norm, fract)

trainloss.set_losstype(losstype)
testloss.set_losstype(losstype)

tic = time()
tr.train(model, lr, data_loader, test_loader,path, end_epochs = ini_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, plot = False, log = True, show = True)
toc = time()
train_time1 = toc-tic

## ------------- PART 2: normalised losses, but reinitialise model ----------------

## Change the ratio of losses via the fraction
mse1 = float(inputfile['mse1'])
rel1 = float(inputfile['rel1'])
evo1 = float(inputfile['evo1'])
idn1 = float(inputfile['idn1'])
fract = {'mse' : mse1, 
         'rel' : rel1, 
         'evo' : evo1, 
         'idn' : idn1}
trainloss.change_fract(fract)
testloss.change_fract(fract)

## normalise the losses
new_norm = {'mse' :np.mean(trainloss.get_loss('mse')), # type: ignore
            'rel' :np.mean(trainloss.get_loss('rel')), # type: ignore
            'evo' :np.mean(trainloss.get_loss('evo')), # type: ignore
            'idn' :np.mean(trainloss.get_loss('idn'))}   # type: ignore
trainloss.change_norm(new_norm)   
testloss.change_norm(new_norm) 

## Continue training
tic = time()
tr.train(model, lr, data_loader, test_loader,path, end_epochs = nb_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_epochs=ini_epochs, plot = False, log = True, show = True)
toc = time()
train_time2 = toc-tic

## ------------- PART 3: increase losses with factor & train further ----------------

## Change the ratio of losses again via the fraction, but keep the normalisation
mse2 = float(inputfile['mse2'])
rel2 = float(inputfile['rel2'])
evo2 = float(inputfile['evo2'])
idn2 = float(inputfile['idn2'])
fract = {'mse' : mse1*mse2, 
         'rel' : rel1*rel2, 
         'evo' : evo1*evo2, 
         'idn' : idn1*idn2}
trainloss.change_fract(fract)
testloss.change_fract(fract)

## Continue training
tic = time()
tr.train(model, lr, data_loader, test_loader, path, end_epochs = tot_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_epochs=nb_epochs, plot = False, log = True, show = False)
toc = time()
train_time3 = toc-tic

print('\n\tTraining done!')

train_time = train_time1+train_time2+train_time3


## -------------- SAVING THE DATA --------------------
trainpath = path+'/train'
testpath  = path+'/test'

## losses
trainloss.save(trainpath)
testloss.save(testpath)

## dataset characteristics
min_max = np.stack((train.mins, train.maxs), axis=1)
np.save(path+'/minmax', min_max) 

## model
torch.save(model.state_dict(),path+'/nn/nn.pt')

## status
np.save(trainpath+'/status', model.get_status('train')) # type: ignore
np.save(testpath +'/status', model.get_status('test') ) # type: ignore

fig_loss = pl.plot_loss(trainloss, testloss, show = True)
plt.savefig(path+'/loss.png')

stop = time()

overhead_time = (stop-start)-train_time

metadata = {'nb_samples'  : nb_samples,
            'lr'        : lr,
            'epochs'    : tot_epochs,
            'z_dim'     : z_dim,
            'dt_fract'  : train.dt_fract,
            'tmax'      : train.dt_max,
            'train_time_h': train_time/(60*60),
            'overhead_s'  : overhead_time,
            'samples'   : len(train),
            'cutoff_abs': train.cutoff,
            'done'      : 'true',
            'losstype'  : losstype,
            'inputfile' : arg
}

json_object = json.dumps(metadata, indent=4)
with open(path+"/meta.json", "w") as outfile:
    outfile.write(json_object)

print('------------------------------')
print('')
print('Training:')
print('Name:', name)
print('------------------------------')
print('      # epochs:', tot_epochs)
print(' learning rate:', lr)
print('# z dimensions:', z_dim)
print('     # samples:', nb_samples)
print('     loss type:', losstype)
print('')


print('** ',name,' ALL DONE! in [min]', round((train_time + overhead_time)/(60*60),2))



