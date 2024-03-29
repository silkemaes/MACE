
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
import CSE_0D.intregr_train      as tr
import neuralODE         as nODE
import utils             as utils
import CSE_0D.plotting   as pl
import CSE_0D.loss       as loss


torch.multiprocessing.set_sharing_strategy('file_system')

start = time()
now = dt.datetime.now()
name = str(sys.argv[1])
logname = name[-7:]
path = '/STER/silkem/MACE/models/CSE_0D/'+name
outloc = '/STER/silkem/MACE/models/CSE_0D/'
# logpath = '/home/silkem/mace_logs/log/grid_'+logname+'.log'

meta, model, trainloss, testloss = utils.load_all(outloc, name, epoch = str(sys.argv[2]))

dt_fracts = {4 : 0.296, 5: 0.269,8: 0.221,10: 0.175,12: 0.146,16: 0.117,20: 0.09,25: 0.078,32: 0.062,48: 0.043,64: 0.033,128: 0.017}


## ================================================== INPUT ========
## ADJUST THESE PARAMETERS FOR DIFFERENT MODELS
arg = meta['inputfile']

## READ INPUT FILE
inFile = '/STER/silkem/MACE/input/'+arg+'.in'
print(inFile)

with open(inFile, 'a') as file:
    file.write('\nName = '+name+'\n')

with open(inFile,'r') as f:
    file = f.readlines()
    lines = []
    for line in file:
        lines.append(line.split())

inputfile = {}
for i in range(len(lines)):
    if not len(lines[i]) == 0 and len(lines[i]) > 2:
        inputfile[lines[i][0]] = lines[i][2]
    elif not len(lines[i]) == 0 and len(lines[i]) <= 2:
        print('You forgot to give an input for '+lines[i][0])

# print(inputfile)
## SET PARAMETERS
lr          = float(inputfile['lr'])
tot_epochs  = int(inputfile['tot_epochs'])
nb_epochs   = int(inputfile['nb_epochs'])
ini_epochs  = 2
restart_epoch = (int(sys.argv[2])+1)*10

losstype    = inputfile['losstype']
z_dim       = int(inputfile['z_dim'])
dt_fract    = dt_fracts[z_dim]
batch_size  = 1
nb_samples  = int(inputfile['nb_samples'])
nb_evol     = int(inputfile['nb_evol'])
n_dim       = 468
nb_hidden   = int(inputfile['nb_hidden'])
ae_type     = str(inputfile['ae_type'])
nb_test     = 3000

# print(lr, tot_epochs, nb_epochs, ini_epochs, losstype, z_dim, dt_fract, batch_size, nb_samples, n_dim)

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
print('  # evolutions:', nb_evol)
print('     loss type:', losstype)
print('      # hidden:', nb_hidden)
print('       ae type:', ae_type)
print('     inputfile:', arg)
print('')

## --------------------------------------- SET UP ------------------

utils.makeOutputDir(path)
utils.makeOutputDir(path+'/nn')

metadata = {'nb_samples'  : nb_samples,
            'lr'        : lr,
            'epochs'    : tot_epochs,
            'z_dim'     : z_dim,
            'losstype'  : losstype,
            'inputfile' : arg,
            'nb_evol'   : nb_evol,
            'nb_hidden' : nb_hidden,
            'ae_type'   : ae_type,
            'done'      : 'false',
}

if 'node_1' in meta:
    metadata['node_1'] = meta['node_1']
else:
    metadata['node_1'] = meta['node']

if 'node_2' in meta:
    metadata['node_2'] = meta['node_2']
else:
    metadata['node_2'] = sys.argv[3]

json_object = json.dumps(metadata, indent=4)
with open(path+"/meta.json", "w") as outfile:
    outfile.write(json_object)

## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1 ## if not 1, dan kan er geen tensor van gemaakt worden

kwargs = {'num_workers': 1, 'pin_memory': True} 


## Load train & test data sets 
trainset, testset, data_loader, test_loader = ds.get_data(dt_fract=dt_fract,nb_samples=nb_samples, batch_size=batch_size, nb_test=nb_test,kwargs=kwargs)

## Make model
# model = nODE.Solver(p_dim=4,z_dim = z_dim, n_dim=n_dim, nb_hidden=nb_hidden, ae_type=ae_type, DEVICE = DEVICE)
# print(model.encoder)

## --------------------------------------- TRAINING ----------------

## ------------- PART 1: unnormalised losses ----------------
norm = {'mse' : 1,
        'rel' : 1,
        'grd' : 1,
        'idn' : 1}

fract = {'mse' : 1, 
         'rel' : 1,
         'grd' : 1,
         'idn' : 1}


## Make loss objects
trainloss = loss.Loss(norm, fract)
testloss  = loss.Loss(norm, fract)

trainloss.set_losstype(losstype)
testloss.set_losstype(losstype)

if restart_epoch == 250:
    train_time1 = 0
else:
    tic = time()
    tr.train(model, lr, data_loader, test_loader, nb_evol=nb_evol, path=path, end_epochs = ini_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_time=start)

    toc = time()
    train_time1 = toc-tic

## ------------- PART 2: normalised losses, but reinitialise model ----------------

## Change the ratio of losses via the fraction
mse1 = float(inputfile['mse1'])
rel1 = float(inputfile['rel1'])
grd1 = float(inputfile['evo1'])
idn1 = float(inputfile['idn1'])
fract = {'mse' : mse1, 
         'rel' : rel1, 
         'grd' : grd1, 
         'idn' : idn1}
trainloss.change_fract(fract)
testloss.change_fract(fract)

## normalise the losses
new_norm = {'mse' :np.mean(trainloss.get_loss('mse')), # type: ignore
            'rel' :np.mean(trainloss.get_loss('rel')), # type: ignore
            'grd' :np.mean(trainloss.get_loss('grd')), # type: ignore
            'idn' :np.mean(trainloss.get_loss('idn'))}   # type: ignore
trainloss.change_norm(new_norm)   
testloss.change_norm(new_norm) 

if restart_epoch <= nb_epochs:

    ## Continue training
    tic = time()
    tr.train(model, lr, data_loader, test_loader, nb_evol=nb_evol, path=path, end_epochs = nb_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_epochs=restart_epoch, start_time=start)
    toc = time()
    train_time2 = toc-tic

    ## Change the ratio o   f losses again via the fraction, but keep the normalisation
    mse2 = float(inputfile['mse2'])
    rel2 = float(inputfile['rel2'])
    grd2 = float(inputfile['evo2'])
    idn2 = float(inputfile['idn2'])
    fract = {'mse' : mse1*mse2, 
            'rel' : rel1*rel2, 
            'grd' : grd1*grd2, 
            'idn' : idn1*idn2}
    trainloss.change_fract(fract)
    testloss.change_fract(fract)

    ## Continue training
    tic = time()
    tr.train(model, lr, data_loader, test_loader, nb_evol=nb_evol, path=path, end_epochs = tot_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_epochs=nb_epochs, start_time=start)
    toc = time()
    train_time3 = toc-tic

else:
    train_time2 = 0


## ------------- PART 3: increase losses with factor & train further ----------------

if restart_epoch > nb_epochs:

    ## Change the ratio o   f losses again via the fraction, but keep the normalisation
    mse2 = float(inputfile['mse2'])
    rel2 = float(inputfile['rel2'])
    grd2 = float(inputfile['evo2'])
    idn2 = float(inputfile['idn2'])
    fract = {'mse' : mse1*mse2, 
            'rel' : rel1*rel2, 
            'grd' : grd1*grd2, 
            'idn' : idn1*idn2}
    trainloss.change_fract(fract)
    testloss.change_fract(fract)

    ## Continue training
    tic = time()
    tr.train(model, lr, data_loader, test_loader, nb_evol=nb_evol, path=path, end_epochs = tot_epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_epochs=restart_epoch, start_time=start)
    toc = time()
    train_time3 = toc-tic

else:
    train_time3 = 0

print('\n\tTraining done!')

train_time = train_time1+train_time2+train_time3


## -------------- SAVING THE DATA --------------------
trainpath = path+'vim 2'
testpath  = path+'/test2'

## losses
trainloss.save(trainpath)
testloss.save(testpath)

## dataset characteristics
min_max = np.stack((trainset.mins, trainset.maxs), axis=1)
np.save(path+'/minmax', min_max) 

## model
torch.save(model.state_dict(),path+'/nn/nn.pt')

## status
np.save(trainpath+'/status', model.get_status('train')) # type: ignore
np.save(testpath +'/status', model.get_status('test') ) # type: ignore

fig_loss = pl.plot_loss(trainloss, testloss, show = False)
plt.savefig(path+'/loss.png')

stop = time()

overhead_time = (stop-start)-train_time

metadata = {'nb_samples'  : nb_samples,
            'lr'        : lr,
            'epochs'    : tot_epochs,
            'z_dim'     : z_dim,
            'dt_fract'  : trainset.dt_fract,
            'tmax'      : trainset.dt_max,
            'train_time_h': train_time/(60*60),
            'overhead_s'  : overhead_time,
            'samples'   : len(trainset),
            'cutoff_abs': trainset.cutoff,
            'losstype'  : losstype,
            'inputfile' : arg,
            'nb_evol'   : nb_evol,
            'nb_hidden' : nb_hidden,
            'ae_type'   : ae_type,
            'node'      : sys.argv[3],
            'done'      : 'true'
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
print('')\

print('------------------------------')
print('>> Testing the model')

sum_step = 0
sum_evol = 0
evol_calctime = list()
step_calctime = list()

for i in range(len(trainset.testpath)):
    print(i,end='\r')

    testpath = trainset.testpath[i]
    # print(testpath)

    # print('>> Loading test data...')
    physpar, info = ds.get_test_data(testpath,trainset)

    n, n_hat, t, mace_step_time = tr.test(model, physpar)
    # print('>> Running model')
    step_calctime.append(mace_step_time)
    n_evol, mace_evol_time = tr.test_evolution(model, physpar, start_idx=0)
    evol_calctime.append(mace_evol_time)

    # print('>> Den ormalising abundances...')
    n = ds.get_abs(n)
    n_hat = ds.get_abs(n_hat)
    n_evol = ds.get_abs(n_evol)

    # print('>> Calculating & saving losses...')
    # print('per time step:')
    mse = loss.mse_loss(n[1:], n_hat)
    sum_step += mse.sum()

    # print('    evolution:')
    mse_evol = loss.mse_loss(n[1:], n_evol)
    sum_evol += mse_evol.sum()

np.save(path+ '/testloss_evol_' + str(len(trainset.testpath)) + '.npy', np.array(sum_evol))
np.save(path+ '/testloss_step_' + str(len(trainset.testpath)) + '.npy', np.array(sum_step))
np.save(path+ '/calctime_evol_' + str(len(trainset.testpath)) + '.npy', evol_calctime)
np.save(path+ '/calctime_step_' + str(len(trainset.testpath)) + '.npy', step_calctime)  



print('** ',name,' ALL DONE! in [hours]', round((train_time + overhead_time)/(60*60),2))



