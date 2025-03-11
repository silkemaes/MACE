import matplotlib.pyplot        as plt
import numpy                    as np      
import sys
import torch
from time                       import time
import datetime                 as dt
from tqdm                       import tqdm
import os

import src.mace.CSE_0D.dataset  as ds
import src.mace.train           as train
import src.mace.test            as test
import src.mace.mace            as mace
from src.mace.loss              import Loss
import src.mace.loss            as loss  
import src.mace.utils           as utils
from src.mace.input             import Input

source_dir = os.path.dirname(os.path.abspath(__file__))

specs_dict, idx_specs = utils.get_specs()

start = time()
now = dt.datetime.now()
name = str(now.strftime("%Y%m%d")+'_'+now.strftime("%H%M%S"))
path = source_dir+'/model/'+name

print('Model path:', path)


## ================================================== INPUT ========
## ADJUST THESE PARAMETERS FOR DIFFERENT MODELS

## READ INPUT FILE
try:
        arg = sys.argv[1]
except Exception:
        print('Please provide an input file.')
        print('$ python routine.py example')
        sys.exit()

infile = source_dir+'/input/'+arg+'.in'

input = Input(infile, name)

input.print()

utils.makeOutputDir(path)
utils.makeOutputDir(path+'/nn')

meta = input.make_meta(path)

## ================================================== SETUP ========

## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")
batch_size = 1 
kwargs = {'num_workers': 1, 'pin_memory': True} 


## Load train & test data sets 
traindata, testdata, data_loader, test_loader = ds.get_data(dt_fract=input.dt_fract,
                                                            nb_samples=input.nb_samples, batch_size=batch_size,
                                                            nb_test=input.nb_test,kwargs=kwargs, inpackage=True)
## Make model
model = mace.Solver(n_dim=input.n_dim, p_dim=4,z_dim = input.z_dim, 
                    nb_hidden=input.nb_hidden, ae_type=input.ae_type, 
                    scheme=input.scheme, nb_evol=input.nb_evol,
                    path = path,
                    DEVICE = DEVICE,
                    lr=input.lr )

num_params = utils.count_parameters(model)
print(f'\nThe model has {num_params} trainable parameters')

## ================================================== TRAIN ========

## ------------- PART 1: unnormalised losses ----------------
norm, fract = loss.initialise()

## Make loss objects
trainloss = Loss(norm, fract, input.losstype)
testloss  = Loss(norm, fract, input.losstype)

## Train
tic = time()
train.train(model,
            data_loader, test_loader,
            end_epochs = input.ini_epochs,
            trainloss=trainloss, testloss=testloss, 
            start_time = start)
toc = time()
train_time1 = toc-tic


## ------------- PART 2: normalised losses, but reinitialise model ----------------

## Change the ratio of losses via the fraction
print('\n\n>>> Continue with normalised losses.')

fract = input.get_facts()
trainloss.change_fract(fract)
testloss.change_fract(fract)

## Normalise the losses
new_norm = trainloss.normalise()  
testloss.change_norm(new_norm) 


## Continue training
tic = time()
train.train(model, 
            data_loader, test_loader, 
            start_epochs = input.ini_epochs, end_epochs = input.nb_epochs, 
            trainloss=trainloss, testloss=testloss, 
            start_time = start)
toc = time()
train_time2 = toc-tic

train_time = train_time1 + train_time2


## ================================================== SAVE ========


## losses
trainloss.save(path+'/train')
testloss.save(path+'/valid')

## dataset characteristics
min_max = np.stack((traindata.mins, traindata.maxs), axis=1)
np.save(path+'/minmax', min_max) 

## model
torch.save(model.state_dict(),path+'/nn/nn.pt')

## status
np.save(path+'/train/status', model.get_status('train')) # type: ignore
np.save(path +'/valid/status', model.get_status('test') ) # type: ignore

fig_loss = loss.plot(trainloss, testloss, len = input.nb_epochs)
plt.savefig(path+'/loss.png')

stop = time()

overhead_time = (stop-start)-train_time

## updating meta file
input.update_meta(traindata, train_time, overhead_time, path)

## ================================================== TEST ========

input.print()

## Test the model on the test samples

print('\n\n>>> Testing model on',len(traindata.testpath),'test samples ...')

sum_err_step = 0
sum_err_evol = 0

step_calctime = list()
evol_calctime = list()

for i in tqdm(range(len(traindata.testpath))):
#     print(i+1,end='\r')
    testpath = traindata.testpath[i]

    err_test, err_evol, step_time, evol_time = test.test_model(model,testpath, meta, printing = False, inpackage=True, datapath='train' )

    sum_err_step += err_test
    sum_err_evol += err_evol

    step_calctime.append(step_time)
    evol_calctime.append(evol_time)

utils.makeOutputDir(path+'/test')

np.save(path+ '/test/sum_err_step.npy', np.array(sum_err_step/len(traindata.testpath)))
np.save(path+ '/test/sum_err_evol.npy', np.array(sum_err_evol/len(traindata.testpath)))

np.save(path+ '/test/calctime_evol.npy', evol_calctime)
np.save(path+ '/test/calctime_step.npy', step_calctime)  

print('\nAverage error:')
print('           Step:', np.round(sum_err_step,3))
print('      Evolution:', np.round(sum_err_evol,3))
print('(following Eq. 23 of Maes et al., 2024)')

stop = time()

print('\n>>> FULLY DONE!')

total_time = stop-start
if total_time < 60:
        print('Total time [secs]:', np.round(total_time,2))
if total_time >= 60:
        print('Total time [mins]:', np.round(total_time/60,2))
if total_time >= 3600:
        print('Total time [hours]:', np.round(total_time/3600,2))

print('Output saved in:', path,'\n')




    