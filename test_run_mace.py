import matplotlib.pyplot as plt
import matplotlib        as mpl
import sys
import torch
from time import time

## import own functions
sys.path.insert(1, '/STER/silkem/MACE/src/mace')

import src.mace.CSE_0D.dataset          as ds
import src.mace.CSE_0D.local_train      as local
import src.mace.mace                    as mace
from src.mace.CSE_0D.loss               import Loss
import src.mace.utils                   as utils



specs_dict, idx_specs = utils.get_specs()

dt_fracts = {4 : 0.296, 5: 0.269,8: 0.221,10: 0.175,12: 0.146,16: 0.117,20: 0.09,25: 0.078,32: 0.062,48: 0.043,64: 0.033,128: 0.017}




## Set up PyTorch 
cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} 

lr = 1.e-4
epochs = 20
batch_size = 1
nb_test = 300
n_dim = 468


losstype = 'abs_idn'
z_dim = 8
nb_samples = 300
nb_hidden = 1
ae_type = 'simple'


print('------------------------------')
print('      # epochs:', epochs)
print(' learning rate:', lr)
print('# z dimensions:', z_dim)
print('     # samples:', nb_samples)
print('     loss type:', losstype)
print('')


train, test, data_loader, test_loader = ds.get_data(dt_fract=dt_fracts[z_dim],nb_samples=nb_samples,nb_test = nb_test, batch_size=batch_size, kwargs=kwargs)

## Local model

model = mace.Solver(p_dim=4,z_dim = z_dim, n_dim=n_dim,nb_hidden=nb_hidden, ae_type=ae_type, DEVICE = DEVICE)

num_params = utils.count_parameters(model)
print(f'The model has {num_params} trainable parameters\n')


norm = {'abs' : 1,
        'grd' : 1,
        'idn' : 1}

fract = {'abs' : 1,
         'grd' : 1,
         'idn' : 1}

plot = True


## Make loss objects
trainloss = Loss(norm, fract, losstype)
testloss  = Loss(norm, fract, losstype)


tic = time()
opt = local.train(model, lr, data_loader, test_loader, path = None, end_epochs = epochs, DEVICE= DEVICE, trainloss=trainloss, testloss=testloss, start_time = time(), plot=plot, show = False)
toc = time()

print('Total time [s]:',toc-tic)


print(model.get_status('train'))

testpath = test.testpath[0]
# print(testpath)

# print('>> Loading test data...')
input_data, info = ds.get_test_data(testpath,test)

n_evol, mace_evol_time = local.test_evolution(model, input_data, start_idx=0)

plt.plot(n_evol, lw = 1)
plt.plot(input_data[0], 'k--', alpha = 0.2, lw = 0.5)


plt.show()