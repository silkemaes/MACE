import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import sys
import os
from time import time
import torch



from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})
mpl.rcParams.update({'font.size': 8})
plt.rcParams['figure.dpi'] = 150

## import own functions
sys.path.insert(1, '/STER/silkem/MACE/src/mace')

import dataset          as ds
import loss             as loss
import utils            as utils
import train            as train





# idx = ['0', '1', '2','3', '4', '5', '6']

outloc  = '/STER/silkem/MACE/models/CSE_0D/'

loss_evol = list()
loss_step = list()
time_evol = list()
time_step = list()

dt_fracts = {4 : 0.296, 5: 0.269,8: 0.221,10: 0.175,12: 0.146,16: 0.117,20: 0.09,25: 0.078,32: 0.062,48: 0.043,64: 0.033,128: 0.017}

cuda   = False
DEVICE = torch.device("cuda" if cuda else "cpu")

lr = 1.e-4
epochs = 2

kwargs = {'num_workers': 1, 'pin_memory': True} 
losstype = 'mse_idn'
batch_size = 1
nb_samples = 10000
n_dim = 468
# nb_hidden = int(2)
# ae_type = 'simple'
nb_test = 3000

idx = ['1','2','3','4','5']
for i in range(len(idx)):
# idx = str(sys.argv[1])
# dirname = '20240106_102404_58729_'+idx
    dirname = '20240207_134859_66656_'+idx[i]
    print('\n__'+dirname+'_______________________________________________')  

    sum_step = 0
    sum_evol = 0
    sum_log_err_step = 0
    sum_log_err_evol = 0
    evol_calctime = list()
    step_calctime = list()

    meta, model_testing, trainloss_, testloss_ = utils.load_all(outloc, dirname, epoch = 5) # type: ignore
    trainset, testset, data_loader, test_loader = ds.get_data(dt_fract=dt_fracts[meta['z_dim']],nb_samples=meta['nb_samples'], nb_test = nb_test,batch_size=batch_size, kwargs=kwargs)


    for i in range(len(trainset.testpath)):
        print(i)

        # Save the original standard output
        original_stdout = sys.stdout 

        # Redirect standard output to a null device
        sys.stdout = open(os.devnull, 'w')

        # --- CODE
        testpath = trainset.testpath[i]
        # print(testpath)

        # print('>> Loading test data...')
        physpar, info = ds.get_test_data(testpath,trainset)

        # print('>> Running model')
        n, n_hat, t, mace_step_time = train.test(model_testing, physpar)
        step_calctime.append(mace_step_time)
        n_evol, mace_evol_time = train.test_evolution(model_testing, physpar, start_idx=0)
        evol_calctime.append(mace_evol_time)

        # print('>> Den ormalising abundances...')
        n = ds.get_abs(n)
        n_hat = ds.get_abs(n_hat)
        n_evol = ds.get_abs(n_evol)

        # print('>> Calculating & saving losses...')
        # print('per time step:')
        # mse = loss.mse_loss(n[1:], n_hat)
        # sum_step += mse.sum()
        log_err_step = np.abs((np.log10(n[1:])-np.log10(n_hat))/np.log10(n[1:]))
        sum_log_err_step += log_err_step.sum()


        # print('    evolution:')
        # mse_evol = loss.mse_loss(n[1:], n_evol)
        # sum_evol += mse_evol.sum()
        log_err = np.abs((np.log10(n[1:])-np.log10(n_evol))/np.log10(n[1:]))
        sum_log_err_evol += log_err.sum()

        # Restore standard output
        sys.stdout = original_stdout

    # np.save(outloc+dirname+'/testloss_evol_' + str(len(trainset.testpath))+'.npy', np.array(sum_evol))
    # np.save(outloc+dirname + '/testloss_step_' + str(len(trainset.testpath))+'.npy', np.array(sum_step))
    np.save(outloc+dirname + '/testloss_step_logerr_' + str(len(trainset.testpath))+'.npy', np.array(sum_log_err_evol))
    np.save(outloc+dirname + '/testloss_evol_logerr_' + str(len(trainset.testpath))+'.npy', np.array(sum_log_err_evol))
    # np.save(outloc + dirname + '/calctime_evol_' + str(len(trainset.testpath)) + '.npy', evol_calctime)
    # np.save(outloc + dirname + '/calctime_step_' + str(len(trainset.testpath)) + '.npy', step_calctime)  
    loss_evol.append(sum_evol)
    loss_step.append(sum_step)
    time_evol.append(np.array(evol_calctime))
    time_step.append(np.array(step_calctime))
