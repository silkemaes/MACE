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
import CSE_0D.integrated            as train
# import train            as train



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


epoch = '14'
# testpath = '/lhome/silkem/CHEM/Output_nov2022/20211015_gridC_Mdot1e-6_v17-5_T_eps/models/model_2022-12-24h17-06-51/csfrac_smooth.out'
# testpath = '/lhome/silkem/CHEM/Output_nov2022/20211014_gridC_Mdot2e-8_v2-5_T_eps/models/model_2022-12-26h16-02-06/csfrac_smooth.out'
testpath = '/lhome/silkem/CHEM/Output_nov2022/20210527_gridC_Mdot5e-5_v22-5_T_eps/models/model_2022-12-27h11-01-25/csfrac_smooth.out'

dirs = ['20240207_134859_66656_1',
        '20240207_134859_66656_2',
        '20240207_134859_66656_3',
        '20240207_134859_66656_4',
        '20240207_134859_66656_5',
        '20240208_135604_66879_6',
        '20240208_135604_66879_7',
        '20240208_135604_66879_8',
        '20240208_135604_66879_9']

for dir in dirs:
    print('\n__'+dir+'_______________________________________________') 

    meta, model_testing, trainloss_, testloss_ = utils.load_all(outloc, dir, epoch = epoch) # type: ignore
    trainset, testset, data_loader, test_loader = ds.get_data(dt_fract=dt_fracts[meta['z_dim']],nb_samples=meta['nb_samples'], nb_test = nb_test,batch_size=batch_size, kwargs=kwargs)

    input, info = ds.get_test_data(testpath,trainset)

    n_evol, mace_time, idv_time = train.test_evolution(model_testing, input, start_idx=0)
    tic = time()
    n_evol = ds.get_abs(n_evol)
    toc = time()

    convert_time = toc-tic

    times = np.array([mace_time, convert_time])


    np.save(outloc + dir + '/time_evol3.npy', times)
    np.save(outloc + dir + '/time_idv_evol3.npy', idv_time)

