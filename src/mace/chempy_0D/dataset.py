
import numpy             as np
from os import listdir
from datetime import datetime
import json
import torch



from torch.utils.data    import Dataset, DataLoader


## own scripts
import plotting     
import utils


### ----------------------- ChemPy/ChemTorch models ----------------------- ###
class ChemTorchMod():
    '''
    Object representing 1 ChemTorch model.
    Contains:
        - n     [2d np.array]: Abundances at different timesteps
        - tstep [1d np.array]: Timesteps that the classical ODE solver is evaluated
        - p     [1d np.array]: input of the model -> [rho, T, delta, Av]
    '''
    def __init__(self, dirname, dir=None):
        outpath = '/STER/silkem/ChemTorch/out/'

        self.dir = dirname+'/'+dir
        
        self.n      = np.load(outpath+dirname+'/'+dir+'/abundances.npy')[:,1:].astype(np.float32)    # type: ignore ## want n_0 dubbel
        self.tstep  = np.load(outpath+dirname+'/'+dir+'/tstep.npy').astype(np.float32) # type: ignore
        input       = np.load(outpath+dirname+'/'+dir+'/input.npy').astype(np.float32) # type: ignore
        self.p      = input[0:-1]
        self.tictoc = np.load(outpath+dirname+'/'+dir+'/tictoc.npy').astype(np.float32) # type: ignore

    def __len__(self):
        return len(self.tstep)


class Data(Dataset):
    '''
    Dataset for training MACE.
    Contains:
        - n     [2d np.array]: Abundances at different timesteps
        - tstep [1d np.array]: Timesteps that the classical ODE solver is evaluated
        - p     [1d np.array]: input of the model -> [rho, T, delta, Av]  
        where different ChemTorchMods are stuck together.
        - idx   [1d np.array]: Containing the indixes of at what location in n, tstep and p a new ChemTorchMod starts.

        This idx is used in the __getitem()__ function.
    '''
    def __init__(self, dirname, dt_fract, train=True, fraction=0.7, cutoff = 1e-20, perc = False):

        outpath = '/STER/silkem/ChemTorch/out/'
        self.dirname = dirname
        self.dirs = listdir(outpath+self.dirname+'/')
        self.dirs.remove('meta.json')

        ## ONLY FOR TESTING
        if perc == True:
            self.dirs = self.dirs[0:1000]

        # Opening JSON file
        with open(outpath+self.dirname+'/meta.json', 'r') as file:
            # Reading from json file
            self.meta = json.load(file)

        self.logρ_min = np.log10(10)
        self.logρ_max = np.log10(1e10)
        self.logT_min = np.log10(10)
        self.logT_max = np.log10(4000)
        self.logδ_min = np.log10(self.meta['delta_min'])
        self.logδ_max = np.log10(self.meta['delta_max'])
        self.Av_min = self.meta['Av_min']
        self.Av_max = self.meta['Av_max']
        self.tmax   = self.meta['tmax']
        self.dt_fract = dt_fract     
        self.n_min = np.log10(cutoff)
        self.n_max = np.log10(0.85e-1)    ## initial abundance He

        self.mins = np.array([self.logρ_min, self.logT_min, self.logδ_min, self.Av_min, self.n_min, self.dt_fract])
        self.maxs = np.array([self.logρ_max, self.logT_max, self.logδ_max, self.Av_max, self.n_max, self.tmax])

        self.cutoff = cutoff
        self.fraction = fraction
        self.train = train

        np.random.seed(0)
        rand_idx = np.random.permutation(len(self.dirs))
        N = int(self.fraction*len(self.dirs))
        if self.train:
            self.rand_idx = rand_idx[:N]
        else:
            self.rand_idx = rand_idx[N:]


    def __len__(self):
        return len(self.rand_idx)


    def __getitem__(self,i):
        '''
        Returns an item of Data --> similar content as a ChemTorchMod instance. 
        
        The self.idx array has stored at what index in Data a new ChemTorchMod instance starts, 
        needed to get a certain item i.
        '''

        idx = self.rand_idx[i]

        mod = ChemTorchMod(self.dirname,self.dirs[idx])

        ## physical parameters
        trans_p = np.empty_like(mod.p)
        for j in range(3):
            trans_p[j] = normalise(np.log10(mod.p[j]), self.mins[j], self.maxs[j])
        trans_p[3] = normalise(mod.p[3], self.mins[3], self.maxs[3])

        ## abundances
        trans_n = np.clip(mod.n, self.cutoff, None)
        trans_n = np.log10(trans_n)
        trans_n = normalise(trans_n, self.n_min, self.n_max)       ## max boundary = rel. abundance of He

        ## timesteps
        ## normaliseren? eens nadenken: JA! --> herschalen
        trans_tstep = mod.tstep
        trans_tstep = trans_tstep/self.tmax * self.dt_fract             ## scale to [0,1] and multiply with dt_fract
 


        return torch.from_numpy(trans_n), torch.from_numpy(trans_p), torch.from_numpy(trans_tstep)

    def tictoc(self,i):
        idx = self.rand_idx[i]

        mod = ChemTorchMod(self.dirname,self.dirs[idx])

        return mod.tictoc[0]
    
    def get_test(self,i):

        idx = self.rand_idx[i] 

        mod = ChemTorchMod(self.dirname,self.dirs[idx])

        n_trans, p_trans, t_trans = self[i]

        print(n_trans.shape, p_trans.shape, t_trans.shape)

        return n_trans.view(1,n_trans.shape[0],-1), p_trans.view(1,-1), t_trans.view(1,-1), mod



def get_dirs(dirname):
    outpath = '/STER/silkem/ChemTorch/out/'
    return listdir(outpath+dirname+'/')



def get_data(dirname, dt_fract,batch_size, kwargs, perc = False):
    ## Make PyTorch dataset
    train = Data(dirname, dt_fract=dt_fract, perc = perc)
    test  = Data(dirname, dt_fract=dt_fract, train = False, perc = perc)
    
    print('Dataset:')
    print('------------------------------')
    print('total # of samples:',len(train)+len(test))
    print('# training samples:',len(train))
    print('#  testing samples:',len(test) )
    print('             ratio:',np.round(len(test)/(len(train)+len(test)),2))

    data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False ,  **kwargs)
    test_loader = DataLoader(dataset=test , batch_size=1 , shuffle=False,  **kwargs)

    return train, test, data_loader, test_loader



def get_test_data(dirname, dt_fract):
    dataset = Data(dirname, dt_fract=dt_fract)
    idx = np.random.randint(0,len(dataset))

    print(idx)

    n_in,p_in,t_in, Chempy_mod = dataset.get_test(idx)

    input = n_in, p_in, t_in

    return input, dataset, Chempy_mod
