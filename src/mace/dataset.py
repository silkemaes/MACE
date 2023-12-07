
import numpy             as np
import os
from os import listdir
from datetime import datetime
import json
import torch
import sys


from torch.utils.data    import Dataset, DataLoader


## own scripts
import plotting     
import utils

sys.path.append('/STER/silkem/ChemTorch/src')
import rates as rate
## get species
specs, parnt, convs = rate.read_specs_file('C', 16)
specs_dict = dict()
idx_specs  = dict()
for i in range(len(specs)):
    specs_dict[specs[i]] = i
    idx_specs[i] = specs[i]

def normalise(x,min,max):
        # print("Normalising")
        norm = (x - min)*(1/np.abs( min - max ))
        # print(x, norm)
        return norm


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


### ----------------------- 1D CSE models ----------------------- ###



def generate_random_numbers(n, start, end):
    return np.random.randint(start, end, size=n)

class CSEdata(Dataset):
    '''
    Class to initialise the dataset to train & test emulator

    Get data from textfiles (output CSE model)
    
    Preprocess:
        - set all abundances < cutoff to cutoff
        - take np.log10 of abudances

    '''
    def __init__(self,nb_samples, train=True, fraction=0.7, cutoff = 1e-20, scale = 'norm'):
        loc = '/STER/silkem/MACE/'
        paths = np.loadtxt(loc+'data/paths_data_C.txt', dtype=str)

        ## select a certain number of paths, given by nb_samples
        self.idxs = generate_random_numbers(nb_samples, 0, len(paths))
        self.path = paths[self.idxs]

        ## These values are the results from a search through the full dataset; see 'minmax.json' file
        self.logρ_min = np.log10(0.008223)
        self.logρ_max = np.log10(5009000000.0)
        self.logT_min = np.log10(10.)
        self.logT_max = np.log10(1851.0)      
        self.logδ_min = np.log10(0.0)

        self.fraction = fraction
        self.train = train

        np.random.seed(0)
        rand_idx = np.random.permutation(len(self.path))
        N = int(self.fraction*len(self.path))
        if self.train:
            self.rand_idx = rand_idx[:N]
        else:
            self.rand_idx = rand_idx[N:]
            
            
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        idx = self.rand_idx[idx]

        mod = CSEmod(self.path[idx])

        Δt, n, p = mod.split_in_0D()


        return 



class CSEmod():
    '''
    Class to initialise the dataset to train & test emulator

    Get data from textfiles (output CSE model)
    
    Preprocess:
        - set all abundances < cutoff to cutoff
        - take np.log10 of abudances

    '''
    def __init__(self, path):

        self.path = path[0:-17]
        abs_path = 'csfrac_smooth.out'
        phys_path = 'csphyspar_smooth.out'
        self.name = path[-43:-18]
        inp_path = 'inputChemistry_'+self.name+'.txt'

        ## retrieve input
        self.Rstar, self.Tstar, self.Mdot, self.v, self.eps, self.rtol, self.atol = read_input_1Dmodel(self.path[:-26]+inp_path)

        ## retrieve abundances
        abs = read_data_1Dmodel(self.path+abs_path)
        self.n = abs

        ## retrieve physical parameters
        arr = np.loadtxt(self.path+phys_path, skiprows=4, usecols=(0,1,2,3,4))
        self.radius, self.dens, self.temp, self.Av, self.delta = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        self.time = self.radius/(self.v) 

    def __len__(self):
        return len(self.time)

    def get_time(self):
        return self.time
    
    def get_phys(self):
        return self.dens, self.temp, self.delta, self.Av
    
    def get_abs(self):
        return self.n
    
    def get_abs_spec(self,spec):
        i = specs_dict[spec]
        return self.n.T[i]
    
    def get_dens(self):
        return self.dens
    
    def get_temp(self):
        return self.temp

    def get_Av(self):
        return self.Av
    
    def get_delta(self):
        return self.delta

    def get_vel(self):
        return self.v
    
    def get_path(self):
        return self.path

    def get_name(self):
        return self.name
    
    def get_dt(self):
        return self.time[1:] - self.time[:-1]
    
    def split_in_0D(self):
        Δt = self.get_dt()
        n_0D = self.n[:-1]
        p = np.array([self.dens[:-1], self.temp[:-1], self.delta[:-1], self.Av[:-1]])
        return Δt, n_0D, p.T
        


def read_input_1Dmodel(file_name):
    with open(file_name) as file:
        lines = file.readlines()
        lines = [item.rstrip() for item in lines]

    Rstar = float(lines[3][9:])
    Tstar = float(lines[4][9:])
    Mdot  = float(lines[5][8:])     ## Msol/yr
    v     = float(lines[6][11:])    ## sec
    eps   = float(lines[8][19:])

    rtol = float(lines[31][7:])
    atol = float(lines[32][6:])

    return Rstar, Tstar, Mdot, v, eps, rtol, atol

def read_data_1Dmodel(file_name):
    '''
    Read data text file of output abundances of 1D CSE models.
    '''
    with open(file_name, 'r') as file:
        dirty = []
        proper = None
        for line in file:
            try:  
                if len(line) > 1: 
                    dirty.append([float(el) for el in line.split()])
            except:
                if len(dirty) != 0:
                    dirty = np.array(dirty)[:,1:]
                    if proper is None:
                        proper = dirty
                    else:
                        proper = np.concatenate((proper, dirty), axis = 1)
                dirty = []
    return proper



def retrieve_file(dir_name):
    all_paths_C = []
    all_paths_O = []

    path = '/lhome/silkem/CHEM/'+dir_name+'/'
    locs = utils.get_files_in(path)
    
    for loc in locs:
        # print(loc[-3:-1])
        if loc[-3:-1] == 'ep':
            # print(path+loc+'/models/')
            path_mods = utils.get_files_in(path+loc+'/models/')
            for mod in path_mods:
                if mod[-1] != 't':
                    if loc[13] == 'O':
                        all_paths_O.append(path+loc+'/models/'+mod+'/csfrac_smooth.out')
                    if loc[13] == 'C':
                        all_paths_C.append(path+loc+'/models/'+mod+'/csfrac_smooth.out')
    
    return all_paths_O, all_paths_C


def get_dataset(dir, batch_size, kwargs, plot = False, scale = 'norm'):
    ## Make PyTorch dataset
    train = MyDataset_1Dmodel_old(dir=dir, scale = scale)
    test  = MyDataset_1Dmodel_old(dir=dir, scale = scale, train = False)
    
    print('Dataset:')
    print('------------------------------')
    print('total # of samples:',len(train)+len(test))
    print('# training samples:',len(train))
    print('# testing samples: ',len(test) )
    print('            ratio: ',np.round(len(test)/(len(train)+len(test)),2))

    data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True ,  **kwargs)
    test_loader = DataLoader(dataset=test , batch_size=len(test) , shuffle=False,  **kwargs)

    if plot == True:
        print('\nPlotting histrogram of dataset...')
        plotting.plot_hist(train.df)

    return train, data_loader, test_loader

class MyDataset_1Dmodel_old(Dataset):
    '''
    Class to initialise the dataset to train & test emulator

    Get data from textfiles (output CSE model)
    
    Preprocess:
        - set all abundances < cutoff to cutoff
        - take np.log10 of abudances

    '''
    def __init__(self, dir=None, file=None, train=True, fraction=0.7, cutoff = 1e-40, scale = 'norm'):
        data = []

        if dir != None:
            locs = os.listdir(dir) 

            for i in range(1,len(locs)+1):
                name = dir+'csfrac_smooth_'+str(i)+'.out'
                proper = read_data_1Dmodel(name)
                data.append(proper)
        
        if file != None:
            proper = read_data_1Dmodel(file)
            data.append(proper)

        df = np.concatenate(data)
        
        ## Clip and take log10
        self.df = np.clip(df, cutoff, 1)
        self.df = np.log10(self.df)

        ## Statistics of the data
        self.mean = np.mean(self.df)
        self.std  = np.std(self.df)
        self.min  = np.min(self.df)
        self.max  = np.max(self.df)

        ## Normalise 
        if scale == 'norm':
            self.df = (self.df - self.mean ) / (self.std)

        ## Scale
        if scale == 'minmax':
            self.df = (self.df - self.min) / (np.abs( self.min - self.max ))
        
        ## Original data
        if scale == None:
            self.df = self.df

        ## Set type
        self.df   = self.df.astype(np.float32)
        self.mean = self.mean.astype(np.float32)
        self.std  = self.std.astype(np.float32)
        self.min  = self.min.astype(np.float32)
        self.max  = self.max.astype(np.float32)

        ## Split training - testing data
        N = int(fraction * self.df.shape[0])
        if train:
            self.df = self.df[:N]
        else:
            self.df = self.df[N:]
            
            
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df[idx]

    def get_stats(self):
        return self.mean, self.std, self.min, self.max
