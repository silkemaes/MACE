
import numpy             as np
import os
from os import listdir
from datetime import datetime
import json
import torch

from torch.utils.data    import Dataset, DataLoader


## own scripts
import plotting     
import utils


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
        
        self.n      = np.load(outpath+dirname+'/'+dir+'/abundances.npy')[:,1:].astype(np.float32)    ## want n_0 dubbel
        self.tstep  = np.load(outpath+dirname+'/'+dir+'/tstep.npy').astype(np.float32)
        input       = np.load(outpath+dirname+'/'+dir+'/input.npy').astype(np.float32)
        self.p      = input[0:-1]

        # ## log10 from rho, T and delta
        # for i in range(3):
        #     self.p[i]  = np.log10(input[i])
        # self.p[3] = input[3]

        ## Clip & log10
        # self.n = np.clip(self.n, cutoff, None)
        # self.n = np.log10(self.n)

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
    def __init__(self, dirname, train=True, fraction=0.7, cutoff = 1e-20, scale = 'norm'):

        outpath = '/STER/silkem/ChemTorch/out/'
        self.dirname = dirname
        self.dirs = listdir(outpath+self.dirname+'/')
        self.dirs.remove('meta.json')

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

        self.mins = np.array([self.logρ_min, self.logT_min, self.logδ_min, self.Av_min])
        self.maxs = np.array([self.logρ_max, self.logT_max, self.logδ_max, self.Av_max])

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


    @staticmethod
    def normalise(x,min,max):
        return (x - min)*(1/np.abs( min - max ))

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
            trans_p[j] = Data.normalise(np.log10(mod.p[j]), self.mins[j], self.maxs[j])
        trans_p[3] = Data.normalise(mod.p[3], self.mins[3], self.maxs[3])

        ## abundances
        trans_n = np.clip(mod.n, self.cutoff, None)
        trans_n = np.log10(trans_n)
        trans_n = Data.normalise(trans_n, self.cutoff, 1)

        ## timesteps
        ## normaliseren? eens nadenken
        trans_tstep = mod.tstep

        # print(trans_n.shape, trans_p.shape, trans_tstep.shape)


        return trans_n, trans_p, trans_tstep


def get_dirs(dirname):
    outpath = '/STER/silkem/ChemTorch/out/'
    return listdir(outpath+dirname+'/')



def get_data(dirname, batch_size, kwargs, plot = False, scale = 'norm'):
    ## Make PyTorch dataset
    train = Data(dirname)
    test  = Data(dirname, train = False)
    
    print('Dataset:')
    print('------------------------------')
    print('total # of samples:',len(train)+len(test))
    print('# training samples:',len(train))
    print('# testing samples: ',len(test) )
    print('            ratio: ',np.round(len(test)/(len(train)+len(test)),2))

    data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True ,  **kwargs)
    test_loader = DataLoader(dataset=test , batch_size=1 , shuffle=False,  **kwargs)

    return train, data_loader, test_loader




class MyDataset_1Dmodel(Dataset):
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

    

def read_data_1Dmodel(file_name):
    '''
    Read data text file of output abundances of 1D CSE models
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
    train = MyDataset_1Dmodel(dir=dir, scale = scale)
    test  = MyDataset_1Dmodel(dir=dir, scale = scale, train = False)
    
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
