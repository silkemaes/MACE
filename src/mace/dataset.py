
import numpy             as np
import os
from os import listdir
from datetime import datetime

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
    def __init__(self, dir=None, cutoff = 1e-40):
        outpath = '/STER/silkem/ChemTorch/out/'
        
        self.n      = np.load(outpath+'new/'+dir+'/abundances.npy')
        self.tstep  = np.load(outpath+'new/'+dir+'/tstep.npy')
        input       = np.load(outpath+'new/'+dir+'/input.npy')
        self.p      = input[0:-1]

        ## Clip
        self.n = np.clip(self.n, cutoff, 1)
        ## Will we take the log10?
        # self.n = np.log10(self.n)

    def __len__(self):
        return len(self.tstep)


class Data(Dataset):
    def __init__(self, dirs, train=True, fraction=0.7, scale = 'norm'):

        df = []
        times = []
        self.idx = np.zeros(len(dirs)+1)
        self.p   = np.zeros(len(dirs)+1)

        for i in range(len(dirs)):
            mod = ChemTorchMod(dirs[i])
            df.append(mod.n)
            times.append(mod.tstep)
            self.idx[i+1] = len(mod)
            self.p[i] = mod.p
        
        self.n      = np.concatenate(df)
        self.tstep  = np.concatenate(times)

        ## Hoe data normaliseren?


    def __len__(self):
        return len(self.idx)-1

    def __getitem__(self,i):
        ## hier uiteindelijk terug ChemTorchMod instance teruggeven?
        start = self.idx[i-1]
        stop  = start + self.idx[i]

        if i > len(self):
            return None

        return self.n[start:stop], self.tstep[start:stop], self.p[i]
    


def get_dirs():
    outpath = '/STER/silkem/ChemTorch/out/'
    return listdir(outpath+'new/')



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
    train = MyDataset(dir=dir, scale = scale)
    test  = MyDataset(dir=dir, scale = scale, train = False)
    
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