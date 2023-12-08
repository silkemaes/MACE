
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
