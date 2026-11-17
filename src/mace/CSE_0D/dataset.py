'''
This script contains classes for loading the 1D CSE models,
which is the training data for MACE,
as well as code to preprocess the data.


Contains:
    Classes:
        - ChemData: Dataset class for chemical data to prepare the dataset for training and validating the emulator
        - CSEmod: Class to load a model
        - PhantomData + Phantommod: Dataset and model class for Phantom models
        - CSEdata + CSEmod: Dataset and model class for 1D CSE models
    
    Functions:
        - get_data: prepare the data for training and validating the emulator, using the PyTorch-specific dataloader
        - get_test_data: get the data of the test a 1D model, given a path and meta-data from a training setup
        - get_abs: get the abundances, given the normalised abundances
        - get_phys: reverse the normalisation of the physical parameters

NOTE:
This script only works for this specific dataset.

'''
import json
import os

import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
import src.mace.utils as utils
from pathlib import Path

specs_dict = utils.get_specs()

### Base class for models and datasets

class ChemModel:
    ''' 
    Basic class to load chemical model: time, abundances, and physical parameters (rho, T, Xi and AV)
    For a new type of model, define a new subclass with its own loading method.
    '''
    def __len__(self):     return len(self.time)
    def get_time(self):    return self.time
    def get_phys(self):    return self.dens, self.temp, self.xi, self.Av
    def get_abs(self):     return self.n
    def get_dens(self):    return self.dens
    def get_temp(self):    return self.temp
    def get_Av(self):      return self.Av
    def get_xi(self):      return self.xi
    def get_path(self):    return self.path
    def get_name(self):    return self.name
    def get_dt(self):      return self.time[1:] - self.time[:-1]

    def split_in_0D(self):
        '''
        Split the 1D model in 0D models (just a collection of separate points)
        '''
        delta_t = self.get_dt()
        n_0D = self.get_abs()
        y = 1.e-100  ## this number is added to xi, since it can contain zeros
        p = np.array([
            self.get_dens()[:-1],
            self.get_temp()[:-1],
            self.get_xi()[:-1] + y,
            self.get_Av()[:-1] + y
        ])

        return delta_t.astype(np.float64), n_0D.astype(np.float64), p.T.astype(
            np.float64)

class ChemData(Dataset):
    ''' basic class to initialise the dataset for PyTorch.
        subclasses for each type of datasets are defined after this class.
        For each new dataset subclass, define how it loads data using load_model(self, path)
    '''
    def __init__(self, meta, train=True, cutoff=1e-20):
        '''
        Initialising the attributes of the dataset.

        Input:
            - meta [object]: contains dataset metadata
            - train [boolean]: True if training, False if validating
            - cutoff [float]: cutoff value for the abundances, depends on tolerances of classical chemistry kinetics solver, 
                default = 1e-20

        Preprocess on data:
            - clip all abundances to cutoff
            - take np.log10 of abudances
            - normalise data to log10 of min and max values

        Structure:
            1. Load the paths of the models
            2. Select a certain number of paths, split in nb_train, nb_test, nb_valid and shuffle according to seed
            3. Set the min and max values of the physical parameters and abundances, 
                resulting from a search through the full dataset; see 'minmax.json' file.
                These values are used for normalisation of the data. 
                --> self.mins, self.maxs
        '''
        print('> Train state:', train)
        loc = str(Path().cwd()) + '/'
        self.cutoff, self.train = cutoff, train

        # grab paths listed in data file
        # atleast_1d to ensure path is an array (loadtxt returns an array scalar (0D array) if only one path) 
        paths = self.find_paths(loc, meta)
        print('Paths found:', len(paths))
        self._split_paths(paths, meta, train)
        self.set_minmax(meta, loc+'data/'+ meta.minmax)
        

    def _split_paths(self, paths, meta, train):
        '''
        Split paths into train/valid/test randomly
        '''
        nb_samples = meta.nb_train + meta.nb_valid + meta.nb_test
        if nb_samples > len(paths):
            raise ValueError('Not enough models in paths_train_data.txt for training, validation and testing.')
        np.random.seed(meta.seed)
        perm = np.random.permutation(len(paths))
        train_idx = perm[:meta.nb_train]
        valid_idx = perm[meta.nb_train:meta.nb_train + meta.nb_valid]
        test_idx  = perm[meta.nb_train + meta.nb_valid:nb_samples]
        if train:
            self.path = paths[train_idx]
            print('Train paths:', len(self.path))
        else:
            self.path = paths[valid_idx]
            print('Valid paths:', len(self.path))
        self.testpath = paths[test_idx]
        print('Test paths:', len(self.testpath))

    def set_minmax(self, meta, file):
        '''
        Set min and max values for normalisation
        '''
        ## These values are the results from a search through the full dataset; see 'minmax.json' file
        with open(file, 'r') as f:
            minmax = json.load(f)
        self.logrho_min = np.log10(minmax['dens_min'])
        self.logrho_max = np.log10(minmax['dens_max'])
        self.logT_min = np.log10(minmax['temp_min'])
        self.logT_max = np.log10(minmax['temp_max'])
        y = 1.e-100  ## this number is added to xi and AV, since they can contain zeros
        self.logdelta_min = np.log10(y + minmax['delta_min'])
        self.logdelta_max = np.log10(y + minmax['delta_max'])
        self.Av_min = np.log10(y + minmax['Av_min'])
        self.Av_max = np.log10(y + minmax['Av_max'])
        self.dt_max = minmax['dt_max']
        self.dt_fract = meta.dt_fract
        self.n_min = np.log10(self.cutoff)
        self.n_max = np.log10(0.85e-1)  ## initial abundance He
        self.mins = np.array([
            self.logrho_min, self.logT_min, self.logdelta_min, self.Av_min,
            self.n_min, self.dt_fract
        ])
        self.maxs = np.array([
            self.logrho_max, self.logT_max, self.logdelta_max, self.Av_max,
            self.n_max, self.dt_max
        ])
    
    def preprocess(self, mod):
        '''
        Preprocess data in model: clip to cutoff value, log10, then normalise
        '''
        delta_t, n, p = mod.split_in_0D()
        # physical params
        p_transf = np.empty_like(p)
        for j in range(p.shape[1]):
            p_transf[:, j] = utils.normalise(np.log10(p[:, j]), self.mins[j], self.maxs[j])
        # abundances
        n_transf = utils.normalise(np.log10(np.clip(n, self.cutoff, None)), self.n_min, self.n_max)
        # timesteps
        delta_t_transf = delta_t / self.dt_max * self.dt_fract
        return (torch.from_numpy(n_transf), torch.from_numpy(p_transf),
                torch.from_numpy(delta_t_transf))
    
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        return self.preprocess(self.load_model(self.path[idx]))
    
### ----------------------- 1D CSE models ----------------------- ###

class CSEdata(ChemData):
    '''
    1D CSE models
    '''
    def find_paths(self, loc, meta):
        return np.atleast_1d(np.loadtxt(loc + 'data/' + meta.paths, dtype=str))
    
    def load_model(self, path):
        print('Loading 1D CSE model from', path)
        return CSEmod(path)

class CSEmod(ChemModel):
    '''
    Class to load a 1D CSE model, calculated with the classical fortan code.
    For more info on this model, see https://github.com/MarieVdS/rate22_cse_code.
    '''

    def __init__(self, path):
        '''
        Load the 1D CSE model, given a path.

        The abundances are stored in a file '
        .out', 
        the physical parameters are stored in a file 'csphyspar_smooth.out'.
        The input of the 1D CSE model is stored in a file 'inputChemistry_*.txt'.

        From these paths, retrieve
            - the abundances            --> self.n
            - the physical parameters (for more info on the parameters, see the paper)
                - radius                --> self.radius
                - density               --> self.dens
                - temperature           --> self.temp
                - visual extinction     --> self.Av
                - radiation parameter   --> self.xi
            - the time steps            --> self.time
            - input --> self.Rstar, self.Tstar, self.Mdot, self.v, self.eps, self.rtol, self.atol
        '''

        self.path = os.path.join(path, '')
        self.name = os.path.basename(os.path.normpath(path))
        self.model = self.path
        self.inp_path = self.path + 'input.txt'

        ## retrieve input
        self.Rstar, self.Tstar, self.Mdot, self.v, self.eps, self.rtol, self.atol = self.read_input()

        ## retrieve abundances
        abs = self.read_data()
        self.n = abs

        ## retrieve physical parameters
        arr = np.loadtxt(self.path + 'csphyspar_smooth.out',
                         skiprows=4,
                         usecols=(0, 1, 2, 3, 4))
        self.radius, self.dens, self.temp, self.Av, self.xi = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        self.time = self.radius / (self.v)
    
    def read_data(self):
        '''
        Read data text file of output abundances of 1D CSE models.

        This data text file is build up in an inconvenient way,
        hence the data is read in this specific way.

        The data is stored in a numpy array.
        '''
        with open(self.path + 'csfrac_smooth.out', 'r') as file:
            part = []
            full = None
            for line in file:
                try:
                    if len(line) > 1:
                        part.append([float(el) for el in line.split()])
                except:
                    if len(part) != 0:
                        part = np.array(part)[:, 1:]
                        if full is None:
                            full = part
                        else:
                            full = np.concatenate((full, part), axis=1)
                    part = []
        return full
    
    def read_input(self):
        '''
        Read input text file of 1D CSE models, given the filename.
        '''
        with open(self.inp_path, 'r') as file:
            lines = file.readlines()
            lines = [item.rstrip() for item in lines]

        Rstar = float(lines[3][9:])
        Tstar = float(lines[4][9:])
        Mdot = float(lines[5][8:])  ## Msol/yr
        v = float(lines[6][11:])  ## sec
        eps = float(lines[8][19:])

        rtol = float(lines[31][7:])
        atol = float(lines[32][6:])

        return Rstar, Tstar, Mdot, v, eps, rtol, atol
    
    def get_abs_spec(self, spec):
        '''
        Return the abundance of a specific species of the 1D model.
        '''
        i = specs_dict[spec]
        abs = self.get_abs()
        if abs is not None:
            abs = abs.T[i]
        return abs

    def get_vel(self):
        '''
        Return the velocity of the 1D model.
        '''
        return self.v

    def get_input(self):
        print('-------------------')
        print('Input of test model')
        print('-------------------')
        print('Mdot [Msol/yr]:      ', self.Mdot)
        print('v [km/s]:            ', self.v / 1e5)
        print('Density proxi Mdot/v:', self.Mdot / self.v)
        print('')
        print('Temp at 1e16 cm [K]: ',
              np.round(utils.temp(self.Tstar, self.eps, 1e16), 2))
        print('Tstar:               ', self.Tstar)
        print('eps:                 ', self.eps)
        print('-------------------\n')

        return self.Mdot, self.v, self.Tstar, self.eps


##--------------------- Phantom+krome models -------------------- ###

class Phantommod(ChemModel):
    '''
    Class to load a phantom+krome wind model, calculated using Krome on particle paths from a phantom simulation.
    One output file contains physical parameters and abundances for one particle at all time steps.
    '''

    def __init__(self, path):
        '''
        Load the phantom model, given a path.

        The abundances and physical parameters are stored in a file 'particleID.chem'.

        From these paths, retrieve
            - the abundances            --> self.n
            - the physical parameters (for more info on the parameters, see the paper)
                - radius                --> self.radius
                - density               --> self.dens
                - temperature           --> self.temp
                - visual extinction     --> self.Av
                - radiation parameter   --> self.xi
            - the time steps            --> self.time
        '''

        ## retrieve abundances and physical parameters
        self.path = path
        self.name = os.path.basename(path)
        abs = self.read_data()
        if len(abs[:, 0]) < 10:
            print(f'Warning: {self.path} has less than 10 time steps.')
        self.time = abs[:, 0]
        self.radius = abs[:,1]
        self.dens, self.temp, self.mu, self.Av, self.xi = abs[:, 2], abs[:, 3], abs[:, 4], abs[:, 5], abs[:, 6]
        self.n = abs[:, 7:]


    def read_data(self):
        '''
        Read data text file of output abundances of phantom+krome models.
        The data is stored in a numpy array.
        '''
        data = []
        with open(self.path, 'r') as f:
            for line in f:
                if not line.startswith(' #'):
                    try:
                        data.append([float(_) for _ in line.strip().split()])
                    except ValueError:
                        print(f'Error: could not convert line to float: {line.strip()} in file {self.path}')
        return np.array(data)

class PhantomData(ChemData):
    '''
    Class to initialise train with data from phantom+Krome models split in 0D models.
    '''
    def find_paths(self, loc, meta):
        dirs = np.atleast_1d(np.loadtxt(loc + 'data/' + meta.paths, dtype=str))
        return np.array([os.path.join(d, f) for d in dirs for f in sorted(os.listdir(d))])
    
    def load_model(self, path):
        print('Loading phantom+krome model from:', path)
        return Phantommod(path)

## ---------------------- common functions ---------------------- ###

DATA_CLASSES = {'1DCSE': CSEdata,'Phantom': PhantomData}

def get_data_class(meta):
    if meta.data_type not in DATA_CLASSES:
        print(f'Error: data_type {meta.data_type} not recognised (Phantom or 1DCSE), defaulting to 1DCSE')
    return DATA_CLASSES.get(meta.data_type, CSEdata)

def get_data(meta, batch_size, kwargs):
    '''
    Prepare the data for training and validating the emulator.

    1. Make PyTorch dataset for the training and validation set.
    2. Make PyTorch dataloader for the 
        training 
            - batch size = batch_size
            - shuffle = True
        and validation set.
            - batch size = 1
            - shuffle = False 

    kwargs = {'num_workers': 1, 'pin_memory': True} for the DataLoader        
    '''
    data_class = get_data_class(meta)
    train = data_class(meta=meta,
                       train=True
                       )
    valid = data_class(meta=meta,
                       train=False
                       )

    print('Dataset:')
    print('------------------------------')
    print('  total # of samples:', len(train) + len(valid))
    print('#   training samples:', len(train))
    print('# validation samples:', len(valid))
    print('               ratio:',
          np.round(len(valid) / (len(train) + len(valid)), 2))
    print('     #  test samples:', len(train.testpath))

    data_loader = DataLoader(dataset=train,
                             batch_size=batch_size,
                             shuffle=True,
                             **kwargs)
    valid_loader = DataLoader(dataset=valid,
                             batch_size=1,
                             shuffle=False,
                             **kwargs)

    return train, valid, data_loader, valid_loader

def get_test_data(testpath,
                  meta,
                  train=False):
    '''
    Get the data of the test 1D model, given a path and meta-data from a training setup.

    Similar procedure as in the __getitem__() of the CSEdata class.

    The specifics of the 1D test model are stored in the 'name' dictionary.

    Input:
        - testpath [str]: path of the 1D test model
        - meta [dict]: meta data from the training setup
    '''
    data_class = get_data_class(meta)
    data = data_class(meta=meta,
                      train=train
                      )
    mod = data.load_model(testpath)
    name = {'path': testpath, 'name': mod.name}
    if isinstance(mod, CSEmod):
        name.update(Tstar=mod.Tstar, Mdot=mod.Mdot,v=mod.v,eps=mod.eps)

    return mod, data.preprocess(mod), name

def get_abs(n):
    '''
    Reverses the normalisation of the abundances.
    '''
    cutoff = 1e-20
    nmin = math.log10(cutoff)
    nmax = math.log10(0.85e-1)

    return 10**utils.unscale(n, nmin, nmax)


def get_phys(p_transf, dataset):
    '''
    Reverse the normalisation of the physical parameters.
    '''
    p = torch.empty_like(p_transf)
    for j in range(p_transf.shape[1]):
        p[:, j] = 10**utils.unscale(p_transf[:, j], dataset.mins[j],
                                    dataset.maxs[j])

    return p