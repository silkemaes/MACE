'''
This script contains classes for loading chemical models used to train MACE,
as well as code to preprocess the data.

Contains:
    Classes:
        - ChemData: Dataset class for chemical data to prepare the dataset for training and validating the emulator
        - ChemModel: Class to load a model
        - PhantomData + PhantomMod: Dataset and model class for Phantom models
        - CseData + CseMod: Dataset and model class for 1D CSE models
    
    Functions:
        - get_data: prepare the data for training and validating the emulator, using the PyTorch-specific dataloader
        - get_test_data: get the data of the test a 1D model, given a path and meta-data from a training setup
        - get_n: reverse normalisation of abundances
        - get_phys: reverse the normalisation of the physical parameters

        NOTE: for any new type of datasets, define a new subclass of ChemData and ChemModel to account 
        for how it should be processed

Version 2.1

TODO: Why are we preprocessing on iterations???
TODO: Convert to binary storage
TODO: Shuffle buffer for faster loading and mixing
'''
# Navigation
import json
import os
from pathlib import Path
import warnings
import shutil

# Math
import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
import src.mace.utils as utils
from src.mace.meta import Meta

# Typechecker
type N_P_DT = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

specs_dict = utils.get_specs()

### Base class for models and datasets

class ChemModel:
    ''' 
    Basic class to load chemical model: time, abundances, and physical parameters (rho, T, Xi and AV)
    For a new type of model, define a new subclass with its own loading method.
    '''
    time: np.ndarray
    dens: np.ndarray
    temp: np.ndarray
    xi: np.ndarray
    Av: np.ndarray
    n: np.ndarray
    path: str
    name: str
    def __init__(self, path:str|Path): raise NotImplementedError("Subclass ChemModel")
    def __len__(self):     return len(self.time)
    def get_time(self):    return self.time
    def get_phys(self):    return self.dens, self.temp, self.xi, self.Av
    def get_n(self):       return self.n
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
        n_0D = self.get_n()
        y = 1.e-100  ## this number is added to xi and Av since it can contain zeros
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
    def __init__(self, meta, train=True, cutoff=1e-20, verbose=True):
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
        paths = self.find_paths(loc, meta)
        print('Paths found:', len(paths))
        self._split_paths(paths, meta)
        self.set_minmax(meta, loc+'data/'+ meta.minmax)

        self.verbose = verbose
        

    def _split_paths(self, paths, meta):
        '''
        Split paths into train/valid/test randomly
        '''
        nb_samples = meta.nb_train + meta.nb_valid + meta.nb_test
        if nb_samples > len(paths):
            raise ValueError(f'Not enough models in {meta.paths} for training, validation and testing.')
        np.random.seed(meta.seed)
        perm = np.random.permutation(len(paths))
        train_idx = perm[:meta.nb_train]
        valid_idx = perm[meta.nb_train:meta.nb_train + meta.nb_valid]
        test_idx  = perm[meta.nb_train + meta.nb_valid:nb_samples]
        if self.train:
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
            mm = json.load(f)
        y = 1.e-100  ## xi and Av can contain zeros
        self.dt_max, self.dt_fract = mm['dt_max'], meta.dt_fract
        self.n_min = np.log10(self.cutoff)
        self.n_max = np.log10(0.85e-1)  ## initial abundance He
        self.mins = np.array([np.log10(mm['dens_min']), np.log10(mm['temp_min']),
                            np.log10(y + mm['delta_min']), np.log10(y + mm['Av_min']),
                            self.n_min, self.dt_fract])
        self.maxs = np.array([np.log10(mm['dens_max']), np.log10(mm['temp_max']),
                            np.log10(y + mm['delta_max']), np.log10(y + mm['Av_max']),
                            self.n_max, self.dt_max])
    
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
        return self.preprocess(self.load_model(self.path[idx])) # TODO: This needs to go. Like. Wow.

    def load_model(self, path): raise NotImplementedError('load_model not implemented in Data subclass')

    def find_paths(self, loc, meta): raise NotImplementedError('find_paths not implemented in Data subclass')

### ----------------------- 1D CSE models ----------------------- ###

class CseData(ChemData):
    '''
    1D CSE models
    '''
    def find_paths(self, loc, meta):
        # atleast_1d to ensure path is an array (loadtxt returns an array scalar (0D array) if only one path) 
        return np.atleast_1d(np.loadtxt(loc + 'data/' + meta.paths, dtype=str))
    
    def load_model(self, path):
        if self.verbose: print('Loading 1D CSE model from', path)
        return CseMod(path)

class CseMod(ChemModel):
    '''
    Class to load a 1D CSE model, calculated with the classical fortan code.
    For more info on this model, see https://github.com/MarieVdS/rate22_cse_code.
    '''

    def __init__(self, path):
        '''
        Load the 1D CSE model, given a path.

        The abundances are stored in a file 'csfrac_smooth.out', 
        the physical parameters are stored in a file 'csphyspar_smooth.out'.
        The input of the 1D CSE model is stored in a file 'input.txt'.

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
        self.inp_path = self.path + 'input.txt'

        ## retrieve input
        self.Rstar, self.Tstar, self.Mdot, self.v, self.eps, self.rtol, self.atol = self.read_input()

        ## retrieve abundances
        self.n = self.read_data()

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
    
    def get_n_spec(self, spec):
        '''
        Return the abundance of a specific species of the 1D model.
        '''
        i = specs_dict[spec]
        n = self.get_n()
        if n is not None:
            n = n.T[i]
        return n

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

class PhantomMod(ChemModel):
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
        data = self.read_data()
        if len(data[:, 0]) < 10:
            print(f'Warning: {self.path} has less than 10 time steps.')
        self.time = data[:, 0]
        self.radius = data[:,1]
        self.dens, self.temp, self.mu, self.Av, self.xi = data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
        self.n = data[:, 7:]


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
        # atleast_1d to ensure path is an array (loadtxt returns an array scalar (0D array) if only one path) 
        dirs = np.atleast_1d(np.loadtxt(loc + 'data/' + meta.paths, dtype=str))
        return np.array([os.path.join(d, f) for d in dirs for f in sorted(os.listdir(d))])
    
    def load_model(self, path):
        if self.verbose: print('Loading phantom+krome model from:', path)
        return PhantomMod(path)

## ---------------------- common functions ---------------------- ###

DATA_CLASSES:dict[str,type[ChemData]] = {'1DCSE': CseData,'Phantom': PhantomData}
MODEL_CLASSES:dict[str,type[ChemModel]] = {'1DCSE': CseMod,'Phantom': PhantomMod}

def get_type_class[T](type_: str, classes: dict[str, type[T]]) -> type[T]:
    if type_ not in classes:
        backup = next(iter(classes))
        warnings.warn(f'Error: data_type {type_} not in defaults ({classes.keys()}), defaulting to {backup}', RuntimeWarning)
        type_ = backup
    return classes[type_]

def get_data(meta:Meta, batch_size:int, kwargs:dict, verbose:bool=True
             ) -> tuple[ChemData, ChemData, DataLoader[N_P_DT], DataLoader[N_P_DT]]:
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
    data_class = get_type_class(meta.data_type, DATA_CLASSES)
    train = data_class(meta=meta,
                       train=True,
                       verbose=verbose
                       )
    valid = data_class(meta=meta,
                       train=False,
                       verbose=verbose
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

def get_test_data(testpath, meta:Meta
                  )-> tuple[ChemModel, N_P_DT, dict[str, str | float]]:
    '''
    Get the data to test a 1D model, given a path to the test data and metadata of the model.
    The specifics of the test model are stored in the 'name' dictionary.

    Input:
        - testpath [str]: path of the 1D test model
        - meta [Meta]: metadata of the model
    '''
    data_class = get_type_class(meta.data_type, DATA_CLASSES)
    data = data_class(meta=meta,
                      train=False
                      )
    mod = data.load_model(testpath)
    name = {'path': testpath, 'name': mod.name}
    if isinstance(mod, CseMod):
        name.update(Tstar=mod.Tstar, Mdot=mod.Mdot,v=mod.v,eps=mod.eps)

    return mod, data.preprocess(mod), name

def get_n(n):
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

def save_as_binary(model:ChemModel, target:Path):
    dt,n,p = model.split_in_0D()
    target.mkdir(exist_ok=True)
    np.save(target/'n.npy', n)
    np.save(target/'p.npy', p)
    np.save(target/'dt.npy', dt)

def make_binary_cache(path:str|Path, data_type:str, 
                      force:bool=False, verbose:bool=True) -> Path:
    '''
    Goes over every model within path/ and converts it to a binary file. Stores the binary files in path/../../cache/.
    
    If force is false, tries to find old cached models and reuses them.
    
    - path:         Path to model set. Example: MACE/data/train/ caches to MACE/cache/train/
    - data_type:    '1DCSE' or 'Phantom'
    - force:        Force full cache reconstruction
    - verbose:      Complain. A lot.

    Returns:
    - target:       Cache path
    '''

    # Final destination
    path=Path(path)
    target = path.parent.parent/"cache"/path.name
    target.parent.mkdir(parents=True, exist_ok=True)

    # Check existing cache
    existing = set()
    if target.is_dir() and not force:
        existing = {m.name for m in target.iterdir() if m.is_dir()}
        if verbose: print(f'Binary cache already exists at {target}, force=True for clean override.')

    # Staging storage
    tmp = target.parent/f'{target.name}.tmp'
    if tmp.is_dir():
        warnings.warn(f'Previous broken cache found at {tmp}, cleaning up.', RuntimeWarning)
        shutil.rmtree(tmp)
    tmp.mkdir()

    # Iterate over models
    chemModel = get_type_class(data_type, MODEL_CLASSES)
    for d in [d_ for d_ in path.iterdir() if d_.is_dir()]:
        name = d.name

        # Copy existing
        if (not force) and name in existing:
            shutil.copytree(target/name, tmp/name)
            if verbose: print(f"Finished copying {name}")
            continue

        # Make new cached
        if verbose: print(f"Converting {name} into {target}", end='\r')
        try:
            # Convert and store
            model = chemModel(d)
            save_as_binary(model, tmp/name)
            if verbose: print(f"Finished converting {name}")
        except Exception as err:
            # Delete partial conversion and log
            shutil.rmtree(tmp/name, ignore_errors=True)
            if verbose: print(f"Failed converting {name} with exit code {err}")

    # Remove target
    if target.is_dir():
        if verbose: print(f'Cleaning up {target}')
        shutil.rmtree(target)
        if verbose: print('Finished cleanup.')

    # Rename target.tmp to target
    tmp.rename(target)

    return target