'''
This script contains classes for loading the 1D CSE models,
which is the training data for MACE,
as well as code to preprocess the data.


Contains:
    Classes:
        - CSEdata: Dataset class (PyTorch) to prepare the dataset for training and validating the emulator
        - CSEmod: Class to load a 1D CSE model
    
    Functions:
        - get_data: prepare the data for training and validating the emulator, using the PyTorch-specific dataloader
        - get_test_data: get the data of the test a 1D model, given a path and meta-data from a training setup
        - get_abs: get the abundances, given the normalised abundances
        - get_phys: reverse the normalisation of the physical parameters
        - read_input_1Dmodel: read input text file of 1D CSE models, given the filename
        - read_data_1Dmodel: read data text file of output abundances of 1D CSE models

NOTE:
This script only works for this specific dataset.

'''


import numpy            as np
import torch
from torch.utils.data   import Dataset, DataLoader
import src.mace.utils   as utils
from pathlib import Path

specs_dict, idx_specs = utils.get_specs()


### ----------------------- 1D CSE models ----------------------- ###



class CSEdata(Dataset):
    '''
    Class to initialise the dataset to train & test emulator.

    More specifically, this Dataset uses 1D CSE models, and splits them in 0D models.
    '''
    def __init__(self, nb_samples,dt_fract, nb_test, inpackage = False, train=True, datapath = 'train', fraction=0.7, cutoff = 1e-20):
        '''
        Initialising the attributes of the dataset.

        Input:
            - nb_samples [int]: number of 1D models to use for training & validation
            - dt_fract [float]: fraction of the timestep to use, depends on number of latent species 
                (see latent dynamics in paper)
            - nb_test [int]: number of 1D models to uses for testing
            - train [boolean]: True if training, False if testing
            - fraction [float]: fraction of the dataset to use for training, 1-fraction is used for validation,
                default = 0.7
            - cutoff [float]: cutoff value for the abundances, depends on tolerances of classical chemistry kinetics solver, 
                default = 1e-20
            - scale [str]: type of scaling to use, default = 'norm'

        Preprocess on data:
            - clip all abundances to cutoff
            - take np.log10 of abudances

        Structure:
            1. Load the paths of the 1D models
            2. Select a certain number of paths, given by nb_samples 
                --> self.path
            3. Select a random test path, that is not in the training set 
                --> self.testpath
            4. Load the rates matrix M (matrix needed for the 'element loss', see appendix of paper)
                (Currently not used) 
                --> self.M
            5. Set the min and max values of the physical parameters and abundances, 
                resulting from a search through the full dataset; see 'minmax.json' file.
                These values are used for normalisation of the data. 
                --> self.mins, self.maxs
            6. Set the cutoff value for the abundances 
                --> self.cutoff
            7. Set the fraction of the dataset to use for training 
                --> self.fraction
            8. Split the dataset in train and test set 
        '''
        print('> Train state:',train)

        # loc = '/STER/silkem/MACE/'
        loc = str(Path().cwd())+'/'
        paths = np.loadtxt(loc+'data/paths_train_data.txt', dtype=str)
        print('Found paths:', len(paths))

        ## select a certain number of paths, given by nb_samples
        np.random.seed(0)
        self.idxs = utils.generate_random_numbers(nb_samples, 0, len(paths))
        self.path = paths[self.idxs]
        

        ## select a random test path, that is not in the training set
        # self.test_idx = utils.generate_random_numbers(1, 0, len(paths))
        self.testpath = list()
        # self.testpath.append(paths[self.test_idx][0])
        self.nb_test = nb_test
        print('number of test paths:', nb_test)
        count = 0
        while count < nb_test:
            self.test_idx = utils.generate_random_numbers(1, 0, len(paths))
            if self.test_idx not in self.idxs:
                count += 1
                self.testpath.append(paths[self.test_idx][0])
            print('count:',count, '\r', end = '')
        print('Selected test paths:', len(self.testpath))

        # print('test path:', self.testpath)

        self.M = np.load(loc+'data/M_rate16.npy')       

        ## These values are the results from a search through the full dataset; see 'minmax.json' file
        self.logρ_min = np.log10(0.008223)
        self.logρ_max = np.log10(5009000000.0)
        self.logT_min = np.log10(10.)
        self.logT_max = np.log10(1851.0)   
        y = 1.e-100   ## this number is added to xi, since it contains zeros    
        self.logδ_min = np.log10(y)
        self.logδ_max = np.log10(y+0.9999)
        self.Av_min = np.log10(2.141e-05)
        self.Av_max = np.log10(1246.0)
        self.dt_max = 434800000000.0
        self.dt_fract = dt_fract
        self.n_min = np.log10(cutoff)
        self.n_max = np.log10(0.85e-1)    ## initial abundance He

        self.mins = np.array([self.logρ_min, self.logT_min, self.logδ_min, self.Av_min, self.n_min, self.dt_fract])
        self.maxs = np.array([self.logρ_max, self.logT_max, self.logδ_max, self.Av_max, self.n_max, self.dt_max])

        self.cutoff = cutoff
        self.fraction = fraction
        self.train = train
        self.inpackage = inpackage
        self.datapath = datapath

        ## Split in train and test set        
        N = int(self.fraction*len(self.path))
        if self.train:
            self.path = self.path[:N]
        else:
            self.path = self.path[N:]

        print('Selected paths:', len(self.path))
        print('\n')
            
            
    def __len__(self):
        '''
        Return the length of the dataset (number of 1D models used for training or validation).
        '''
        return len(self.path)

    def __getitem__(self, idx):
        '''
        Get the data of the idx-th 1D model.

        The CSEmod class is used to get the data of the 1D model.
        Subsequently, this data is preprocessed:
            - abundances (n) are
                - clipped to the cutoff value
                - np.log10 is taken 
                - normalised to [0,1]
            - physical parameters (p) are
                - np.log10 is taken
                - normalised to [0,1]
            - timesteps (Δt) are 
                - scaled to [0,1]
                - multiplied with dt_fract

        Returns the preprocessed data in torch tensors.
        '''

        mod = CSEmod(self.path[idx], inpackage = self.inpackage, data = self.datapath)

        Δt, n, p = mod.split_in_0D()

        ## physical parameters
        p_transf = np.empty_like(p)
        for j in range(p.shape[1]):
            # print(j)
            p_transf[:,j] = utils.normalise(np.log10(p[:,j]), self.mins[j], self.maxs[j])

        ## abundances
        n_transf = np.clip(n, self.cutoff, None)
        n_transf = np.log10(n_transf)
        n_transf = utils.normalise(n_transf, self.n_min, self.n_max)       ## max boundary = rel. abundance of He

        ## timesteps
        Δt_transf = Δt/self.dt_max * self.dt_fract             ## scale to [0,1] and multiply with dt_fract

        return torch.from_numpy(n_transf), torch.from_numpy(p_transf), torch.from_numpy(Δt_transf)
    

def get_data( nb_samples, dt_fract, nb_test,inpackage, batch_size, kwargs):
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
    ## Make PyTorch dataset
    train = CSEdata(nb_samples=nb_samples, dt_fract=dt_fract, nb_test = nb_test, inpackage = inpackage, train = True , datapath='train')
    valid = CSEdata(nb_samples=nb_samples, dt_fract=dt_fract, nb_test = nb_test, inpackage = inpackage, train = False, datapath='train')
    
    print('Dataset:')
    print('------------------------------')
    print('  total # of samples:',len(train)+len(valid))
    print('#   training samples:',len(train))
    print('# validation samples:',len(valid) )
    print('               ratio:',np.round(len(valid)/(len(train)+len(valid)),2))
    print('     #  test samples:',train.nb_test)

    data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True ,  **kwargs)
    test_loader = DataLoader(dataset=valid , batch_size=1 , shuffle=False,  **kwargs)

    return train, valid, data_loader, test_loader


def get_test_data(testpath, meta, inpackage = False, train = False, datapath = 'test'):
    '''
    Get the data of the test 1D model, given a path and meta-data from a training setup.

    Similar procedure as in the __getitem__() of the CSEdata class.

    The specifics of the 1D test model are stored in the 'name' dictionary.

    Input:
        - testpath [str]: path of the 1D test model
        - meta [dict]: meta data from the training setup
    '''
    
    data = CSEdata(nb_samples=meta['nb_samples'],dt_fract=meta['dt_fract'],nb_test= meta['nb_test'], train=train, fraction=0.7, cutoff = 1e-20, inpackage=inpackage)
    
    mod = CSEmod(testpath, inpackage, datapath)

    if inpackage:
        input = mod.get_input()

    Δt, n, p = mod.split_in_0D()

    name = {'path' : testpath[49:-57],
            'name' : mod.name,
            'Tstar' : mod.Tstar,
            'Mdot' : mod.Mdot,
            'v' : mod.v,
            'eps' : mod.eps}

    ## physical parameters
    p_transf = np.empty_like(p)
    for j in range(p.shape[1]):
        p_transf[:,j] = utils.normalise(np.log10(p[:,j]), data.mins[j], data.maxs[j])

    ## abundances
    n_transf = np.clip(n, data.cutoff, None)
    n_transf = np.log10(n_transf)
    n_transf = utils.normalise(n_transf, data.n_min, data.n_max)       ## max boundary = rel. abundance of He

    ## timesteps
    Δt_transf = Δt/data.dt_max * data.dt_fract             ## scale to [0,1] and multiply with dt_fract

    return mod, (torch.from_numpy(n_transf), torch.from_numpy(p_transf), torch.from_numpy(Δt_transf)), name


def get_abs(n):
    '''
    Get the abundances, given the normalised abundances.

    This function reverses the normalisation of the abundances.
    '''
    cutoff = 1e-20
    nmin = np.log10(cutoff)
    nmax = np.log10(0.85e-1)

    return 10**utils.unscale(n,nmin, nmax)

def get_phys(p_transf,dataset):
    '''
    Reverse the normalisation of the physical parameters.
    '''
    p = torch.empty_like(p_transf)
    for j in range(p_transf.shape[1]):
        p[:,j] = 10**utils.unscale(p_transf[:,j],dataset.mins[j], dataset.maxs[j])
    
    return p



class CSEmod():
    '''
    Class to load a 1D CSE model, calculated with the classical fortan code.
    For more info on this model, see https://github.com/MarieVdS/rate22_cse_code.
    '''
    def __init__(self, path, inpackage = False, data = 'test'):
        '''
        Load the 1D CSE model, given a path.

        The abundances are stored in a file 'csfrac_smooth.out', 
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

        if not inpackage:
            self.path = '/STER/silkem/CSEchem/' + path[34:-17]
            self.model = path[34:-51]
            self.name = path[-43:-18]
            inp_path = self.path[:-26]+ 'inputChemistry_'+self.name+'.txt'

        if inpackage:
            if data == 'test':
                parentpath = str(Path(__file__).parent)[:-15]
                print(parentpath)
                self.path = parentpath + 'data/test/' + path +'/'
                self.model = path[-62:-1]
                self.name = path
                inp_path = self.path+'input.txt'
            if data == 'train':
                parentpath = str(Path(__file__).parent)[:-15]
                self.path = parentpath + 'data/train/' + path[:-18] +'/'
                self.model = self.path
                self.name = self.path
                inp_path = self.path+'input.txt'
                

        abs_path = 'csfrac_smooth.out'
        phys_path = 'csphyspar_smooth.out'

        ## retrieve input
        self.Rstar, self.Tstar, self.Mdot, self.v, self.eps, self.rtol, self.atol = read_input_1Dmodel(inp_path)

        ## retrieve abundances
        abs = read_data_1Dmodel(self.path+abs_path)
        self.n = abs

        ## retrieve physical parameters
        arr = np.loadtxt(self.path+phys_path, skiprows=4, usecols=(0,1,2,3,4))
        self.radius, self.dens, self.temp, self.Av, self.xi = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        self.time = self.radius/(self.v) 
                

    def __len__(self):
        '''
        Return the length of the time array, which indicates the length of the 1D model.
        '''

        return len(self.time)

    def get_time(self):
        '''
        Return the time array of the 1D model.
        '''
        return self.time
    
    def get_phys(self):
        '''
        Return the physical parameters of the 1D model.
        '''
        return self.dens, self.temp, self.xi, self.Av
    
    def get_abs(self):
        '''
        Return the abundances of the 1D model.
        '''
        return self.n
    
    def get_abs_spec(self, spec):
        '''
        Return the abundance of a specific species of the 1D model.
        '''
        i = specs_dict[spec]
        abs = self.get_abs()
        if abs is not None:
            abs = abs.T[i]
        return abs

    def get_dens(self):
        '''
        Return the density of the 1D model.
        '''
        return self.dens 
    
    def get_temp(self):
        '''
        Return the temperature of the 1D model.
        '''
        return self.temp

    def get_Av(self):
        '''
        Return the visual extinction of the 1D model.
        '''
        return self.Av
    
    def get_xi(self):
        '''
        Return the radiation parameter of the 1D model.
        '''
        return self.xi

    def get_vel(self):
        '''
        Return the velocity of the 1D model.
        '''
        return self.v
    
    def get_path(self):
        '''
        Return the path of the 1D model.
        '''
        return self.path

    def get_name(self):
        '''
        Return the name of the 1D model.
        '''
        return self.name
    
    def get_dt(self):
        '''
        Return the time steps of the 1D model.
        '''
        return self.time[1:] - self.time[:-1]
    
    def split_in_0D(self):
        '''
        Split the 1D model in 0D models.
        '''
        Δt   = self.get_dt()
        n_0D = self.get_abs()
        y    = 1.e-100  ## this number is added to xi, since it contains zeros
        p    = np.array([self.get_dens()[:-1], self.get_temp()[:-1], self.get_xi()[:-1]+y, self.get_Av()[:-1]])

        return Δt.astype(np.float64), n_0D.astype(np.float64), p.T.astype(np.float64) # type: ignore
    
    def get_input(self):
        print('-------------------')
        print('Input of test model')
        print('-------------------')
        print('Mdot [Msol/yr]:      ', self.Mdot)
        print('v [km/s]:            ', self.v/1e5)
        print('Density proxi Mdot/v:', self.Mdot/self.v)
        print('')
        print('Temp at 1e16 cm [K]: ', np.round(utils.temp( self.Tstar, self.eps, 1e16),2))
        print('Tstar:               ', self.Tstar)
        print('eps:                 ', self.eps)
        print('-------------------\n')

        return self.Mdot, self.v, self.Tstar, self.eps
        


def read_input_1Dmodel(file_name):
    '''
    Read input text file of 1D CSE models, given the filename.
    '''
    with open(file_name, 'r') as file:
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

    This data text file is build up in an inconvenient way,
    hence the data is read in this specific way.

    The data is stored in a numpy array.
    '''
    with open(file_name, 'r') as file:
        part = []
        full = None
        for line in file:
            try:  
                if len(line) > 1: 
                    part.append([float(el) for el in line.split()])
            except:
                if len(part) != 0:
                    part = np.array(part)[:,1:]
                    if full is None:
                        full = part
                    else:
                        full = np.concatenate((full, part), axis = 1)
                part = []
    return full






