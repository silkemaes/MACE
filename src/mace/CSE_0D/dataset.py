
import numpy             as np
import torch

from torch.utils.data    import Dataset, DataLoader

## own scripts
import utils

specs_dict, idx_specs = utils.get_specs()


### ----------------------- 1D CSE models ----------------------- ###



class CSEdata(Dataset):
    '''
    Class to initialise the dataset to train & test emulator

    Get data from textfiles (output CSE model)
    
    Preprocess:
        - set all abundances < cutoff to cutoff
        - take np.log10 of abudances

    '''
    def __init__(self, nb_samples,dt_fract, train=True, fraction=0.7, cutoff = 1e-20, scale = 'norm'):
        loc = '/STER/silkem/MACE/'
        paths = np.loadtxt(loc+'data/paths_data_C.txt', dtype=str)

        ## select a certain number of paths, given by nb_samples
        np.random.seed(0)
        self.idxs = utils.generate_random_numbers(nb_samples, 0, len(paths))
        self.path = paths[self.idxs]
        self.test_idx = utils.generate_random_numbers(1, 0, len(paths))
        self.testpath = paths[self.test_idx]
        while self.test_idx in self.idxs:
            self.testpath = paths[self.test_idx]
            self.test_idx = utils.generate_random_numbers(1, 0, len(paths))
        


        ## These values are the results from a search through the full dataset; see 'minmax.json' file
        self.logρ_min = np.log10(0.008223)
        self.logρ_max = np.log10(5009000000.0)
        self.logT_min = np.log10(10.)
        self.logT_max = np.log10(1851.0)   
        y = 1.e-100   ## this number is added to delta, since it contains zeros    
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


        ## Split in train and test set        
        N = int(self.fraction*len(self.path))
        if self.train:
            self.path = self.path[:N]
        else:
            self.path = self.path[N:]
            
            
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        mod = CSEmod(self.path[idx])

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


    def get_test(self):
        print(self.testpath)
        mod = CSEmod(self.testpath[0])

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





def get_data( nb_samples, dt_fract, batch_size, kwargs):
    ## Make PyTorch dataset
    train = CSEdata( nb_samples=nb_samples, dt_fract=dt_fract)
    test  = CSEdata( nb_samples=nb_samples, dt_fract=dt_fract, train = False)
    
    print('Dataset:')
    print('------------------------------')
    print('total # of samples:',len(train)+len(test))
    print('# training samples:',len(train))
    print('#  testing samples:',len(test) )
    print('             ratio:',np.round(len(test)/(len(train)+len(test)),2))

    data_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True ,  **kwargs)
    test_loader = DataLoader(dataset=test , batch_size=1 , shuffle=False,  **kwargs)

    return train, test, data_loader, test_loader



class CSEmod():
    '''
    Class to initialise the dataset to train & test emulator

    Get data from textfiles (output CSE model)
    
    Preprocess:
        - set all abundances < cutoff to cutoff
        - take np.log10 of abudances

    '''
    def __init__(self, path):

        self.path = '/STER/silkem/CSEchem/' + path[34:-17]
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

        # print(self.delta.shape, self.temp.shape)
        # for i in range(len(self.delta)):
        #     # if self.temp[i] == 0.:
        #     #     print('yes')
        #     #     self.temp[i] = 10
        #     if self.delta[i] == 0.:
        #         self.delta[i] = 1.e-40
                

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
        n_0D = self.n
        y=1.e-100
        p = np.array([self.get_dens()[:-1], self.get_temp()[:-1], self.get_delta()[:-1]+y, self.get_Av()[:-1]])
        # print(self.get_delta()[:-1]+y)
        # print(self.delta)

        return Δt.astype(np.float64), n_0D.astype(np.float64), p.T.astype(np.float64)
        


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
        if loc[-3:-1] == 'ep':
            path_mods = utils.get_files_in(path+loc+'/models/')
            for mod in path_mods:
                if mod[-1] != 't':
                    if loc[13] == 'O':
                        all_paths_O.append(path+loc+'/models/'+mod+'/csfrac_smooth.out')
                    if loc[13] == 'C':
                        all_paths_C.append(path+loc+'/models/'+mod+'/csfrac_smooth.out')
    
    return all_paths_O, all_paths_C


