'''
Metadata holder for MACE models.
Top-level object.

TODO: think really hard about a better way to do this
that doesn't bork user experience, ts spaghetti

Adding meta parameter: 
- add in .setMeta()
- add in .makeMeta() OR .updateMeta()
- add in .print() [optional]
Version 2.0
'''
import json
import os

dt_fracts = {4 : 0.296, 5: 0.269,8: 0.221,10: 0.175,12: 0.146,16: 0.117,20: 0.09,25: 0.078,32: 0.062,48: 0.043,64: 0.033,128: 0.017}

class Meta():
    '''
    Loads metadata from .in or meta.json. Stores metadata to .json files.

    If meta_path is given: search meta_path/meta.json

    If infile is given: search infile
    
    - meta_path:    Path to meta_path/meta.json
    - infile:       Path of infile
    - name:         Name for current model (defaults 'Unnamed')
    - complain:     Give warnings for missing input and edge cases

    Required input variables:
    - self.n_dim, self.p_dim, self.z_dim
    - lr, nb_evol, nb_train, nb_valid, nb_test, seed
    - losstype

    Optional input variables:
    - ae_type, atol, rtol, g_nn
    - ini_epochs, nb_epochs, data_type
    '''
    # Setup
    def __init__(self, infile=None, meta_path=None, 
                 name=None, complain=True):
        self.complain = complain
        self.name = name

        if None not in (meta_path, infile):
            raise ValueError('Too many arguments: choose meta_path or infile.')
        if meta_path is not None:
            self.loadJson(meta_path)
        elif infile is not None:
            self.loadInfile(infile, name)
        else:
            raise ValueError('Needs a meta_path or infile_in.')

    def loadInfile(self, path, name):
        '''
        Setup '''
        self.file = path

        if name is not None:
            with open(self.file, 'a') as file:
                file.write('\nName = '+name+'\n')
            self.name = name
        else: self.name = 'Unnamed'

        with open(self.file,'r') as f:
            file = f.readlines()
            lines = []
            for line in file:
                lines.append(line.split())

        self.inputfile = {}
        for i in range(len(lines)):
            if '##' in lines[i]:
                pass
            elif not len(lines[i]) == 0 and len(lines[i]) > 2:
                self.inputfile[lines[i][0]] = lines[i][2]
            elif not len(lines[i]) == 0 and len(lines[i]) <= 2 and self.complain:
                print('You forgot to give an input for '+lines[i][0])

        self.setMeta()

    def loadJson(self, path):
        '''
        Load an existing meta.json into self.meta.

        - path:         Path to meta.json
        '''
        with open(os.path.join(path, 'meta.json')) as f:
            self.meta = json.load(f)

        if self.name is None or self.name=='Unnamed': # Overwrite name if empty
            self.name = self.meta.get('name')
        self.meta['name'] = self.name
        self.file = self.meta.get('inputfile')

        self.path = path
        self.inputfile = self.meta | self.meta.get('lossweights', {})
        self.setMeta()

    def setMeta(self):
        '''
        Sets all metadata based on the contents of self.inputfile.
        '''
        ## MODEL INPUT
        self.n_dim      = int(self._req('n_dim'))
        self.p_dim      = int(self._req('p_dim'))
        self.z_dim      = int(self._req('z_dim'))
        self.ae_type    = str(self._opt('ae_type', 'simple'))
        self.atol       = float(self._opt('atol', 1e-5))
        self.rtol       = float(self._opt('rtol', 1e-2))
        self.g_nn       = str(self._opt('g_nn', False)).lower() in ('true','1')

        ## TRAINING INPUT
        self.lr          = float(self._req('lr'))
        self.nb_evol     = int(self._req('nb_evol'))
        self.nb_train    = int(self._req('nb_train'))
        self.nb_valid    = int(self._opt('nb_valid', 3))
        self.nb_test     = int(self._req('nb_test'))
        self.seed        = int(self._opt('seed', 0))
        self.ini_epochs  = int(self._opt('ini_epochs', 5))
        self.nb_epochs   = int(self._opt('nb_epochs', 150))
        self.data_type   = str(self._opt('data_type', '1DCSE'))
        self.dt_fract    = dt_fracts[self.z_dim]

        ## LOSS INPUT
        self.losstype    = str(self._req('losstype'))
        self.lossweights = {}
        for ltype in self.losstype.split('_'):
            lossweight   = float(self._opt(ltype, 1, True))
            self.lossweights[ltype] = lossweight

    def _req(self, key):
        '''
        Find required key in input, error if missing.
        '''
        if key in self.inputfile.keys():
            return self.inputfile[key]
        raise KeyError(f"Missing value for required key {key}.")

    def _opt(self, key, default, complain=False):
        '''
        Find optional key in input, go default and complain if missing.'''
        if key in self.inputfile.keys():
            return self.inputfile[key]
        if self.complain or complain:
            print(f"Missing value for optional key {key}.")
        return default

    # Operations
    def print(self):
        '''
        You're never gonna guess what this does.
        '''
        print('------------------------------')
        print('Name:', self.name)
        print('------------------------------')
        print('     inputfile:', self.file)
        print(f'    #n, #p, #z: {self.n_dim} {self.p_dim} {self.z_dim}')
        print('   Autoencoder:', self.ae_type)
        print('      Neural G:', self.g_nn)
        print(' Learning Rate:', self.lr)
        print('       #Epochs:', self.nb_epochs)
        print('   #Evolutions:', self.nb_evol)
        print(f'Train|Validate|Test: {self.nb_train}|{self.nb_valid}|{self.nb_test}')
        print('      Datatype:', self.data_type)
        print('     Loss Type:', self.losstype)
        print(f'  Loss Weights: {self.lossweights}')
        print('')
        
    def makeMeta(self, path):
        '''
        Makes a new meta.json.
        Saves initial metadata to path/meta.json.
        Overwrites existing meta.json.
        '''
        self.path = path
        self.meta ={'inputfile' : self.file,
                    'name'      : self.name,
                    'n_dim'     : self.n_dim,
                    'p_dim'     : self.p_dim,
                    'z_dim'     : self.z_dim,
                    'ae_type'   : self.ae_type,
                    'atol'      : self.atol,
                    'rtol'      : self.rtol,
                    'g_nn'      : self.g_nn,
                    'lr'        : self.lr,
                    'nb_evol'   : self.nb_evol,
                    'nb_train'  : self.nb_train,
                    'nb_valid'  : self.nb_valid,
                    'nb_test'   : self.nb_test,
                    'seed'      : self.seed,
                    'ini_epochs': self.ini_epochs,
                    'nb_epochs' : self.nb_epochs,
                    'data_type' : self.data_type,
                    'dt_fract'  : self.dt_fract,
                    'losstype'  : self.losstype,
                    'lossweights':self.lossweights}

        json_object = json.dumps(self.meta, indent=4)
        with open(os.path.join(path, 'meta.json'), "w") as outfile:
            outfile.write(json_object)
        
    def updateMeta(self, path=None, **kwargs):
        '''
        Update metadata with any with passed kwargs.
        
        - path:         Write to this path. Uses existing meta.json if None.

        Updates meta.py with any of the following attributes:
        - exit_code:    float
        - test_errors:  dict{str,float}
        - test_times:   dict{str,float}        
        If no meta.json exists, runs .makeMeta
        '''
        # Handle empty paths and nonexistent meta.json
        if not hasattr(self, 'path'): # Checks if meta.json exists
            if path is None: # No existing or given path to search
                raise RuntimeError('No path found to write metadata to.')
            if os.path.exists(os.path.join(path, 'meta.json')): # Check for existing metadata
                self.loadJson(path)
            else: # No meta.json exists yet
                if self.complain:         # Notify of new path
                    print(f'No existing meta.json found, creating at {path}/meta.json')
                self.makeMeta(path)
        if path is None: path = self.path 
        
        possible = ['exit_code','test_errors','test_times']
        for key in kwargs:
            if key not in possible: # Check if valid metadata
                raise AttributeError(f'Unrecognized meta attribute: {key}')
        self.meta.update(kwargs)

        json_object = json.dumps(self.meta, indent=4)
        with open(os.path.join(path, 'meta.json'), "w") as outfile:
            outfile.write(json_object)
