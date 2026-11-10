import json

dt_fracts = {4 : 0.296, 5: 0.269,8: 0.221,10: 0.175,12: 0.146,16: 0.117,20: 0.09,25: 0.078,32: 0.062,48: 0.043,64: 0.033,128: 0.017}

class Meta():
    '''
    Load .in files to metadata. Store metadata to .json files.
    
    Required input variables:
    - self.n_dim, self.p_dim, self.z_dim
    - lr, nb_evol, nb_train, nb_valid
    - losstype

    Optional input variables:
    - ae_type, atol, rtol, g_nn
    - ini_epochs, nb_epochs, data_type
    '''
    def __init__(self, infile, name=None, complain=False):

        self.file = infile
        self.complain = complain

        if name is not None:
            with open(infile, 'a') as file:
                file.write('\nName = '+name+'\n')
            self.name = name
        else: self.name = 'Unnamed'

        with open(infile,'r') as f:
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
            elif not len(lines[i]) == 0 and len(lines[i]) <= 2:
                print('You forgot to give an input for '+lines[i][0])

        ## MODEL INPUT
        self.n_dim      = int(self._req('n_dim'))
        self.p_dim      = int(self._req('p_dim'))
        self.z_dim      = int(self._req('z_dim'))
        self.ae_type    = str(self._opt('ae_type', 'simple'))
        self.atol       = float(self._opt('atol', 1e-5))
        self.rtol       = float(self._opt('rtol', 1e-2))
        self.g_nn       = self._opt('g_nn', 'False') in ('True','true','1')

        ## TRAINING INPUT
        self.lr          = float(self._req('lr'))
        self.nb_evol     = int(self._req('nb_evol'))
        self.nb_train    = int(self._req('nb_train'))
        self.nb_valid    = int(self._req('nb_valid'))
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

    def print(self):
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
        print(f'Train|Validate: {self.nb_train}|{self.nb_valid}')
        print('      Datatype:', self.data_type)
        print('     Loss Type:', self.losstype)
        print(f'  Loss Weights: {self.lossweights}')
        print('')

        return
        
    def make_meta(self, path):
        metadata = {'inputfile' : self.file,
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
                    'ini_epochs': self.ini_epochs,
                    'nb_epochs' : self.nb_epochs,
                    'data_type' : self.data_type,
                    'dt_fract'  : self.dt_fract,
                    'losstype'  : self.losstype,
                    'lossweights':self.lossweights
                    }

        json_object = json.dumps(metadata, indent=4)
        with open(path+"/meta.json", "w") as outfile:
            outfile.write(json_object)

        return metadata


'''
    def update_meta(self, traindata, train_time, overhead_time, path):

        metadata = {'nb_samples'  : self.nb_samples,
            'lr'        : self.lr,
            'epochs'    : self.nb_epochs,
            'z_dim'     : self.z_dim,
            'dt_fract'  : self.dt_fract,
            'tmax'      : traindata.dt_max,
            'train_time_h': train_time/(60*60),
            'overhead_s'  : overhead_time,
            'samples'   : len(traindata),
            'cutoff_abs': traindata.cutoff,
            'losstype'  : self.losstype,
            'inputfile' : self.file,
            'scheme'    : self.scheme,
            'nb_evol'   : self.nb_evol,
            'ae_type'   : self.ae_type,
            'nb_test'   : self.nb_test,
            'done'      : 'true'
        }

        json_object = json.dumps(metadata, indent=4)
        with open(path+"/meta.json", "w") as outfile:
            outfile.write(json_object)

        return

    '''


