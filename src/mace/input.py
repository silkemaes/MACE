import json


dt_fracts = {4 : 0.296, 5: 0.269,8: 0.221,10: 0.175,12: 0.146,16: 0.117,20: 0.09,25: 0.078,32: 0.062,48: 0.043,64: 0.033,128: 0.017}

class Input():
    def __init__(self, infile, name):

        self.name = name
        self.file = infile 

        with open(infile, 'a') as file:
            file.write('\nName = '+name+'\n')

        with open(infile,'r') as f:
            file = f.readlines()
            lines = []
            for line in file:
                lines.append(line.split())

        inputfile = {}
        for i in range(len(lines)):
            if not len(lines[i]) == 0 and len(lines[i]) > 2:
                inputfile[lines[i][0]] = lines[i][2]
            elif not len(lines[i]) == 0 and len(lines[i]) <= 2:
                print('You forgot to give an input for '+lines[i][0])

        ## SET PARAMETERS
        self.lr          = float(inputfile['lr'])
        self.nb_epochs   = int(inputfile['nb_epochs'])
        self.ini_epochs  = int(inputfile['ini_epochs'])
        self.scheme      = inputfile['scheme']
        self.losstype    = inputfile['losstype']
        self.z_dim       = int(inputfile['z_dim'])
        self.dt_fract    = dt_fracts[self.z_dim]
        self.batch_size  = 1
        self.nb_samples  = int(inputfile['nb_samples'])
        self.nb_evol     = int(inputfile['nb_evol'])
        self.n_dim       = 468
        self.nb_hidden   = int(inputfile['nb_hidden'])
        self.ae_type     = str(inputfile['ae_type'])
        self.nb_test     = int(inputfile['nb_test'])  

        ## fractions for losses
        self.abs = float(inputfile['abs'])
        self.grd = float(inputfile['grd'])
        self.idn = float(inputfile['idn'])
        self.elm = float(inputfile['elm'])

    def print(self):
        print('------------------------------')
        print('Name:', self.name)
        print('------------------------------')
        print('     inputfile:', self.file)
        print('      # hidden:', self.nb_hidden)
        print('       ae type:', self.ae_type)
        print('# z dimensions:', self.z_dim)
        print('        scheme:', self.scheme)
        print('  # evolutions:', self.nb_evol)
        print('     loss type:', self.losstype)
        print('      # epochs:', self.nb_epochs)
        print(' learning rate:', self.lr)
        print('')

        return
        
    def make_meta(self, path):

        metadata = {'nb_samples': self.nb_samples,
                    'lr'        : self.lr,
                    'epochs'    : self.nb_epochs,
                    'z_dim'     : self.z_dim,
                    'dt_fract'  : self.dt_fract,
                    'losstype'  : self.losstype,
                    'inputfile' : self.file,
                    'scheme'    : self.scheme,
                    'nb_evol'   : self.nb_evol,
                    'nb_hidden' : self.nb_hidden,
                    'ae_type'   : self.ae_type,
                    'nb_test'   : self.nb_test,
                    'done'      : 'false',
                }

        json_object = json.dumps(metadata, indent=4)
        with open(path+"/meta.json", "w") as outfile:
            outfile.write(json_object)

        return metadata

    def get_facts(self):
        fract = {'abs' : self.abs, 
                 'grd' : self.grd, 
                 'idn' : self.idn, 
                 'elm' : self.elm}
        
        return fract


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
            'nb_hidden' : self.nb_hidden,
            'ae_type'   : self.ae_type,
            'nb_test'   : self.nb_test,
            'done'      : 'true'
        }

        json_object = json.dumps(metadata, indent=4)
        with open(path+"/meta.json", "w") as outfile:
            outfile.write(json_object)

        return

    


