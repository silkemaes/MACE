import sys

## import own functions
sys.path.insert(1, '/lhome/silkem/MACE/MACE/src/mace')

import utils

def retrieve_file(dir_name):
    '''
    Function to retrieve the paths of the all 1D CSE models.

    Originally these models are not stored in a convenient way,
    hence with this function, the paths are collected in a list, 
    separate for C-rich and O-rich models.
    '''
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

dir_name = 'Output_nov2022'

paths_O, paths_C = retrieve_file(dir_name)


dir = '/lhome/silkem/MACE/MACE/data/'

title = dir+'paths_data_C.txt'
with open (title,'w') as f:
    for path in paths_C:
        f.write(path+'\n')

title = dir+'paths_data_O.txt'
with open (title,'w') as f:
    for path in paths_O:
        f.write(path+'\n')

print('DONE!')



