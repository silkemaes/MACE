import sys

## import own functions
sys.path.insert(1, '/lhome/silkem/MACE/MACE/src/mace')

import dataset      as ds



dir_name = 'Output_nov2022'

paths_O, paths_C = ds.retrieve_file(dir_name)


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



