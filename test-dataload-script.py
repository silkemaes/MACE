import sys

## import own functions
sys.path.insert(1, '/lhome/silkem/MACE/MACE/src/mace')

import dataset      as ds



dir_name = 'Output_nov2022'

paths_O, paths_C = ds.retrieve_file(dir_name)

print(len(paths_O))
print(paths_O[4])

print('\ntest')

proper = ds.read_data(paths_O[4])

print(proper)




