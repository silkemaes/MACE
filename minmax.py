'''
Browse training dataset and find min and max values for all input parameters.
Outputs a .txt file with the min and max values.
'''

import numpy as np
import pandas as pd

import os
import multiprocessing as mp
import time

DIRS = ['/path/to/dir']
SOURCE_DIR = os.path.dirname(os.path.npath(__file__))
outputfile = os.path.join(SOURCE_DIR,'data', 'minmax.txt')

files = []

# headers to skip
skip = {'radius(cm)', 'mu'}
# headers to keep (ignoring abundances and time, they'll be treated separately)
keep = {'n(cm-3)','T(K)', 'A_UV', 'xi'}

# abundance cutoff
abundance_cutoff = 1e-20

def find_min_max(file, skip=skip, keep=keep, abundance_cutoff=abundance_cutoff):
    data = pd.read_csv(file, sep=r'\s+\s+', engine='python')
    n = []
    mins = {}
    maxs = {}
    for key in data.keys():
        if key in skip:
            data = data.drop(columns=key)
        elif key in keep:
                mins[key] = (data[key].min())
                maxs[key] = (data[key].max())
        elif key == '# time(s)':
            timestep = [b-a for a,b in zip(data[key][:-1], data[key][1:])]
            mins['time(s)'] = (min(timestep))
            maxs['time(s)'] = (max(timestep))
        else:
            # abundances, concatenate all values before querying min and max
            n += list(data[key].values)
    mins['n'] = (max(np.min(n), abundance_cutoff))
    maxs['n'] = (np.max(n))
    return mins, maxs

if __name__ == '__main__':

    print('- Searching for .chem files in:')
    for DIR in DIRS:
        print(f'        - {DIR}')
        files += [os.path.join(DIR,f) for f in sorted(os.listdir(DIR)) if f.endswith('.chem')]
        total = len(files)
        print(f'            -> Found {total} files')

    mins = {}
    maxs = {}   
    for key in keep:
        mins[key] = []
        maxs[key] = []
    mins['n'] = []
    maxs['n'] = []
    mins['time(s)'] = []
    maxs['time(s)'] = []

    # get number of cpus available
    num_cpus = mp.cpu_count()//2
    print(f'- Using {num_cpus} CPUs.')
    processed = 0
    outputs = []
    print('- Computing min and max values for all parameters...')
    with mp.Pool(processes=num_cpus) as pool:
        for result in pool.imap_unordered(find_min_max, files, chunksize=1):
            processed += 1
            outputs.append(result)
            print(f"Processed {processed}/{total} files", end="\r", flush=True)
    
    for output in outputs:
        for key in output[0].keys():
            mins[key].append(output[0][key])
            maxs[key].append(output[1][key])
    print('- Finding global min and max values for all parameters...')
    for key in mins.keys():
        mins[key] = min(mins[key])
        maxs[key] = max(maxs[key])

    with open(outputfile, 'w') as f:
        f.write(f'rho_min: {mins["n(cm-3)"]}\n')
        f.write(f'rho_max: {maxs["n(cm-3)"]}\n')
        f.write(f'T_min: {mins["T(K)"]}\n')
        f.write(f'T_max: {maxs["T(K)"]}\n')
        f.write(f'delta_min: {mins["xi"]}\n')
        f.write(f'delta_max: {maxs["xi"]}\n')
        f.write(f'A_UV_min: {mins["A_UV"]}\n')
        f.write(f'A_UV_max: {maxs["A_UV"]}\n')
        f.write(f'n_min: {mins["n"]}\n')
        f.write(f'n_max: {maxs["n"]}\n')
        f.write(f'dt_max: {maxs["time(s)"]}\n')
