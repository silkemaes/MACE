import os
import numpy as np
import sys

'''
Makes the output directory - if nessecary.
Returns the path of that dir.
'''
def makeOutputDir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def get_absolute_residuals(real, pred):

    nb_specs = np.shape(pred)[1]

    for i in range(nb_specs):  ## loop over all specs
        res = np.abs((np.array(real[:,i])-np.array(pred[:,i])))/float(max(np.array(real[:,i])))
    res = res/nb_specs

    return res

def get_files_in(path):
    files = os.listdir(path) 
    locs = []
    for file in files:
        locs.append(file)
    
    return locs


def unscale(x, min, max):

    unscaled = x*np.abs(max-min)+min

    return unscaled


sys.path.append('/STER/silkem/ChemTorch/src')
import rates as rate

def get_specs():
    specs, parnt, convs = rate.read_specs_file('C', 16)
    specs_dict = dict()
    idx_specs  = dict()
    for i in range(len(specs)):
        specs_dict[specs[i]] = i
        idx_specs[i] = specs[i]

    return specs_dict, idx_specs


def normalise(x,min,max):
        # print("Normalising")
        norm = (x - min)*(1/np.abs( min - max ))
        # print(x, norm)
        return norm

def generate_random_numbers(n, start, end):
    return np.random.randint(start, end, size=n)