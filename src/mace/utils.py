import os
import numpy as np

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