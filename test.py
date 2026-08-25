'''
Load and test a trained MACE model.
'''

import matplotlib           as mpl

from matplotlib          import rcParams, rc
rcParams.update({'figure.dpi': 200})
mpl.rcParams.update({'font.size': 10})
rc('font', family='serif')
rc('text', usetex=True)

from pathlib import Path
import sys
import os

parentpath = str(Path().cwd())+'/'
print(parentpath)

sys.path.append(parentpath)

import src.mace.test            as test
import src.mace.load            as load


dt_fracts = {4 : 0.296, 5: 0.269,8: 0.221,10: 0.175,12: 0.146,16: 0.117,20: 0.09,25: 0.078,32: 0.062,48: 0.043,64: 0.033,128: 0.017}

# Load the model
outloc  = parentpath+'model/'
dirname = '20260630_111730'    

trained = load.Trained_MACE(outloc=outloc, dirname=dirname, epoch=14)
meta = trained.get_meta()
model = trained.model

# Plot the evolution of the loss functions

lossfig = trained.plot_loss(save=True)

# Testing the model
testpath = f'{os.environ["MACE_TRAINING_DIR"]}/3d/v17-5_5e7_a20/chem_trace/40001.chem'

macetime = test.test_model('Phantom',model,testpath, meta, plotting=True, save = True, inpackage = True)


