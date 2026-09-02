'''
Prepare a MACE model for inference with FTorch.
Load a MACE model and save it to TorchScript format.
Save the ODE solver weights to text files for Fortran reading.
Copy everything useful for the model to a state folder in both
the model folder and the run directory (if desired).
'''

import os
from pathlib import Path

import numpy as np
import torch

import src.mace.load as load
import src.mace.utils as utils

model_name = '20260602_071233'
epoch=10

# If you want to copy the model in a different directory afterwards (e.g. where it will be run)
# leave this as None to skip this step
final_dir = '/STER/hydroModels/camille/phantom/macetraining/phantom_mace/'

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SOURCE_DIR, 'model/')


print('   - Loading model located at:', os.path.join(MODEL_PATH, model_name))

# Load trained model
trained = load.Trained_MACE(outloc=MODEL_PATH, dirname=model_name, epoch=epoch)
meta = trained.get_meta()
model = trained.model

save_path = os.path.join(MODEL_PATH, model_name, 'state')
os.makedirs(save_path, exist_ok=True)


# Get the weights of the ODE solver
weights = model.jit_solver.state_dict()

# save ODE weights to be readable in fortran
print('   - Saving ODE weights to text file...')
with open(os.path.join(save_path, f'{epoch}_ODE_params.txt'), 'w') as f:
    f.write(f"atol   {weights['_orig_mod.step_size_controller.atol']} \n")
    f.write(f"rtol   {weights['_orig_mod.step_size_controller.rtol']}")

with open(os.path.join(save_path, f'{epoch}_ODE_C.txt'), 'w') as f:
    for val in weights['_orig_mod.step_method.term.f.C']:
        f.write(f'{val}\n')
with open(os.path.join(save_path, f'{epoch}_ODE_A.txt'), 'w') as f:
    for val in weights['_orig_mod.step_method.term.f.A'].flatten():
        f.write(f'{val}\n')
with open(os.path.join(save_path, f'{epoch}_ODE_B.txt'), 'w') as f:
    for val in weights['_orig_mod.step_method.term.f.B'].flatten():
        f.write(f'{val}\n')

# Save scripted autoencoder
print('   - Saving autoencoder with TorchScript...')
scripted_encoder = torch.jit.script(model.encoder)
scripted_decoder = torch.jit.script(model.decoder)
scripted_encoder.save(os.path.join(save_path, f'{epoch}_encoder.pt'))
scripted_decoder.save(os.path.join(save_path, f'{epoch}_decoder.pt'))

# Save relevant quantities from meta to state folder (json are annoying in Fortran so 
# we move everything to text files)
print('   - Saving meta file to state folder...')
with open(os.path.join(save_path, 'meta.txt'), 'w') as f:
    for key in meta:
        f.write(f'{key}: {meta[key]}\n')
    f.write(f'epoch: {epoch}\n')

# copy the minmax file too
os.system(f'cp {os.path.join(SOURCE_DIR, "data", "minmax.txt")} {save_path}')

# read Krome species file and store it in simpler format for Fortran
print('   - Saving species file to state folder...')
krome_file = utils.get_specs('Phantom')
with open(os.path.join(save_path, 'abundance_label.txt'), 'w') as f:
    for spec in krome_file.keys():
        if krome_file[spec]+1 <= 468:  
            # only keep species with atomic number <= 468 (remove dummy species used by krome)
            f.write(f'{spec} {krome_file[spec]+1}\n')

# If desired, copy the state folder to final_dir
if final_dir is not None:
    print('   - Copying model to final directory: ', final_dir)
    if final_dir.startswith('~'):
        final_dir = os.path.expanduser(final_dir)
    final_dir = os.path.join(final_dir, model_name)
    os.makedirs(final_dir, exist_ok=True)
    os.system(f'cp -r {save_path}/* {final_dir}/')