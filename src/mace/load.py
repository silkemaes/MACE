'''
This script loads a trained MACE model 
and provides the user with the possibility to apply (test) the model on a test dataset.

The class Trained_MACE() contains the loaded model, 
together with the training and test losses, and the meta data.

Version 2.0
'''

import src.mace.utils       as utils
import src.mace.loss        as loss
import src.mace.meta        as meta
import src.mace.mace        as mace
import matplotlib.pyplot    as plt
from pathlib import Path
import torch
import os



def load_MACE(dir, epoch=-1):
    '''
    Loads a MACE model from weights. Loads losses.

    - dir:      Directory where meta.json is stored
    - epoch:    Epoch to load. Defaults to last weights.

    Returns
    - model:    A MACE model with loaded weights
    - lossses:  A dict with Loss objects (train, valid, test)
    - metadata: meta.json
    '''
    metadata = meta.Meta(meta_path=dir)
    model, _ = load_model(dir, metadata, epoch)

    losses = {}

    for d_ in [d_ for d_ in Path(dir).iterdir() if d_.is_dir()]:
        d_ = os.path.basename(os.path.normpath(d_))
        if d_ in ('train', 'valid', 'test'):
            print(f'Found {d_}ing losses')
            losses[d_] = loss.load_loss(f'{dir}/{d_}')

    return model, losses, metadata

def load_model(dir, meta, epoch=-1):
    '''
    Initializes MACE model. Loads hyperparameters from meta.json.
    Overwrites MACE weights by those found in meta.json.

    - dir:      Directory where meta.json is stored
    - metadata: meta.json
    - epoch:    Epoch to load. Defaults to last weights.
    
    '''
    n_dim = meta.n_dim
    p_dim = meta.p_dim
    z_dim = meta.z_dim
    ae_type = meta.ae_type
    atol = meta.atol
    rtol = meta.rtol

    model = mace.MACE(n_dim, p_dim, z_dim, ae_type, atol=atol, rtol=rtol)
    overwrite_weights(model, dir, epoch)
    num_params = utils.count_parameters(model)
    
    return model, num_params

def overwrite_weights(model, dir, epoch=-1):
    '''
    Overwrites a MACE model's weights by those in dir/nn/nn{epoch}.pt
    '''
    if epoch >= 0:
        infile = f'nn/nn{epoch}.pt'
    else:
        infile = 'nn/nn.pt'
    state_dict = torch.load(f'{dir}/{infile}', map_location='cpu', weights_only=True)

    model.eval()
    model.load_state_dict(state_dict) 
    model.eval()