'''
This script loads a trained MACE model 
and provides the user with the possibility to apply (test) the model on a test dataset.

The class Trained_MACE() contains the loaded model, 
together with the training and test losses, and the meta data.
'''



import src.mace.utils       as utils
import src.mace.loss        as loss
import matplotlib.pyplot    as plt
from pathlib import Path
import os



def load_MACE(dir, epoch=-1):
    '''
    Loads a MACE model from weights. Loads losses.

    - dir:      Directory where meta.json is stored
    - epoch:    Epoch to load. Defaults to last weights.

    Returns
    - model:    A MACE model with loaded weights
    - lossses:  A dict with Loss objects (train, valid, test)
    - meta:     meta.json
    '''
    meta = utils.load_meta(dir)
    model = utils.load_model(dir, meta, epoch)

    losses = {}

    for d_ in [d_ for d_ in Path(dir).iterdir() if d_.is_dir()]:
        d_ = os.path.basename(os.path.normpath(d_))
        if d_ in ('train', 'valid', 'test'):
            print(f'Found {d_}ing losses')
            losses[d_] = loss.load_loss(f'{dir}/{d_}')

    return model, losses, meta