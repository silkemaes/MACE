import torch
import torch.nn as nn
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from matplotlib.lines import Line2D



##################
### Loss Types ###
##################

class LossTerm(nn.Module):
    '''
    General contributor to overall loss. Tracks weight, scale, and history.
    '''
    def __init__(self, weight=1.):
        '''
        A loss contributor with predefined weight and scale.
        '''
        super().__init__()
        # Multiplicators
        self.weight = weight # user-defined weight
        self.scale = 1. # scale normalisation
        # Loss history
        self.history = [] # average loss per epoch
        self._sum    = 0. # loss sum over the current epoch
        self._count  = 0 # amount of calls in current epoch

    # Computing
    def compute(self, **input):
        '''
        Throws NotImplementedError. Override with loss based on **input.
        '''
        raise NotImplementedError('Unspecified Loss type, please subclass')

    def forward(self, **input):
        '''
        Passes on self.compute.
        Saves loss value to history.
        Does NOT apply weight and scale!
        '''
        value = self.compute(**input).mean() # combine all tensor output
        self._sum   += value.detach() # don't save autograd
        self._count += 1 # count calls within epoch
        return value          

    # History
    def setScale(self, scale):
        '''
        Set the normalisation scale of the LossTerm.
        '''
        self.scale = scale

    def endEpoch(self):
        '''
        Store the mean loss of this epoch in history and reset the accumulator.
        '''
        if self._count == 0:
            raise RuntimeError('endEpoch() called prematurely.')
        self.history.append(float(self._sum) / self._count)
        self._sum, self._count = 0., 0


class AbsLoss(LossTerm):
    '''
    'Absolute' difference between real and predicted
    (n - ň)²
    '''
    def compute(self, n_e, n_e_hat, **_):
        '''
        Arguments:
        - n_e:      Real evolution
        - n_e_hat:  Predicted evolution
        '''
        return (n_e - n_e_hat)**2


class GrdLoss(LossTerm):
    '''
    Gradient difference between real and predicted
    (∂n - ∂ň)²
    '''
    def compute(self, n_e, n_e_hat, **_):
        '''
        Arguments:
        - n_e:      Real evolution
        - n_e_hat:  Predicted evolution
        '''
        return (torch.gradient(n_e)[0] - torch.gradient(n_e_hat[0])[0])**2 # TODO fix ts


class IdnLoss(LossTerm):
    '''
    Identity difference on dataset
    (n0 - D( E( n0 )))²
    '''
    def compute(self, n, p, model, **_):
        '''
        Arguments:
        - n:        Data abundance sample
        - p:        Data physical sample
        - model:    MACE model (must have autoencoder)
        '''
        E = model.encoder  # TODO make solver return z to skip this
        D = model.decoder
        x = torch.cat((p, n), dim=-1) # TODO switch up for Gnn models
        return (n - D( E( x )))**2 


class MseLoss(LossTerm):
    '''
    Mean squared error between real and predicted # TODO check against AbsLoss
    MSE( ň, n)
    '''
    def compute(self, n_e, n_e_hat, **_):
        '''Arguments:
        - n_e:      Real evolution
        - n_e_hat:  Predicted evolution
        '''
        return nn.functional.mse_loss(n_e_hat, n_e)



##################
### Loss Group ###
##################

class Loss(nn.Module):
    '''
    A collection of LossTerms. LossTerms are stored in a dict.
    Tracks all previous losses.
    '''
    def __init__(self, terms:dict):
        '''
        Arguments:
        - terms: dict with named LossTerms.'''
        super().__init__()
        # Loss terms
        self.terms = nn.ModuleDict(terms)
        # Loss history
        self.tot_history = [] # average loss per epoch
        self._tot_sum = 0. # running loss sum over the current epoch
        self._count = 0 # amount of calls in current epoch

    # Computing
    def forward(self, **input):
        '''
        For every loss term, computes weight * scale * value(input).
        Returns sum of the above, and raw loss terms.
        '''
        parts = {name:term(**input) for name,term in self.terms.items()}
        value = sum(term.weight * term.scale * parts[name] for name,term in self.terms.items()) ##type:ignore
        self._tot_sum += value.detach() ##type:ignore
        self._count   += 1 # count calls within epoch
        return value, parts

    # History
    def endEpoch(self):
        '''
        Store the mean loss of this epoch in history and reset the accumulator.
        '''
        for term in self.terms.values():
            term.endEpoch() ## type:ignore
        self.tot_history.append(float(self._tot_sum) / self._count)
        self._tot_sum, self._count = 0., 0

    def getHistory(self, **interval):
        '''
        Returns a dict with the selected history of all stored losses.
        '''
        totHist = {'tot': self.getTotHistory(**interval)}
        termHist = self.getTermHistory(**interval)
        return totHist|termHist

    def getTermHistory(self, start=0, stop=None):
        '''
        Returns a dict with the selected history of raw LossTerms.
        '''
        return {name:term.history[start:stop] for name,term in self.terms.items()} ## type:ignore

    def getTotHistory(self, start=0, stop=None):
        '''
        Returns a list with the selected history of the total loss.
        '''
        return self.tot_history[start:stop]

    def normalize(self, N=5):
        '''
        Normalize the LossTerm scales by the average over the last N epochs.
        Afterward, scale * raw_loss = mean(recent_loss)
        '''
        mean_losses = {}
        recent_losses = self.getTermHistory(start=-N)
        for name in self.terms.keys():
            recent_loss = recent_losses[name]
            if not recent_loss:
                raise RuntimeError(f'No history for loss term "{name}" yet.')
            mean_losses[name] = max(abs(sum(recent_loss) / len(recent_loss)), 1e-8) ## stability control

        mean_loss = sum(mean_losses.values())/len(mean_losses)
        for name, term in self.terms.items():
            term.setScale(mean_loss / mean_losses[name]) ## type:ignore

    # Storage
    def save(self, path, **interval):
        '''
        Save the history, weight and scale of LossTerms
        to eponymous files in the given path.
        '''
        os.makedirs(path, exist_ok=True)
        self.saveHistory(path, **interval)
        self.saveWeight(path)
        self.saveScale(path)

    def saveHistory(self,path,**interval):
        for name, hist in self.getHistory(**interval).items():
            if not hist:
                warnings.warn(f'No history for loss term "{name}" yet.', RuntimeWarning)
            np.save(f'{path}/{name}.npy', hist)

    def saveWeight(self,path):
        with open(f'{path}/weight.json', 'w') as f:
            json.dump({name:float(term.weight) for name,term in self.terms.items()}, f, indent=4) ## type:ignore

    def saveScale(self,path):
        with open(f'{path}/scale.json', 'w') as f:
            json.dump({name:float(term.scale) for name,term in self.terms.items()}, f, indent=4) ## type:ignore

    def load(self, path):
        '''
        Override the history, weight and scale of LossTerms
        by those in the given path.
        '''
        if not os.path.isdir(path): raise FileNotFoundError(f"Invalid path at {path}")
        self.loadHistory(path)
        self.loadWeight(path)
        self.loadScale(path)

    def loadHistory(self,path):
        for name in self.getHistory().keys():
            if not os.path.exists(f'{path}/{name}.npy'):
                raise FileNotFoundError(f"Missing history for '{name}' term in {path}.")
            hist = np.load(f'{path}/{name}.npy').tolist()
            if name=='tot': self.tot_history = hist
            else:   self.terms[name].history = hist
        print(f"Loaded losses for {self.getHistory().keys()}")

    def loadWeight(self,path):
        if not os.path.exists(f'{path}/weight.json'):
            raise FileNotFoundError(f"Missing weight.json in {path}. Load old models using Loss.loadHistory().")
        with open(f'{path}/weight.json') as f:
            weights = json.load(f)
        for name, term in self.terms.items():
            term.weight = weights[name]

    def loadScale(self,path):
        if not os.path.exists(f'{path}/scale.json'):
            raise FileNotFoundError(f"Missing scale.json in {path}. Load old models using Loss.loadHistory().")
        with open(f'{path}/scale.json') as f:
            scales = json.load(f)
        for name, term in self.terms.items():
            term.setScale(scales[name]) ## type:ignore



##################
### Loss Setup ###
##################

def setup_loss(types:str, weights=None):
    '''
    Sets up a loss with terms_like_this and weights.

    Accepted losses: abs, grd, idn
    '''
    terms = {}
    if weights is None:
        weights=[1 for t in types.split('_')]
    if len(types.split('_')) != len(weights):
        raise RuntimeError(f"{len(weights)} weights for {len(types.split('_'))} lossterms.")
    
    for i, t in enumerate(types.split('_')):
        if   t=="abs": terms[t] = AbsLoss(weights[i])
        elif t=='grd': terms[t] = GrdLoss(weights[i])
        elif t=='idn': terms[t] = IdnLoss(weights[i])
        else: raise RuntimeError(f"Unrecognized loss type: {t}")
    return Loss(terms)

def load_loss(path, types=None):
    '''
    Load a Loss saved with Loss.save().
    If types is not given, the loss terms are read from weight.json.
    '''
    if not types:
        with open(f'{path}/weight.json') as f:
            types = '_'.join(json.load(f)) ## keys in saved order
    loss_fn = setup_loss(types)
    loss_fn.load(path)
    return loss_fn

def load_old_loss(path, types='abs_idn', **weights):
    '''
    Load a depracated loss history into a new Loss object.
    Old loss combined raw, weight, and scale, 
    so they are all standard set to 1 here. 
    Not ideal for training further.
    You can add weights in **kwargs.
    '''
    loss_fn = setup_loss(types, **weights)
    loss_fn.loadHistory(path)
    return loss_fn



##################
### Loss Plots ###
##################

COLORS = {'tot': 'k', 
          'abs': 'cornflowerblue', 
          'grd': 'gold', 
          'idn': 'salmon', 
          'elm': 'darkorchid'}
STYLES = {'train': dict(ls='-',  marker='.'),
          'valid': dict(ls='--', marker='x'),
          'test':  dict(ls=':',  marker='^')}

def plot(losses:dict, ax=None, log=True, ms=3, **interval):
    '''
    Plot per-epoch losses. Colour = loss term, line style = dataset.

    losses: {dataset: Loss}, e.g. {'train': trainloss, 'valid': validloss}
    
    Claude is 100% responsible for this one, I hold no accountability
    '''
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    histories = {dataset: loss.getHistory(**interval) for dataset, loss in losses.items()}
    for dataset, history in histories.items():
        for term, values in history.items():
            ax.plot(values, color=COLORS.get(term, 'grey'), ms=ms, **STYLES[dataset])

    terms    = dict.fromkeys(t for h in histories.values() for t in h)   ## unique, in order
    handles  = [Line2D([], [], color='grey', label=d, **STYLES[d]) for d in histories]
    handles += [Line2D([], [], color=COLORS.get(t, 'grey'), lw=4, label=t) for t in terms]
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    if log: ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(True, ls='--', lw=0.2)
    return ax