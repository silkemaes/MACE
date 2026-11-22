'''
Initialize losses using setup_loss(type, weights)
Pass loss types as abs_grd_idn_pidn or dicts.

Creating a new loss term:
- Subclass LossTerm
- Add args to Loss.forward() docs
- Add term name to setup_loss()
- Add term color to COLOR

Version 2.1
'''
import torch
import torch.nn as nn
from torch import Tensor
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import os
import json
from matplotlib.lines import Line2D
from src.mace.mace import MACE
import pathlib



##################
### Loss Types ###
##################

class LossTerm(nn.Module):
    '''
    General contributor to overall loss. Tracks weight, scale, and history.
    '''
    def __init__(self, weight:float=1.):
        '''
        A loss contributor with predefined weight and scale.
        '''
        super().__init__()
        # Multiplicators
        self.weight = weight # user-defined weight
        self.scale = 1. # scale normalisation
        # Loss history
        self.history:list[float] = [] # average loss per epoch
        self._sum = 0. # loss sum over the current epoch
        self._count = 0 # amount of calls in current epoch

    # Computing
    def compute(self, **input) -> Tensor:
        '''
        Throws NotImplementedError. Override with loss based on **input.
        '''
        raise NotImplementedError('Unspecified Loss type, please subclass')

    def forward(self, **input) -> Tensor:
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
    def setScale(self, scale:float):
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
    def compute(self, n_e:Tensor, n_e_hat:Tensor, **_
                ) -> Tensor:
        '''
        Arguments:
        - n_e:     Real evolution
        - n_e_hat: Predicted evolution
        '''
        return (n_e - n_e_hat)**2


class GrdLoss(LossTerm):
    '''
    Gradient difference between real and predicted
    (∂n - ∂ň)²
    '''
    def compute(self, n_e:Tensor, n_e_hat:Tensor, **_
                ) -> Tensor:
        '''
        Arguments:
        - n_e:     Real evolution
        - n_e_hat: Predicted evolution
        '''
        return (torch.gradient(n_e,dim=1)[0] - torch.gradient(n_e_hat,dim=1)[0])**2 # TODO fix ts


class IdnLoss(LossTerm):
    '''
    Identity difference on dataset
    (n - D( E( n,p )))²
    '''
    def compute(self, n:Tensor, p:Tensor, model:MACE, ae:tuple[Tensor, Tensor|None]|None=None, **_
                ) -> Tensor:
        '''
        Arguments:
        - n:       Data abundance sample
        - p:       Data physical sample
        - model:   MACE model (must have autoencoder)
        - ae:      Precomputed model.autoencode(n, p)
        '''
        n_hat, _ = ae if ae is not None else model.autoencode(n,p)
        return (n - n_hat)**2


class PhysIdnLoss(LossTerm):
    '''
    Identity difference on physical state
    (p - D( E( n,p )))²
    '''
    def compute(self, n:Tensor, p:Tensor, model:MACE, ae:tuple[Tensor, Tensor|None]|None=None, **_
                ) -> Tensor:
        '''
        Arguments:
        - n:       Data abundance sample
        - p:       Data physical sample
        - model:   MACE model (must have autoencoder)
        - ae:      Optional precomputed model.autoencode(n, p)
        '''
        _, p_hat = ae if ae is not None else model.autoencode(n, p)
        if p_hat is None:
            raise RuntimeError("'pidn' loss needs a model with p_out=True")
        return (p - p_hat)**2


class MseLoss(LossTerm):
    '''
    Mean squared error between real and predicted # TODO check against AbsLoss
    MSE( ň, n)
    '''
    def compute(self, n_e:Tensor, n_e_hat:Tensor, **_
                ):
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

    Loss.forward() with the following arguments:
    - n_e:     Real evolution
    - n_e_hat: Predicted evolution
    - n:       Data abundance sample
    - p:       Data physical sample
    - model:   MACE model
    - ae:      Precomputed model.autoencode(n, p) (idn, pidn)
    '''
    def __init__(self, terms:dict[str,LossTerm]):
        '''
        Arguments:
        - terms: dict with named LossTerms.'''
        super().__init__()
        # Loss terms
        self.terms = terms
        # Loss history
        self.tot_history:list[float] = [] # average loss per epoch
        self._tot_sum = 0. # running loss sum over the current epoch
        self._count = 0 # amount of calls in current epoch

    # Computing
    def forward(self, **input) -> tuple[float, dict[str, Tensor]]:
        '''
        For every loss term, computes weight * scale * value(input).
        Returns sum of the above, and raw loss terms.
        '''
        parts = {name:term(**input) for name,term in self.terms.items()}
        value = sum(term.weight * term.scale * parts[name] for name,term in self.terms.items())
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

    def getHistory(self, start:int=0, stop:int|None=None) -> dict[str, list[float]]:
        '''
        Returns a dict with the selected history of all stored losses.
        '''
        totHist = {'tot': self.getTotHistory(start, stop)}
        termHist = self.getTermHistory(start, stop)
        return totHist|termHist

    def getTermHistory(self, start:int=0, stop:int|None=None) -> dict[str, list[float]]:
        '''
        Returns a dict with the selected history of raw LossTerms.
        '''
        return {name:term.history[start:stop] for name,term in self.terms.items()} ## type:ignore

    def getTotHistory(self, start:int=0, stop:int|None=None) -> list[float]:
        '''
        Returns a list with the selected history of the total loss.
        '''
        return self.tot_history[start:stop]

    def normalize(self, N:int=5):
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
    def save(self, path:str|pathlib.Path, start:int=0, stop:int|None=None):
        '''
        Save the history, weight and scale of LossTerms
        to eponymous files in the given path.
        '''
        os.makedirs(path, exist_ok=True)
        self.saveHistory(path, start, stop)
        self.saveWeight(path)
        self.saveScale(path)

    def saveHistory(self,path:str|pathlib.Path, start:int=0, stop:int|None=None):
        '''
        Save raw histories and total history to path
        '''
        for name, hist in self.getHistory(start, stop).items():
            if not hist:
                warnings.warn(f'No history for loss term "{name}" yet.', RuntimeWarning)
            np.save(f'{path}/{name}.npy', hist)

    def saveWeight(self,path:str|pathlib.Path):
        '''
        @Claude
        '''
        with open(f'{path}/weight.json', 'w') as f:
            json.dump({name:float(term.weight) for name,term in self.terms.items()}, f, indent=4) ## type:ignore

    def saveScale(self,path:str|pathlib.Path):
        '''
        @Claude
        '''
        with open(f'{path}/scale.json', 'w') as f:
            json.dump({name:float(term.scale) for name,term in self.terms.items()}, f, indent=4) ## type:ignore

    def load(self, path:str|pathlib.Path):
        '''
        Override the history, weight and scale of LossTerms
        by those in the given path.
        '''
        if not os.path.isdir(path): raise FileNotFoundError(f"Invalid path at {path}")
        self.loadHistory(path)
        self.loadWeight(path)
        self.loadScale(path)

    def loadHistory(self,path:str|pathlib.Path):
        '''
        Load raw histories and total history from path
        '''
        for name in self.getHistory().keys():
            if not os.path.exists(f'{path}/{name}.npy'):
                raise FileNotFoundError(f"Missing history for '{name}' term in {path}.")
            hist = np.load(f'{path}/{name}.npy').tolist()
            if name=='tot': self.tot_history = hist
            else:   self.terms[name].history = hist
        print(f"Loaded losses for {self.getHistory().keys()}")

    def loadWeight(self,path:str|pathlib.Path):
        '''
        @Claude
        '''
        if not os.path.exists(f'{path}/weight.json'):
            raise FileNotFoundError(f"Missing weight.json in {path}. Load old models using Loss.loadHistory().")
        with open(f'{path}/weight.json') as f:
            weights = json.load(f)
        for name, term in self.terms.items():
            term.weight = weights[name]

    def loadScale(self,path:str|pathlib.Path):
        '''
        @Claude
        '''
        if not os.path.exists(f'{path}/scale.json'):
            raise FileNotFoundError(f"Missing scale.json in {path}. Load old models using Loss.loadHistory().")
        with open(f'{path}/scale.json') as f:
            scales = json.load(f)
        for name, term in self.terms.items():
            term.setScale(scales[name]) ## type:ignore


##################
### Loss Setup ###
##################

def setup_loss(losstypes:dict[str,float]|str={'abs':1,'idn':1}, weights:list[float]|None=None) -> Loss:
    '''
    Sets up a loss with {types: weights} or types_like_this.

    Accepted losses: abs, grd, idn
    '''
    terms:dict[str,LossTerm] = {}
    if isinstance(losstypes, str):
        types = losstypes.split('_')
        if weights is not None and len(types) != len(weights):
            raise ValueError(f"Amount of losstypes ({len(types)}) does not match amount of weights ({len(weights)})")
        losstypes = {t: (1 if weights is None else weights[i]) for i, t in enumerate(types)}
    
    for t, weight in losstypes.items():
        if   t=="abs": terms[t] = AbsLoss(weight)
        elif t=='grd': terms[t] = GrdLoss(weight)
        elif t=='idn': terms[t] = IdnLoss(weight)
        elif t=='pidn': terms[t] = PhysIdnLoss(weight)
        else: raise RuntimeError(f"Unrecognized loss type: {t}")
    return Loss(terms)

def load_loss(path, types:str|None=None, weights:list[float]|None=None) -> Loss:
    '''
    Load a Loss saved with Loss.save().
    If types is not given, the loss terms are read from weight.json.

    @Claude
    '''
    if not types:
        with open(f'{path}/weight.json') as f:
            types = '_'.join(json.load(f)) ## keys in saved order
    loss_fn = setup_loss(types, weights)
    loss_fn.load(path)
    return loss_fn

def load_old_loss(path, types:str='abs_idn', weights:list[float]|None=None) -> Loss:
    '''
    Load a depracated loss history into a new Loss object.
    Old loss combined raw, weight, and scale, 
    so they are all standard set to 1 here. 
    Not ideal for training further.

    @Claude
    '''
    loss_fn = setup_loss(types, weights)
    loss_fn.loadHistory(path)
    return loss_fn



##################
### Loss Plots ###
##################

COLORS = {'tot': 'k', 
          'abs': 'cornflowerblue', 
          'grd': 'gold', 
          'idn': 'salmon', 
          'pidn':'maroon'}
STYLES = {'train': dict(ls='-',  marker='.'),
          'valid': dict(ls='--', marker='x'),
          'test':  dict(ls=':',  marker='^')}

def plot(losses:dict, ax:Axes|None=None, log:bool=True, ms:float=3, start:int=0, stop:int|None=None) -> Axes:
    '''
    Plot per-epoch losses. Colour = loss term, line style = dataset.

    losses: {dataset: Loss}, e.g. {'train': trainloss, 'valid': validloss}
    
    @Claude
    '''
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))

    histories = {dataset: loss.getHistory(start, stop) for dataset, loss in losses.items()}
    for dataset, history in histories.items():
        for term, values in history.items():
            ax.plot(values, color=COLORS.get(term, 'grey'), ms=ms, **STYLES[dataset]) #type:ignore

    terms    = dict.fromkeys(t for h in histories.values() for t in h)   ## unique, in order
    handles  = [Line2D([], [], color='grey', label=d, **STYLES[d]) for d in histories] #type:ignore
    handles += [Line2D([], [], color=COLORS.get(t, 'grey'), lw=4, label=t) for t in terms]
    ax.legend(handles=handles, loc='upper right', fontsize=10)

    if log: ax.set_yscale('log')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(True, ls='--', lw=0.2)
    return ax