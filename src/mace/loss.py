import torch
import torch.nn as nn
import warnings
import numpy as np
import os



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

    def forward(self, **input):
        '''
        For every loss term, computes weight * scale * value(input).
        Returns sum of the above, and raw loss terms.
        '''
        parts = {name:term(**input) for name,term in self.terms.items()}
        value = sum(term.weight * term.scale * parts[name] for name,term in self.terms.items()) ## type:ignore
        self._tot_sum += value.detach()
        self._count   += 1 # count calls within epoch
        return value, parts

    def endEpoch(self):
        '''
        Store the mean loss of this epoch in history and reset the accumulator.
        '''
        for term in self.terms.values():
            term.endEpoch() ## type:ignore
        self.tot_history.append(float(self._tot_sum) / self._count)
        self._tot_sum, self._count = 0., 0
                 
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

    def getHistory(self, **interval):
        '''
        Returns a dict with the selected history of all stored losses.
        '''
        history = self.getTermHistory(**interval)
        history['tot'] = self.getTotHistory(**interval)
        return history

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

    def save(self, path, **interval):
        '''
        Save the LossTerms to a eponymous .npy files in the given path.
        '''
        os.makedirs(path, exist_ok=True)
        for name, term in self.getHistory(**interval).items():
            if not term:
                warnings.warn(f'No history for loss term "{name}" yet.', RuntimeWarning)
            np.save(path+'/'+name+'.npy', term)