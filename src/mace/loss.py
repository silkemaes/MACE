'''
This script handles the losses of the training.

Contains:
    - Loss: class to store the losses of the training
    - LoadedLoss: class to store the losses of a trained model
    - functions for calculating the different losses
    - function to plot the losses
'''



import torch
import numpy                    as np
import src.mace.utils           as utils
from torch.autograd.functional  import jacobian
import os

import matplotlib.pyplot    as plt
import matplotlib.lines     as mlines
from pathlib import Path

class Loss():
    def __init__(self, norm, fract, losstype):
        '''
        Initialise the loss object, which will contain the losses for each epoch. 

        norm:       dict with normalisation factors for each loss
        fract:      dict with factors to multiply each loss with

        Different types of losses:
            - 'abs':    absolute
            - 'grd':    gradient
            - 'idn':    identity = losses due to autoencoder
            - 'elm':    element conservation 
        '''
        self.norm  = norm
        self.fract = fract

        self.tot = list()

        self.abs = list()
        self.grd = list()
        self.idn = list()
        self.elm = list()

        parentpath = str(Path(__file__).parent)[:-15]

        self.M = np.load(parentpath+'M_rate16.npy')

        ## initialise
        self.set_losstype(losstype)
        

    def set_losstype(self, losstype):
        '''
        Set the type of loss used.
            losstype:   string with the type of loss used
                - 'abs':                            absolute loss
                - 'grd':                            gradient loss
                - 'idn':                            identity loss
                - 'elm':                            element  loss
                - 'abs_grd' or 'grd_abs':           abs + grd
                - 'abs_idn' or 'idn_abs':           abs + idn
                - 'grd_idn' or 'idn_grd':           grd + idn
                - 'abs_elm' or 'elm_abs':           abs + elm
                - 'grd_elm' or 'elm_grd':           grd + elm
                - 'idn_elm' or 'elm_idn':           idn + elm
                - 'abs_grd_idn' or permutations:    abs + grd + idn
                - 'abs_grd_elm' or permutations:    abs + grd + elm
                - 'abs_idn_elm' or permutations:    abs + idn + elm
                - 'grd_idn_elm' or permutations:    grd + idn + elm
                - 'abs_elm_grd_idn' or permutations: abs + grd + idn + elm
        '''

        self.losstype = losstype

    def get_losstype(self):
        return self.losstype

    def init_loss(self):
        '''
        Initialise the losses for the current epoch.

        This loss increases by the losses calculated for each sample, 
        summed (see 'adjust_loss()').
        '''
        self.tot.append(0)
        self.abs.append(0)
        self.grd.append(0)
        self.idn.append(0)
        self.elm.append(0)

    def change_norm(self, norm):
        '''
        Change the normalisation factors for the losses.
        '''
        self.norm = norm

    def change_fract(self, fract):
        '''
        Change the factors to multiply the losses with.
        '''
        self.fract = fract

    def set_tot_loss(self,loss):   
        '''
        Set the total loss for an epoch.
        ''' 
        self.tot.append(loss)

    def get_tot_loss(self):   
        '''
        Get the total loss over epochs.
        '''
        return self.tot
    
    def set_loss(self,loss,type, nb):
        '''
        Set the loss, normalised by the number of samples (nb) 
        for the epoch per type.
        '''
        loss = loss/nb

        if type == 'tot':
            self.tot.append(loss)
        if type == 'abs':
            self.abs.append(loss)
        elif type == 'grd':
            self.grd.append(loss)
        elif type == 'idn':
            self.idn.append(loss)
        elif type == 'elm':
            self.elm.append(loss)

        return

    def normalise_loss(self,nb):
        '''
        Normalise all losses by number of samples (nb).
        '''
        self.tot = np.array(self.tot)/nb
        self.abs = np.array(self.abs)/nb
        self.grd = np.array(self.grd)/nb
        self.idn = np.array(self.idn)/nb
        self.elm = np.array(self.elm)/nb

        return

    def get_loss(self,type):
        '''
        Get the loss, given the type.
        '''
        if type == 'tot':
            return self.tot
        if type == 'abs':
            return self.abs
        elif type == 'grd':
            return self.grd
        elif type == 'idn':
            return self.idn
        elif type == 'elm':
            return self.elm
        
    def adjust_loss(self, type,loss):
        '''
        Adjust the loss within one epoch, per type.
        '''
        if type == 'tot':
            self.tot[-1] += loss
        if type == 'abs':
            self.abs[-1] += loss
        elif type == 'grd':
            self.grd[-1] += loss
        elif type == 'idn':
            self.idn[-1] += loss
        elif type == 'elm':
            self.elm[-1] += loss

    
    def get_all_losses(self):
        '''
        Get all losses.
        '''
        all_loss = {'tot': self.get_loss('tot'),
                    'abs': self.get_loss('abs'), 
                    'grd': self.get_loss('grd'),
                    'idn': self.get_loss('idn'),
                    'elm': self.get_loss('elm')}

        return all_loss
    

    def calc_loss(self, n, n_evol, nhat_evol,z_hat, p, model):
        '''
        Function to calculate the losses of the model.

        Input:
        - n         = abundances
        - n_evol    = real evolution
        - nhat_evol = predicted abundances/evolution
        - p         = physical parameters
        - model     = ML architecture to be trained
        - loss_obj  = loss object to store losses of training

        Returns:
        - mse of the abs and idn losses
        '''

        if 'abs' in self.losstype:
            abs = abs_loss(n_evol, nhat_evol)  /self.norm['abs']* self.fract['abs']
        if 'abs' not in self.losstype:
            abs = torch.from_numpy(np.array([0.]))

        if 'grd' in self.losstype:
            grd = grd_loss(n_evol,nhat_evol)   /self.norm['grd']* self.fract['grd']
        if 'grd' not in self.losstype:
            grd = torch.from_numpy(np.array([0.]))

        if 'idn' in self.losstype:
            idn = idn_loss(n[:-1], p, model)   /self.norm['idn']* self.fract['idn']
        if 'idn' not in self.losstype:
            idn = torch.from_numpy(np.array([0.]))

        if 'elm' in self.losstype:
            elm = elm_loss(z_hat,model, self.M) /self.norm['elm']* self.fract['elm']
        if 'elm' not in self.losstype:
            elm = torch.from_numpy(np.array([0.]))

        # print(grd) 
        loss = abs.mean() + grd.mean() + idn.mean() + elm.mean()
        # print(loss)
        self.adjust_loss('tot', loss.item())
        self.adjust_loss('abs', abs.mean().item())
        self.adjust_loss('grd', grd.mean().item())
        self.adjust_loss('idn', idn.mean().item())
        self.adjust_loss('elm', elm.mean().item())

        return loss
        

    def save(self, path):
        '''
        Save the losses to a .npy file in the given path.
        '''
        utils.makeOutputDir(path)
        
        tot_loss = self.get_tot_loss()
        abs_loss = self.get_loss('abs')
        grd_loss = self.get_loss('grd')
        idn_loss = self.get_loss('idn')
        elm_loss = self.get_loss('elm')

        
        if tot_loss is not None:
            np.save(path+'/tot.npy', tot_loss)
        if abs_loss is not None:
            np.save(path+'/abs.npy', abs_loss)
        if grd_loss is not None:
            np.save(path+'/grd.npy', grd_loss)
        if idn_loss is not None:
            np.save(path+'/idn.npy', idn_loss)
        if elm_loss is not None:
            np.save(path+'/elm.npy', elm_loss)

    def normalise(self):
        ## normalise the losses
        norm = {'abs' :np.mean(self.get_loss('abs')), # type: ignore
                'grd' :np.mean(self.get_loss('grd')), # type: ignore
                'idn' :np.mean(self.get_loss('idn')), # type: ignore
                'elm' :np.mean(self.get_loss('elm'))}   # type: ignore
        
        self.change_norm(norm) 
        
        return norm

    

def abs_loss(x, x_hat):
    '''
    Return the squared absolute loss (abs) per x_i.
    '''
    loss = (x-x_hat)**2
    # print(x.shape, x_hat.shape)
    return loss


def grd_loss(x,x_hat):
    '''
    Return the squared gradient loss per x_i, using the gradient function of PyTorch. 
    '''

    loss = (torch.gradient(x)[0] - torch.gradient(x_hat[0])[0])**2
    
    return loss

def idn_loss(x,p, model):
    '''
    Return the squared identity loss per x_i, 
        i.e. compares x to D(E(x)), 
        with E the encoder and D the decoder.
    '''
    E = model.encoder
    D = model.decoder

    x_E     = torch.cat((p, x), axis=-1) # type: ignore

    loss = (x-D(E(x_E)))**2 

    return loss

def elm_loss(z_hat,model, M):
    '''
    Return the element conservation loss per x_i.
        M is at matrix that gives the elemental composition of each species. --> use buildM.py to create this matrix.
        We know that M x n_hat should be conserved at all times in the network, hence d(M x n_hat)/dt = 0.
        Since n_hat = D(g(z_hat)), with D the decoder, g=C+Az+Bzz the ODE function,
            we can rewrite the element conservation loss 
            as d(M x D(g(z_hat)))/dt = Mgrad(D)g = Mgrad(D)(C+A+B).
        The einsum summation takes into account the right indexing.

    (For more details, see Maes et al., 2024)

    NOTE:
        This function is not used in the current version of MACE, since it 
        is found to be computationally to slow in the way it is currently implemented.
    '''
    # tic = time()

    M = torch.from_numpy(M).T     ## eventueel nog specifiek een sparse matrix van maken    

    D = model.decoder
    A = model.g.A
    B = model.g.B
    C = model.g.C
    dt_dim = z_hat.shape[0]
    jac_D = jacobian(D,z_hat, strategy='forward-mode', vectorize=True).view(468,dt_dim,dt_dim,-1)

    # print(M.shape, A.shape, B.shape, C.shape, jac_D.shape)
    
    L0 = torch.einsum("ZN , Nbci , i   -> bcZ  ", M , jac_D , C).mean()
    L1 = torch.einsum("ZN , Nbci , ij  -> bcZj ", M , jac_D , A).mean()
    L2 = torch.einsum("ZN , Nbci , ijk -> bcZjk", M , jac_D , B).mean()
    # toc = time()
    # print('time elm loss: ', toc-tic)
    
    loss = (L0 + L1 + L2)**2
    return loss


def initialise():
    norm = {'abs' : 1,
            'grd' : 1,
            'idn' : 1,
            'elm' : 1}

    fract = {'abs' : 1,
            'grd' : 1,
            'idn' : 1, 
            'elm' : 1}
    
    return norm, fract


class LoadedLoss():
    def __init__(self, loc, meta, type):
        '''
        Initialise the loss object, used for analysing a trained model.

        Input:
            - loc:  location of the model
            - meta: meta data of the model
            - type: type of loss object, either 'train' or 'test'
        '''

        self.losstype  = meta['losstype']
        
        self.type = type

        self.load(loc)
    

    def get_losstype(self):
        return self.losstype

    def set_tot_loss(self,loss):   
        '''
        Set the total loss for the epoch.
        ''' 
        self.tot = loss

    def get_tot_loss(self):   
        '''
        Get the total loss for the epoch.
        '''
        return self.tot
    
    def set_loss(self,loss,type):
        '''
        Set the loss for the epoch.
        '''
        if type == 'abs':
            self.abs = loss
        elif type == 'rel':
            self.rel = loss
        elif type == 'grd':
            self.grd = loss 
        elif type == 'idn':
            self.idn = loss

        
    def get_loss(self,type):
        '''
        Get the loss for the epoch.
        '''
        if type == 'abs':
            return self.abs
        elif type == 'rel':
            return self.rel
        elif type == 'grd':
            return self.grd 
        elif type == 'idn':
            return self.idn 
    
    def get_all_losses(self):
        return self.get_loss('abs'), self.get_loss('rel'), self.get_loss('grd'), self.get_loss('idn')
        

    def load(self, loc):
        '''
        Load the losses from a .npy file in the given path.
        '''
        if os.path.exists(loc+self.type+'/abs.npy'):
            self.set_loss(np.load(loc+self.type+'/abs.npy'), 'abs')
        else:
            self.abs = None

        if os.path.exists(loc+self.type+'/rel.npy'):
            self.set_loss(np.load(loc+self.type+'/rel.npy'), 'rel')
        else:
            self.rel = None

        if os.path.exists(loc+self.type+'/grd.npy'):
            self.set_loss(np.load(loc+self.type+'/grd.npy'), 'grd')
        else:
            self.grd = None

        if os.path.exists(loc+self.type+'/idn.npy'):
            self.set_loss(np.load(loc+self.type+'/idn.npy'), 'idn')
        else:
            self.idn = None
        
        self.set_tot_loss(np.load(loc+self.type+'/tot.npy'))


        return

    
def plot(train, test, len=10, log = True, ylim = False, limits = None, show = False):
    '''
    Plot the loss of the model as a function of epoch.
    The total loss is plotted, as well as the individual losses.

    Input:
    - train:    training loss object
    - test:     test loss object
    - len:      number of epochs                        [default = 10]
    - log:      boolean, plot the y-axis in log scale   [default = True]
    - ylim:     boolean, set the y-axis limits          [default = False]
    - limits:   tuple with the limits of the y-axis     [default = None]
    - show:     boolean, show the plot                  [default = False]

    Returns the figure.
    '''

    fig = plt.figure(figsize = (6,3))
    ax1 = fig.add_subplot((111))

    a = 0.8
    lw2 = 4

    if len >= 50:
        ms = 0.1
        lw = 1.5
    else:
        ms = 5
        lw = 1
    m1 = '.'
    m2 = 'x'

    l1 = '-'
    l2 = '--'

    ## ------------ legend ----------------

    l_train = mlines.Line2D([],[], color = 'grey', ls = l1, marker = m1, label='train',lw = lw, alpha = 1)
    l_test  = mlines.Line2D([],[], color = 'grey', ls = l2, marker = m2, label='validation' ,lw = lw, alpha = 1)
    l_tot   = mlines.Line2D([],[], color = 'k'   , ls = l1, label='total',lw = lw2, alpha = 1)
    
    handles = [l_train, l_test, l_tot]

    ## ------------- TOTAL ------------
    ax1.plot(test.get_tot_loss() , ls = l2, marker = m2, ms = ms, lw = lw, c='k')
    ax1.plot(train.get_tot_loss(), ls = l1, marker = m1, ms = ms, lw = lw, c='k')

    ## ------------- GRD -------------
    c_grd = 'gold'
    if 'evo' in train.losstype or 'grd' in train.losstype:
        ax1.plot(test.get_loss('grd') , ls = l2, marker = m2, ms=ms, lw = lw, c=c_grd, alpha = a)
        ax1.plot(train.get_loss('grd'), ls = l1, marker = m1, ms=ms, lw = lw, c=c_grd, alpha = a)
        l_grd = mlines.Line2D([],[], color = c_grd, ls = l1, label='GRD',lw = lw2, alpha = 1)
        handles.append(l_grd)

    ## ------------- IDN -------------
    c_idn = 'salmon'
    if 'idn' in train.losstype:
        ax1.plot(test.get_loss('idn') , ls = l2, marker = m2, ms=ms, lw = lw, c=c_idn, alpha = a)
        ax1.plot(train.get_loss('idn'), ls = l1, marker = m1, ms=ms, lw = lw, c=c_idn, alpha = a)
        l_idn = mlines.Line2D([],[], color = c_idn, ls = l1, label='IDN',lw = lw2, alpha = 1)
        handles.append(l_idn)

    ## ------------- ELM -------------
    c_elm = 'darkorchid'
    if 'elm' in train.losstype:
        ax1.plot(test.get_loss('elm') , ls = l2, marker = m2, ms=ms, lw = lw, c=c_elm, alpha = a)
        ax1.plot(train.get_loss('elm'), ls = l1, marker = m1, ms=ms, lw = lw, c=c_elm, alpha = a)
        l_elm = mlines.Line2D([],[], color = c_elm, ls = l1, label='elm',lw = lw2, alpha = 1)
        handles.append(l_elm)

    ## ------------- ABS -------------
    c_abs = 'cornflowerblue'
    if 'mse' in train.losstype or 'abs' in train.losstype:
        ax1.plot(test.get_loss('abs') , ls = l2, marker = m2, ms=ms, lw = lw, c=c_abs, alpha = a)
        ax1.plot(train.get_loss('abs'), ls = l1, marker = m1, ms=ms, lw = lw, c=c_abs, alpha = a)
        l_abs   = mlines.Line2D([],[], color = c_abs, ls = l1,label='ABS',lw = lw2, alpha = 1)
        handles.append(l_abs)

    ## ------------ settings --------------
    plt.rcParams.update({'font.size': 14})    

    if log == True:
        ax1.set_yscale('log') 

    if ylim == True:
        if limits == None:
            ax1.set_ylim(1e-2,1e0)
        else:
            ax1.set_ylim(limits)

    # ax1.set_xlim(5,100)

    fs= 12
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    # ax1.set_xlim([5.5,7.5])

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    fs1 = 10
    ax1.legend(handles=handles,loc = 'upper right', fontsize = fs1)

    
    plt.tight_layout()

    if show == True:
        plt.show()


    return fig
    



    
