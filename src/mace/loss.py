import torch
import numpy as np
import utils
from torch.autograd.functional import jacobian
from time import time
import os

import matplotlib.pyplot as plt
import matplotlib        as mpl
import matplotlib.lines     as mlines

class Loss():
    def __init__(self, norm, fract, losstype):
        '''
        Initialise the loss object, which will contain the losses for each epoch. 

        norm:       dict with normalisation factors for each loss
        fract:      dict with factors to multiply each loss with

        Different types of losses:
            - 'abs':    mean squared error
            - 'grd':    gradient
            - 'idn':    identity loss = losses due to autoencoder
            - 'elm':    element conservation loss
        '''
        self.norm  = norm
        self.fract = fract

        self.tot = list()

        self.abs = list()
        self.grd = list()
        self.idn = list()
        self.elm = list()

        self.M = np.load('/STER/silkem/ChemTorch/rates/M_rate16.npy')

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
        self.type = losstype

    def get_losstype(self):
        return self.type

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

    def normalise_loss(self,nb):
        '''
        Normalise all losses by number of samples (nb).
        '''
        self.tot = np.array(self.tot)/nb
        self.abs = np.array(self.abs)/nb
        self.grd = np.array(self.grd)/nb
        self.idn = np.array(self.idn)/nb
        self.elm = np.array(self.elm)/nb

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
        all_loss = {'tot': self.get_loss('tot'),
                    'abs': self.get_loss('abs'), 
                    'grd': self.get_loss('grd'),
                    'idn': self.get_loss('idn'),
                    'elm': self.get_loss('elm')}

        return all_loss
        

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

    x   = x[1:]   ## ignore initial abundances

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
        M is at matrix that gives the elemental composition of each species.
        We know that M x n_hat should be conserved at all times in the network, hence d(M x n_hat)/dt = 0.
        Since n_hat = D(g(z_hat)), with D the decoder, g=C+Az+Bzz the ODE function,
            we can rewrite the element conservation loss 
            as d(M x D(g(z_hat)))/dt = Mgrad(D)g = Mgrad(D)(C+A+B).
        The einsum summation takes into account the right indexing.

    (More in paper)
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


def loss_function(loss_obj, model, x, x_hat,z_hat, p):
    ## wordt niet gebruikt
    '''
    Get the abs loss and the relative loss, normalised to the maximum lossvalue.
        - fracts are scaling factors, put to 0 if you want to exclude one of both losses.
    Returns the abs loss per species, and the relative loss per species.
    '''
    abs = (abs_loss(x[1:],x_hat))     ## Compare with the final abundances for that model
    grd = (grd_loss(x,x_hat))
    idn = (idn_loss(x[:-1],p,model))
    if 'elm' in loss_obj.type:
        elm = (elm_loss(z_hat,model, loss_obj.M))
    else:
        elm = torch.tensor([0.0,0.0])

    abs = abs/loss_obj.norm['abs']* loss_obj.fract['abs']
    grd = grd/loss_obj.norm['grd']* loss_obj.fract['grd']
    if 'idn' in loss_obj.norm:
        idn = idn/loss_obj.norm['idn']* loss_obj.fract['idn']

    return abs, grd, idn, elm


def get_loss(abs, rel, grd, idn, elm, type):
    ## wordt niet gebruikt
    abs = abs.mean()
    rel = rel.mean()
    grd = grd.mean()
    idn = idn.mean()
    elm = elm.mean()
    # print(elm.grad,elm)
    # elm.grad.zero_()

    ## only 1 type of loss
    if type == 'abs':
        return abs
    elif type =='grd':
        return grd
    elif type =='idn':
        return idn
    elif type =='elm':
        return elm
    
    ## 2 types of losses
    elif type =='abs_grd' or type == 'grd_abs':
        return abs+grd
    elif type =='abs_idn' or type == 'idn_abs':
        return abs+idn
    elif type =='grd_idn' or type == 'idn_grd':
        return grd+idn
    elif type =='elm_idn' or type == 'idn_elm':
        return elm+idn
    elif type =='elm_grd' or type == 'grd_elm':
        return elm+grd
    elif type =='elm_abs' or type == 'abs_elm':
        return elm+abs
    
    ## 3 types of losses
    elif type =='abs_grd_idn' or type == 'abs_idn_grd' or type == 'grd_abs_idn' or type == 'grd_idn_abs' or type == 'idn_abs_grd' or type == 'idn_grd_abs':
        return abs+grd+idn
    elif type =='elm_grd_idn' or type == 'elm_idn_grd' or type == 'grd_elm_idn' or type == 'grd_idn_elm' or type == 'idn_elm_grd' or type == 'idn_grd_elm':
        return elm+grd+idn
    elif type =='elm_abs_idn' or type == 'elm_idn_abs' or type == 'abs_elm_idn' or type == 'abs_idn_elm' or type == 'idn_elm_abs' or type == 'idn_abs_elm':
        return elm+abs+idn
    elif type =='elm_abs_rel' or type == 'elm_rel_abs' or type == 'abs_elm_rel' or type == 'abs_rel_elm' or type == 'rel_elm_abs' or type == 'rel_abs_elm':
        return elm+abs+rel
    elif type =='elm_abs_grd' or type == 'elm_grd_abs' or type == 'abs_elm_grd' or type == 'abs_grd_elm' or type == 'grd_elm_abs' or type == 'grd_abs_elm':
        return elm+abs+grd

    ## 4 types of losses
    elif type =='abs_grd_idn_elm' or type == 'abs_grd_elm_idn' or type == 'abs_idn_grd_elm' or type == 'abs_idn_elm_grd' or type == 'abs_elm_grd_idn' or type == 'abs_elm_idn_grd' or type == 'grd_abs_idn_elm' or type == 'grd_abs_elm_idn' or type == 'grd_idn_abs_elm' or type == 'grd_idn_elm_abs' or type == 'grd_elm_abs_idn' or type == 'grd_elm_idn_abs' or type == 'idn_abs_grd_elm' or type == 'idn_abs_elm_grd' or type == 'idn_grd_abs_elm' or type == 'idn_grd_elm_abs' or type == 'idn_elm_abs_grd' or type == 'idn_elm_grd_abs' or type == 'elm_abs_grd_idn' or type == 'elm_abs_idn_grd' or type == 'elm_grd_abs_idn' or type == 'elm_grd_idn_abs' or type == 'elm_idn_abs_grd' or type == 'elm_idn_grd_abs':
        return abs+grd+idn+elm      ## no rel




class Loss_analyse():
    def __init__(self, loc, meta, type):
        '''
        Initialise the loss object, used for analysing a trained model
        '''
        # self.norm  = meta['norm']
        # self.fract = meta['fract']
        self.losstype  = meta['losstype']
        
        self.type = type

        self.load(loc)
    

    def get_losstype(self):
        return self.type

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

    # def set_idv_loss(self,loss,type):
    #     if type == 'abs':
    #         self.abs_idv = loss
    #     elif type == 'rel':
    #         self.rel_idv = loss
    #     elif type == 'grd':
    #         self.grd_idv = loss 
        
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

    
def plot(train, test, log = True, ylim = False, limits = None, show = False):

    fig = plt.figure(figsize = (6,3))
    ax1 = fig.add_subplot((111))

    lw = 1.5
    a = 0.8
    lw2 = 4
    ms = 0.1
    ## ------------ legend ----------------

    l_train = mlines.Line2D([],[], color = 'grey', ls = '-' , marker = 'none', label='train',lw = lw, alpha = 1)
    l_test  = mlines.Line2D([],[], color = 'grey', ls = '--', marker = 'none', label='validation' ,lw = lw, alpha = 1)
    l_tot   = mlines.Line2D([],[], color = 'k'   , ls = '-' , label='total',lw = lw2, alpha = 1)
    
    handles = [l_train, l_test, l_tot]

    ## ------------- TOTAL ------------
    ax1.plot(test.get_tot_loss(), ls = '--', marker = 'None', lw = lw, c='k')
    ax1.plot(train.get_tot_loss(), ls = '-', marker = 'None', lw = lw, c='k')

    ## ------------- GRD -------------
    c_grd = 'gold'
    if 'evo' in train.losstype or 'grd' in train.losstype:
        ax1.plot(test.get_loss('grd'), ls = '--', marker = 'x', ms=ms, lw = lw, c=c_grd, alpha = a)
        ax1.plot(train.get_loss('grd'), ls = '-', marker = '.', ms=ms, lw = lw, c=c_grd, alpha = a)
        l_grd = mlines.Line2D([],[], color = c_grd, ls = '-', label='GRD',lw = lw2, alpha = 1)
        handles.append(l_grd)

    ## ------------- IDN -------------
    c_idn = 'salmon'
    if 'idn' in train.losstype:
        ax1.plot(test.get_loss('idn'), ls = '--', marker = 'x', ms=ms, lw = lw, c=c_idn, alpha = a)
        ax1.plot(train.get_loss('idn'), ls = '-', marker = '.', ms=ms, lw = lw, c=c_idn, alpha = a)
        l_idn = mlines.Line2D([],[], color = c_idn, ls = '-', label='IDN',lw = lw2, alpha = 1)
        handles.append(l_idn)

    ## ------------- ELM -------------
    c_elm = 'darkorchid'
    if 'elm' in train.losstype:
        ax1.plot(test.get_loss('elm'), ls = '--', marker = 'x', ms=ms, lw = lw, c=c_elm, alpha = a)
        ax1.plot(train.get_loss('elm'), ls = '-', marker = '.', ms=ms, lw = lw, c=c_elm, alpha = a)
        l_elm = mlines.Line2D([],[], color = c_elm, ls = '-', label='elm',lw = lw2, alpha = 1)
        handles.append(l_elm)

    ## ------------- ABS -------------
    c_abs = 'cornflowerblue'
    if 'mse' in train.losstype or 'abs' in train.losstype:
        ax1.plot(test.get_loss('abs'), ls = '--', marker = 'x', ms=ms, lw = lw, c=c_abs, alpha = a)
        ax1.plot(train.get_loss('abs'), ls = '-', marker = '.', ms=ms, lw = lw, c=c_abs, alpha = a)
        l_abs   = mlines.Line2D([],[], color = c_abs, ls = '-',label='ABS',lw = lw2, alpha = 1)
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

    ax1.set_xlim(5,100)

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
    



    
