import torch
import numpy as np
import utils
from torch.autograd.functional import jacobian
from time import time
import os

class Loss():
    def __init__(self, norm, fract):
        '''
        Initialise the loss object, which will contain the losses for each epoch. 

        norm:       dict with normalisation factors for each loss
        fract:      dict with factors to multiply each loss with

        Different types of losses:
            - 'mse':    mean squared error
            - 'rel':    relative change in abundance
            - 'grd':    relative grdlution
            - 'idn':    identity loss = losses due to autoencoder
            - 'elm':    element conservation loss
        '''
        self.norm  = norm
        self.fract = fract

        self.tot = list()

        self.mse = list()
        self.rel = list()
        self.grd = list()
        self.idn = list()
        self.elm = list()

        self.mse_idv = list()
        self.rel_idv = list()
        self.grd_idv = list()

        self.M = np.load('/STER/silkem/ChemTorch/rates/M_rate16.npy')

    def set_losstype(self, losstype):
        '''
        Set the type of loss used.
            losstype:   string with the type of loss used
                - 'mse':                            mean squared error
                - 'rel':                            relative change in abundance
                - 'grd':                            relative grdlution
                - 'mse_rel' or 'rel_mse':           mse + rel
                - 'mse_grd' or 'grd_rel':           mse + grd
                - 'rel_grd' or 'grd_rel':           rel + grd
                - 'mse_rel_grd' or permulations:    mse + rel + grd
        '''
        self.type = losstype

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
        Set the total loss for the epoch.
        ''' 
        self.tot.append(loss)

    def get_tot_loss(self):   
        '''
        Get the total loss for the epoch.
        '''
        return np.array(self.tot)
    
    def set_loss(self,loss,type, nb):
        '''
        Set the loss for the epoch.
        '''
        loss = loss/nb

        if type == 'tot':
            self.tot.append(loss)
        if type == 'mse':
            self.mse.append(loss)
        elif type == 'rel':
            self.rel.append(loss)
        elif type == 'grd':
            self.grd.append(loss)
        elif type == 'idn':
            self.idn.append(loss)
        elif type == 'elm':
            self.elm.append(loss)

    def set_loss_all(self,loss_dict,nb):
        '''
        Set the losses for the epoch.
        '''
        for key in loss_dict.keys():
            self.set_loss(loss_dict[key],key,nb)

    def set_idv_loss(self,loss,type):
        if type == 'mse':
            self.mse_idv.append(loss)
        elif type == 'rel':
            self.rel_idv.append(loss)
        elif type == 'grd':
            self.grd_idv.append(loss)
        
    def get_loss(self,type):
        '''
        Get the loss for the epoch.
        '''
        if type == 'mse':
            return np.array(self.mse)
        elif type == 'rel':
            return np.array(self.rel)
        elif type == 'grd':
            return np.array(self.grd)
        elif type == 'idn':
            return np.array(self.idn)
        elif type == 'elm':
            return np.array(self.elm)
    
    def get_all_losses(self):
        return self.get_loss('mse'), self.get_loss('rel'), self.get_loss('grd'), self.get_loss('idn'), self.get_loss('elm')
        
    def get_idv_loss(self,type):
        if type == 'mse':
            return np.array(self.mse_idv).T
        elif type == 'rel':
            return np.array(self.rel_idv).T
        elif type == 'grd':
            return np.array(self.grd_idv).T
        
    def get_all_idv_losses(self):
        return self.get_idv_loss('mse'), self.get_idv_loss('rel'), self.get_idv_loss('grd')


    def save(self, path):
        '''
        Save the losses to a .npy file in the given path.
        '''
        utils.makeOutputDir(path)
        
        tot_loss = self.get_tot_loss()
        mse_loss = self.get_loss('mse')
        rel_loss = self.get_loss('rel')
        grd_loss = self.get_loss('grd')
        idn_loss = self.get_loss('idn')
        elm_loss = self.get_loss('elm')

        # mse_idv_loss = self.get_idv_loss('mse')
        # rel_idv_loss = self.get_idv_loss('rel')
        # grd_idv_loss = self.get_idv_loss('grd')
        
        if tot_loss is not None:
            np.save(path+'/tot.npy', tot_loss)
        if mse_loss is not None:
            np.save(path+'/mse.npy', mse_loss)
        if rel_loss is not None:
            np.save(path+'/rel.npy', rel_loss)
        if grd_loss is not None:
            np.save(path+'/grd.npy', grd_loss)
        if idn_loss is not None:
            np.save(path+'/idn.npy', idn_loss)
        if elm_loss is not None:
            np.save(path+'/elm.npy', elm_loss)

        # if mse_idv_loss is not None:
        #     np.save(path+'/mse_idv.npy', mse_idv_loss)
        # if mse_idv_loss is not None:
        #     np.save(path+'/mse_idv.npy', mse_idv_loss)
        # if rel_idv_loss is not None:
        #     np.save(path+'/rel_idv.npy', rel_idv_loss)
        # if grd_idv_loss is not None:
        #     np.save(path+'/grd_idv.npy', grd_idv_loss)

    

def mse_loss(x, x_hat):
    '''
    Return the mean squared loss (MSE) per x_i.
    '''
    loss = (x-x_hat)**2
    # print(x.shape, x_hat.shape)
    return loss

def rel_loss(x, x_hat):
    '''
    Return the relative mean squared loss (REL) per x_i.
    '''
    eps = 1e-10
    loss = (x_hat/(x+eps) - 1)**2    
    return loss

def grd_loss(x,x_hat):
    '''
    Return the relative gradient loss per x_i.
    The relative gradient loss (grd) is given by ((x-x_0) - (x_hat-x_0))**2, 
        
    '''

    x   = x[1:]   ## ignore initial abundances

    loss = (torch.gradient(x)[0] - torch.gradient(x_hat[0])[0])**2
    
    return loss

def idn_loss(x,x_hat,p, model):
    '''
    Return the identity loss per x_i, 
        i.e. compares x to D(E(x)), 
        with E the encoder and D the decoder.
    '''
    E = model.encoder
    D = model.decoder

    x_E     = torch.cat((p, x), axis=-1) # type: ignore
    # print(x.shape,x_hat.shape,p.shape)
    # xhat_E  = torch.cat((p, x_hat.view(-1,468)), axis=-1) # type: ignore

    loss = (x-D(E(x_E)))**2 #+ (x_hat-D(E(xhat_E)))**2

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
    '''
    Get the MSE loss and the relative loss, normalised to the maximum lossvalue.
        - fracts are scaling factors, put to 0 if you want to exclude one of both losses.
    Returns the MSE loss per species, and the relative loss per species.
    '''
    mse = (mse_loss(x[1:],x_hat))     ## Compare with the final abundances for that model
    rel = (rel_loss(x[1:],x_hat))     ## Compare with the final abundances for that model
    grd = (grd_loss(x,x_hat))
    idn = (idn_loss(x[1:],x_hat,p,model))
    if 'elm' in loss_obj.type:
        elm = (elm_loss(z_hat,model, loss_obj.M))
    else:
        elm = torch.tensor([0.0,0.0])

    mse = mse/loss_obj.norm['mse']* loss_obj.fract['mse']
    rel = rel/loss_obj.norm['rel']* loss_obj.fract['rel']
    grd = grd/loss_obj.norm['grd']* loss_obj.fract['grd']
    if 'idn' in loss_obj.norm:
        idn = idn/loss_obj.norm['idn']* loss_obj.fract['idn']

    return mse, rel, grd, idn, elm

def get_loss(mse, rel, grd, idn, elm, type):
    mse = mse.mean()
    rel = rel.mean()
    grd = grd.mean()
    idn = idn.mean()
    elm = elm.mean()
    # print(elm.grad,elm)
    # elm.grad.zero_()

    ## only 1 type of loss
    if type == 'mse':
        return mse
    elif type =='rel':
        return rel
    elif type =='grd':
        return grd
    elif type =='idn':
        return idn
    elif type =='elm':
        return elm
    
    ## 2 types of losses
    elif type =='mse_rel' or type == 'rel_mse':
        return mse+rel
    elif type =='rel_grd' or type == 'grd_rel':
        return rel+grd
    elif type =='mse_grd' or type == 'grd_mse':
        return mse+grd
    elif type =='mse_idn' or type == 'idn_mse':
        return mse+idn
    elif type =='rel_idn' or type == 'idn_rel':
        return rel+idn
    elif type =='grd_idn' or type == 'idn_grd':
        return grd+idn
    elif type =='elm_idn' or type == 'idn_elm':
        return elm+idn
    elif type =='elm_rel' or type == 'rel_elm':
        return elm+rel
    elif type =='elm_grd' or type == 'grd_elm':
        return elm+grd
    elif type =='elm_mse' or type == 'mse_elm':
        return elm+mse
    

    ## 3 types of losses
    elif type =='mse_rel_grd' or type == 'mse_grd_rel' or type == 'rel_mse_grd' or type == 'rel_grd_mse' or type == 'grd_mse_rel' or type == 'grd_rel_mse':
        return mse+rel+grd
    elif type =='mse_rel_idn' or type == 'mse_idn_rel' or type == 'rel_mse_idn' or type == 'rel_idn_mse' or type == 'idn_mse_rel' or type == 'idn_rel_mse':
        return mse+rel+idn
    elif type =='mse_grd_idn' or type == 'mse_idn_grd' or type == 'grd_mse_idn' or type == 'grd_idn_mse' or type == 'idn_mse_grd' or type == 'idn_grd_mse':
        return mse+grd+idn
    elif type =='rel_grd_idn' or type == 'rel_idn_grd' or type == 'grd_rel_idn' or type == 'grd_idn_rel' or type == 'idn_rel_grd' or type == 'idn_grd_rel':
        return rel+grd+idn
    elif type =='elm_rel_idn' or type == 'elm_idn_rel' or type == 'rel_elm_idn' or type == 'rel_idn_elm' or type == 'idn_rel_elm' or type == 'idn_elm_rel':
        return elm+rel+idn
    elif type =='elm_grd_idn' or type == 'elm_idn_grd' or type == 'grd_elm_idn' or type == 'grd_idn_elm' or type == 'idn_elm_grd' or type == 'idn_grd_elm':
        return elm+grd+idn
    elif type =='elm_mse_idn' or type == 'elm_idn_mse' or type == 'mse_elm_idn' or type == 'mse_idn_elm' or type == 'idn_elm_mse' or type == 'idn_mse_elm':
        return elm+mse+idn
    elif type =='elm_mse_rel' or type == 'elm_rel_mse' or type == 'mse_elm_rel' or type == 'mse_rel_elm' or type == 'rel_elm_mse' or type == 'rel_mse_elm':
        return elm+mse+rel
    elif type =='elm_mse_grd' or type == 'elm_grd_mse' or type == 'mse_elm_grd' or type == 'mse_grd_elm' or type == 'grd_elm_mse' or type == 'grd_mse_elm':
        return elm+mse+grd
    elif type =='rel_grd_elm' or type == 'rel_elm_grd' or type == 'grd_rel_elm' or type == 'grd_elm_rel' or type == 'elm_rel_grd' or type == 'elm_grd_rel':
        return rel+grd+elm

    


    ## 4 types of losses
    elif type =='mse_rel_grd_idn' or type == 'mse_rel_idn_grd' or type == 'mse_grd_rel_idn' or type == 'mse_grd_idn_rel' or type == 'mse_idn_rel_grd' or type == 'mse_idn_grd_rel' or type == 'rel_mse_grd_idn' or type == 'rel_mse_idn_grd' or type == 'rel_grd_mse_idn' or type == 'rel_grd_idn_mse' or type == 'rel_idn_mse_grd' or type == 'rel_idn_grd_mse' or type == 'grd_mse_rel_idn' or type == 'grd_mse_idn_rel' or type == 'grd_rel_mse_idn' or type == 'grd_rel_idn_mse' or type == 'grd_idn_mse_rel' or type == 'grd_idn_rel_mse' or type == 'idn_mse_rel_grd' or type == 'idn_mse_grd_rel' or type == 'idn_rel_mse_grd' or type == 'idn_rel_grd_mse' or type == 'idn_grd_mse_rel' or type == 'idn_grd_rel_mse':
        return mse+rel+grd+idn      ## no elm
    elif type =='mse_rel_grd_elm' or type == 'mse_rel_elm_grd' or type == 'mse_grd_rel_elm' or type == 'mse_grd_elm_rel' or type == 'mse_elm_rel_grd' or type == 'mse_elm_grd_rel' or type == 'rel_mse_grd_elm' or type == 'rel_mse_elm_grd' or type == 'rel_grd_mse_elm' or type == 'rel_grd_elm_mse' or type == 'rel_elm_mse_grd' or type == 'rel_elm_grd_mse' or type == 'grd_mse_rel_elm' or type == 'grd_mse_elm_rel' or type == 'grd_rel_mse_elm' or type == 'grd_rel_elm_mse' or type == 'grd_elm_mse_rel' or type == 'grd_elm_rel_mse' or type == 'elm_mse_rel_grd' or type == 'elm_mse_grd_rel' or type == 'elm_rel_mse_grd' or type == 'elm_rel_grd_mse' or type == 'elm_grd_mse_rel' or type == 'elm_grd_rel_mse':
        return mse+rel+grd+elm      ## no idn
    elif type =='mse_rel_idn_elm' or type == 'mse_rel_elm_idn' or type == 'mse_idn_rel_elm' or type == 'mse_idn_elm_rel' or type == 'mse_elm_rel_idn' or type == 'mse_elm_idn_rel' or type == 'rel_mse_idn_elm' or type == 'rel_mse_elm_idn' or type == 'rel_idn_mse_elm' or type == 'rel_idn_elm_mse' or type == 'rel_elm_mse_idn' or type == 'rel_elm_idn_mse' or type == 'idn_mse_rel_elm' or type == 'idn_mse_elm_rel' or type == 'idn_rel_mse_elm' or type == 'idn_rel_elm_mse' or type == 'idn_elm_mse_rel' or type == 'idn_elm_rel_mse' or type == 'elm_mse_rel_idn' or type == 'elm_mse_idn_rel' or type == 'elm_rel_mse_idn' or type == 'elm_rel_idn_mse' or type == 'elm_idn_mse_rel' or type == 'elm_idn_rel_mse':
        return mse+rel+idn+elm      ## no grd
    elif type =='mse_grd_idn_elm' or type == 'mse_grd_elm_idn' or type == 'mse_idn_grd_elm' or type == 'mse_idn_elm_grd' or type == 'mse_elm_grd_idn' or type == 'mse_elm_idn_grd' or type == 'grd_mse_idn_elm' or type == 'grd_mse_elm_idn' or type == 'grd_idn_mse_elm' or type == 'grd_idn_elm_mse' or type == 'grd_elm_mse_idn' or type == 'grd_elm_idn_mse' or type == 'idn_mse_grd_elm' or type == 'idn_mse_elm_grd' or type == 'idn_grd_mse_elm' or type == 'idn_grd_elm_mse' or type == 'idn_elm_mse_grd' or type == 'idn_elm_grd_mse' or type == 'elm_mse_grd_idn' or type == 'elm_mse_idn_grd' or type == 'elm_grd_mse_idn' or type == 'elm_grd_idn_mse' or type == 'elm_idn_mse_grd' or type == 'elm_idn_grd_mse':
        return mse+grd+idn+elm      ## no rel
    elif type =='rel_grd_idn_elm' or type == 'rel_grd_elm_idn' or type == 'rel_idn_grd_elm' or type == 'rel_idn_elm_grd' or type == 'rel_elm_grd_idn' or type == 'rel_elm_idn_grd' or type == 'grd_rel_idn_elm' or type == 'grd_rel_elm_idn' or type == 'grd_idn_rel_elm' or type == 'grd_idn_elm_rel' or type == 'grd_elm_rel_idn' or type == 'grd_elm_idn_rel' or type == 'idn_rel_grd_elm' or type == 'idn_rel_elm_grd' or type == 'idn_grd_rel_elm' or type == 'idn_grd_elm_rel' or type == 'idn_elm_rel_grd' or type == 'idn_elm_grd_rel' or type == 'elm_rel_grd_idn' or type == 'elm_rel_idn_grd' or type == 'elm_grd_rel_idn' or type == 'elm_grd_idn_rel' or type == 'elm_idn_rel_grd' or type == 'elm_idn_grd_rel':
        return rel+grd+idn+elm      ## no mse
    
    ## 5 types of losses
    elif (type =='mse_rel_grd_idn_elm' or type == 'mse_rel_grd_elm_idn' or type == 'mse_rel_idn_grd_elm' or type == 'mse_rel_idn_elm_grd' or type == 'mse_rel_elm_grd_idn' 
        or type == 'mse_rel_elm_idn_grd' or type == 'mse_grd_rel_idn_elm' or type == 'mse_grd_rel_elm_idn' or type == 'mse_grd_idn_rel_elm' or type == 'mse_grd_idn_elm_rel' 
        or type == 'mse_grd_elm_rel_idn' or type == 'mse_grd_elm_idn_rel' or type == 'mse_idn_rel_grd_elm' or type == 'mse_idn_rel_elm_grd' or type == 'mse_idn_grd_rel_elm' 
        or type == 'mse_idn_grd_elm_rel' or type == 'mse_idn_elm_rel_grd' or type == 'mse_idn_elm_grd_rel' or type == 'mse_elm_rel_grd_idn' or type == 'mse_elm_rel_idn_grd' 
        or type == 'mse_elm_grd_rel_idn' or type == 'mse_elm_grd_idn_rel' or type == 'mse_elm_idn_rel_grd' or type == 'mse_elm_idn_grd_rel' or type == 'rel_mse_grd_idn_elm' 
        or type == 'rel_mse_grd_elm_idn' or type == 'rel_mse_idn_grd_elm' or type == 'rel_mse_idn_elm_grd' or type == 'rel_mse_elm_grd_idn' or type == 'rel_mse_elm_idn_grd' 
        or type == 'rel_grd_mse_idn_elm' or type == 'rel_grd_mse_elm_idn' or type == 'rel_grd_idn_mse_elm' or type == 'rel_grd_idn_elm_mse' or type == 'rel_grd_elm_mse_idn' 
        or type == 'rel_grd_elm_idn_mse' or type == 'rel_idn_mse_grd_elm' or type == 'rel_idn_mse_elm_grd' or type == 'rel_idn_grd_mse_elm' or type == 'rel_idn_grd_elm_mse'):
        return mse+rel+grd+idn+elm


class Loss_analyse():
    def __init__(self):
        '''
        Initialise the loss object, used for analysing a trained model
        '''
    
    def set_norm(self, norm):
        '''
        Set the normalisation factors for the losses.
        '''
        self.norm = norm

    def set_fract(self, fract):
        '''
        Set the factors to multiply the losses with.
        '''
        self.fract = fract

    def set_losstype(self, losstype):
        '''
        Set the type of loss used.
            losstype:   string with the type of loss used
                - 'mse':                            mean squared error
                - 'rel':                            relative change in abundance
                - 'grd':                            relative grdlution
                - 'mse_rel' or 'rel_mse':           mse + rel
                - 'mse_grd' or 'grd_rel':           mse + grd
                - 'rel_grd' or 'grd_rel':           rel + grd
                - 'mse_rel_grd' or permulations:    mse + rel + grd
        '''
        self.type = losstype

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
        if type == 'mse':
            self.mse = loss
        elif type == 'rel':
            self.rel = loss
        elif type == 'grd':
            self.grd = loss 
        elif type == 'idn':
            self.idn = loss

    def set_idv_loss(self,loss,type):
        if type == 'mse':
            self.mse_idv = loss
        elif type == 'rel':
            self.rel_idv = loss
        elif type == 'grd':
            self.grd_idv = loss 
        
    def get_loss(self,type):
        '''
        Get the loss for the epoch.
        '''
        if type == 'mse':
            return self.mse
        elif type == 'rel':
            return self.rel
        elif type == 'grd':
            return self.grd 
        elif type == 'idn':
            return self.idn 
    
    def get_all_losses(self):
        return self.get_loss('mse'), self.get_loss('rel'), self.get_loss('grd'), self.get_loss('idn')
        
    def get_idv_loss(self,type):
        if type == 'mse':
            return self.mse_idv
        elif type == 'rel':
            return self.rel_idv
        elif type == 'grd':
            return self.grd_idv
        
    def get_all_idv_losses(self):
        return self.get_idv_loss('mse'), self.get_idv_loss('rel'), self.get_idv_loss('grd')

    def load(self, loc, type, meta):
        '''
        Load the losses from a .npy file in the given path.
        '''
        self.set_loss(np.load(loc+type+'/mse.npy'), 'mse')
        self.set_loss(np.load(loc+type+'/rel.npy'), 'rel')
        self.set_loss(np.load(loc+type+'/grd.npy'), 'grd')
        if os.path.exists(loc+type+'/idn.npy'):
            self.set_loss(np.load(loc+type+'/idn.npy'), 'idn')
        self.set_tot_loss(np.load(loc+type+'/tot.npy'))

        # self.set_idv_loss(np.load(loc+type+'/mse_idv.npy'), 'mse')
        # self.set_idv_loss(np.load(loc+type+'/rel_idv.npy'), 'rel')
        # self.set_idv_loss(np.load(loc+type+'/grd_idv.npy'), 'grd')

        # self.set_norm(meta['norm'])
        # self.set_fract(meta['fract'])

        self.set_losstype(meta['losstype'])

        return

    

    



    
