import torch
import numpy as np
import utils

class Loss():
    def __init__(self, norm, fract):
        self.norm  = norm
        self.fract = fract


        self.tot = list()

        self.mse = list()
        self.rel = list()
        self.evo = list()

        self.mse_idv = list()
        self.rel_idv = list()
        self.evo_idv = list()

    def change_norm(self, norm):
        self.norm = norm

    def change_fract(self, fract):
        self.fract = fract

    def set_tot_loss(self,loss):    
        self.tot.append(loss)

    def get_tot_loss(self):   
        return np.array(self.tot)
    
    def set_loss(self,loss,type):
        if type == 'mse':
            self.mse.append(loss)
        elif type == 'rel':
            self.rel.append(loss)
        elif type == 'evo':
            self.evo.append(loss)
        
    def get_loss(self,type):
        if type == 'mse':
            return np.array(self.mse)
        elif type == 'rel':
            return np.array(self.rel)
        elif type == 'evo':
            return np.array(self.evo )
    
    def get_all_losses(self):
        return self.get_loss('mse'), self.get_loss('rel'), self.get_loss('evo')
    
    def set_idv_loss(self,loss,type):
        if type == 'mse':
            self.mse_idv.append(loss)
        elif type == 'rel':
            self.rel_idv.append(loss)
        elif type == 'evo':
            self.evo_idv.append(loss)
        
    def get_idv_loss(self,type):
        if type == 'mse':
            return np.array(self.mse_idv).T
        elif type == 'rel':
            return np.array(self.rel_idv).T
        elif type == 'evo':
            return np.array(self.evo_idv).T
        
    def get_all_idv_losses(self):
        return self.get_idv_loss('mse'), self.get_idv_loss('rel'), self.get_idv_loss('evo')
    

    def save(self, path):
        utils.makeOutputDir(path)
        np.save(path+'/tot.npy',self.get_tot_loss())
        np.save(path+'/mse.npy',self.get_loss('mse'))
        np.save(path+'/rel.npy',self.get_loss('rel'))
        np.save(path+'/evo.npy',self.get_loss('evo'))
        np.save(path+'/mse_idv.npy',self.get_idv_loss('mse'))
        np.save(path+'/rel_idv.npy',self.get_idv_loss('rel'))
        np.save(path+'/evo_idv.npy',self.get_idv_loss('evo'))  

    

    

def mse_loss(x, x_hat):
    '''
    Return the mean squared loss (MSE) per x_i.
    '''
    loss = (x-x_hat)**2
    return loss

def rel_loss(x, x_hat):
    '''
    Return the relative mean squared loss (REL) per x_i.
    '''
    loss = (x_hat/x -1 )**2
    return loss

def evo_loss(x,x_hat):
    '''
    Return the relative evolutions loss per x_i.
    The relative evolutions loss (EVO) is given by ((x_hat-x_0+eps**2)/(x-x_0+eps))**2, 
        where eps makes sure we don't devide by 0.
    '''

    x_0   = x[:,0,:]
    eps = 1e-10
    loss  = ((torch.abs(x_hat-x_0)+eps**2)/(torch.abs(x-x_0)+eps))**2     ## absolute waarden nemen rond x, zodat het niet nog 0 kan worden
    return loss

def loss_function(loss_obj, x,x_hat):
    '''
    Get the MSE loss and the relative loss, normalised to the maximum lossvalue.
        - f_mse and f_rel are scaling factors, put to 0 if you want to exclude one of both losses.
    Returns the MSE loss per species, and the relative loss per species.
    '''
    mse = (mse_loss(x,x_hat))
    rel = (rel_loss(x,x_hat))
    evo = (evo_loss(x,x_hat))

    mse = mse/loss_obj.norm['mse']* loss_obj.fract['mse']
    rel = rel/loss_obj.norm['rel']* loss_obj.fract['rel']
    evo = evo/loss_obj.norm['evo']* loss_obj.fract['evo']

    return mse, rel, evo
