import torch
import numpy as np
import utils

class Loss():
    def __init__(self, norm, fract):
        '''
        Initialise the loss object, which will contain the losses for each epoch. 

        norm:       dict with normalisation factors for each loss
        fract:      dict with factors to multiply each loss with
        '''
        self.norm  = norm
        self.fract = fract

        self.tot = list()

        self.mse = list()
        self.rel = list()
        self.evo = list()

        self.mse_idv = list()
        self.rel_idv = list()
        self.evo_idv = list()

    def set_losstype(self, losstype):
        '''
        Set the type of loss used.
            losstype:   string with the type of loss used
                - 'mse':                            mean squared error
                - 'rel':                            relative change in abundance
                - 'evo':                            relative evolution
                - 'mse_rel' or 'rel_mse':           mse + rel
                - 'mse_evo' or 'evo_rel':           mse + evo
                - 'rel_evo' or 'evo_rel':           rel + evo
                - 'mse_rel_evo' or permulations:    mse + rel + evo
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
    
    def set_loss(self,loss,type):
        '''
        Set the loss for the epoch.
        '''
        if type == 'mse':
            self.mse.append(loss)
        elif type == 'rel':
            self.rel.append(loss)
        elif type == 'evo':
            self.evo.append(loss)

    def set_idv_loss(self,loss,type):
        if type == 'mse':
            self.mse_idv.append(loss)
        elif type == 'rel':
            self.rel_idv.append(loss)
        elif type == 'evo':
            self.evo_idv.append(loss)
        
    def get_loss(self,type):
        '''
        Get the loss for the epoch.
        '''
        if type == 'mse':
            return np.array(self.mse)
        elif type == 'rel':
            return np.array(self.rel)
        elif type == 'evo':
            return np.array(self.evo )
    
    def get_all_losses(self):
        return self.get_loss('mse'), self.get_loss('rel'), self.get_loss('evo')
        
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
        '''
        Save the losses to a .npy file in the given path.
        '''
        utils.makeOutputDir(path)
        
        tot_loss = self.get_tot_loss()
        mse_loss = self.get_loss('mse')
        rel_loss = self.get_loss('rel')
        evo_loss = self.get_loss('evo')
        mse_idv_loss = self.get_idv_loss('mse')
        rel_idv_loss = self.get_idv_loss('rel')
        evo_idv_loss = self.get_idv_loss('evo')
        
        if tot_loss is not None:
            np.save(path+'/tot.npy', tot_loss)
        if mse_loss is not None:
            np.save(path+'/mse.npy', mse_loss)
        if rel_loss is not None:
            np.save(path+'/rel.npy', rel_loss)
        if evo_loss is not None:
            np.save(path+'/evo.npy', evo_loss)
        if mse_idv_loss is not None:
            np.save(path+'/mse_idv.npy', mse_idv_loss)
        if rel_idv_loss is not None:
            np.save(path+'/rel_idv.npy', rel_idv_loss)
        if evo_idv_loss is not None:
            np.save(path+'/evo_idv.npy', evo_idv_loss)

    

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
    eps = 1e-10
    loss = (x_hat/(x+eps) - 1)**2    
    return loss

def evo_loss(x,x_hat):
    '''
    Return the relative evolutions loss per x_i.
    The relative evolutions loss (EVO) is given by ((x_hat-x_0+eps**2)/(x-x_0+eps))**2, 
        where eps makes sure we don't devide by 0.
    '''

    x_0 = x[:-1]  ## Initial abundances
    x   = x[1:]   ## Abundances to compare with
    ## Hence, x-x_0 gives the evolution of the abundances

    eps = 1e-10
    loss  = ((torch.abs(x_hat-x_0)+eps**2)/(torch.abs(x-x_0)+eps))**2     ## absolute waarden nemen rond x, zodat het niet nog 0 kan worden
    return loss

def loss_function(loss_obj, x,x_hat):
    '''
    Get the MSE loss and the relative loss, normalised to the maximum lossvalue.
        - fracts are scaling factors, put to 0 if you want to exclude one of both losses.
    Returns the MSE loss per species, and the relative loss per species.
    '''
    mse = (mse_loss(x[1:],x_hat))     ## Compare with the final abundances for that model
    rel = (rel_loss(x[1:],x_hat))     ## Compare with the final abundances for that model
    evo = (evo_loss(x,x_hat))

    mse = mse/loss_obj.norm['mse']* loss_obj.fract['mse']
    rel = rel/loss_obj.norm['rel']* loss_obj.fract['rel']
    evo = evo/loss_obj.norm['evo']* loss_obj.fract['evo']

    return mse, rel, evo

def get_loss(mse, rel, evo, type):
    mse = mse.mean()
    rel = rel.mean()
    evo = evo.mean()

    ## only 1 type of loss
    if type == 'mse':
        return mse
    if type =='rel':
        return rel
    if type =='evo':
        return evo
    
    ## 2 types of losses
    if type =='mse_rel' or type == 'rel_mse':
        return mse+rel
    if type =='rel_evo' or type == 'evo_rel':
        return rel+evo
    if type =='mse_evo' or type == 'evo_mse':
        return mse+evo

    ## 3 types of losses
    if type =='mse_rel_evo' or type == 'mse_evo_rel' or type == 'rel_mse_evo' or type == 'rel_evo_mse' or type == 'evo_mse_rel' or type == 'evo_rel_mse':
        return mse+rel+evo


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
                - 'evo':                            relative evolution
                - 'mse_rel' or 'rel_mse':           mse + rel
                - 'mse_evo' or 'evo_rel':           mse + evo
                - 'rel_evo' or 'evo_rel':           rel + evo
                - 'mse_rel_evo' or permulations:    mse + rel + evo
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
        elif type == 'evo':
            self.evo = loss 

    def set_idv_loss(self,loss,type):
        if type == 'mse':
            self.mse_idv = loss
        elif type == 'rel':
            self.rel_idv = loss
        elif type == 'evo':
            self.evo_idv = loss 
        
    def get_loss(self,type):
        '''
        Get the loss for the epoch.
        '''
        if type == 'mse':
            return self.mse
        elif type == 'rel':
            return self.rel
        elif type == 'evo':
            return self.evo 
    
    def get_all_losses(self):
        return self.get_loss('mse'), self.get_loss('rel'), self.get_loss('evo')
        
    def get_idv_loss(self,type):
        if type == 'mse':
            return self.mse_idv
        elif type == 'rel':
            return self.rel_idv
        elif type == 'evo':
            return self.evo_idv
        
    def get_all_idv_losses(self):
        return self.get_idv_loss('mse'), self.get_idv_loss('rel'), self.get_idv_loss('evo')

    def load(self, loc, type, meta):
        '''
        Load the losses from a .npy file in the given path.
        '''
        self.set_loss(np.load(loc+type+'/mse.npy'), 'mse')
        self.set_loss(np.load(loc+type+'/rel.npy'), 'rel')
        self.set_loss(np.load(loc+type+'/evo.npy'), 'evo')
        self.set_tot_loss(np.load(loc+type+'/tot.npy'))

        self.set_idv_loss(np.load(loc+type+'/mse_idv.npy'), 'mse')
        self.set_idv_loss(np.load(loc+type+'/rel_idv.npy'), 'rel')
        self.set_idv_loss(np.load(loc+type+'/evo_idv.npy'), 'evo')

        self.set_norm(meta['norm'])
        self.set_fract(meta['fract'])

        self.set_losstype(meta['losstype'])

        return

    

    



    
