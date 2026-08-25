'''
This script contains the core of the local training scheme.
More details on this training scheme can be found in Maes et a;. (2024).

https://ui.adsabs.harvard.edu/abs/2024arXiv240503274M/abstract
'''


def run_epoch(data_loader, model, loss_obj, training):
    '''
    Function to train 1 epoch.

    Input:
    - data_loader   = data, torchtensor
    - model         = ML architecture to be trained
    - loss_obj      = object that stores the losses
    - training      = boolean to indicate if the model is training or validating

    Method:
    1. initialise loss for this epoch
    2. get data
    3. pass it through the model, x_hat = result
    4. calculate loss (difference between x & x_hat), according to loss function defined in loss_function()
    5. with optimiser, get the gradients and update weights using back propagation.
    
    Returns 
    - total number of samples
    - status of the solver
    '''    

    loss_obj.init_loss()
    DEVICE = model.DEVICE
    optimiser = model.optimiser

    status = 0

    for i, (n,p,dt) in enumerate(data_loader):
        
        n  = n.to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
        p  = p.to(DEVICE)
        dt = dt.to(DEVICE)

        print('calculate evolution...', flush=True)
        n_hat, z_hat, modstatus = model(n[:,:-1,:],p,dt)    ## Give to the solver abundances[0:tlast-1] 

        ## Calculate losses
        print('calculate losses...', flush=True)
        loss = loss_obj.calc_loss(n[:,:-1,:],n[:,1:,:], n_hat, z_hat, p, model)
        status += modstatus.sum().item()

        if training == True:
            ## Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    return i+1, status



