

# def calc_loss(n, n_hat, z_hat, p, model, loss_obj):
#     '''
#     Function to calculate the different losses.

#     Input:
#     - n         = true abundances
#     - n_hat     = predicted abundances
#     - z_hat     = predicted latent variables  (needed for elm loss, not included here)
#     - p         = physical parameters
#     - model     = MACE model
#     - loss_obj  = object that stores the losses

#     Method:
#     1. calculate the absolute loss, normalised (given by norm) and weighted (given by fract)
#     2. calculate the gradient loss, normalised (given by norm) and weighted (given by fract)
#     3. calculate the identity loss, normalised (given by norm) and weighted (given by fract)
#     4. calculate the total loss, which is the sum of the absolute, gradient and identity loss
#     5. stores the losses in the loss object

#     Returns:
#     - mse of abs, idn and grd losses
#     '''

#     abs = ls.abs_loss(n[1:], n_hat)          /loss_obj.norm['abs']* loss_obj.fract['abs']
#     grd = ls.grd_loss(n[1:], n_hat)              /loss_obj.norm['grd']* loss_obj.fract['grd']
#     idn = ls.idn_loss(n[:-1], p, model)      /loss_obj.norm['idn']* loss_obj.fract['idn']

#     loss = abs.mean() + idn.mean() + grd.mean()

#     loss_obj.adjust_loss('tot', loss.item())
#     loss_obj.adjust_loss('abs', abs.mean().item())
#     loss_obj.adjust_loss('grd', grd.mean().item())
#     loss_obj.adjust_loss('idn', idn.mean().item())


#     return loss



def run_epoch(data_loader, model, loss_obj, DEVICE, optimizer, training):
    '''
    Function to train 1 epoch.

    Input:
    - data_loader   = data, torchtensor
    - model         = ML architecture to be trained
    - loss_obj      = object that stores the losses
    - DEVICE        = device to run the model on
    - optimizer     = type of optimizer to update the weights of the model
    - training      = boolean to indicate if the model is training or validating
    - nb_evol       == 0 ; needed to be compatible with 'run_epoch()' function in integrated.py

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

    status = 0

    for i, (n,p,dt) in enumerate(data_loader):
        
        n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
        p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
        dt = dt.view(dt.shape[1]).to(DEVICE)

        n_hat, z_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 

        ## Calculate losses
        loss = loss_obj.calc_loss(n,n[1:], n_hat, z_hat, p, model)
        status += modstatus.sum().item()

        if training == True:
            ## Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return i+1, status



