# from time   import time
# import matplotlib.pyplot as plt

# import torch
# from torch.optim  import Adam

# ## own scripts
# impor
# t chempy_0D.plotting as plotting 
import loss as ls



def calc_losses(n, n_hat, z_hat, p, model, loss_obj):
    '''
    Function to calculate the different losses.

    Input:
    - n         = true abundances
    - n_hat     = predicted abundances
    - z_hat     = predicted latent variables  (needed for elm loss, not included here)
    - p         = physical parameters
    - model     = MACE model
    - loss_obj  = object that stores the losses

    Method:
    1. calculate the absolute loss, normalised (given by norm) and weighted (given by fract)
    2. calculate the gradient loss, normalised (given by norm) and weighted (given by fract)
    3. calculate the identity loss, normalised (given by norm) and weighted (given by fract)
    4. calculate the total loss, which is the sum of the absolute, gradient and identity loss
    5. stores the losses in the loss object

    Returns:
    - mse of abs, idn and grd losses
    '''

    abs = ls.abs_loss(n[1:], n_hat)          /loss_obj.norm['abs']* loss_obj.fract['abs']
    grd = ls.grd_loss(n, n_hat)              /loss_obj.norm['grd']* loss_obj.fract['grd']
    idn = ls.idn_loss(n[:-1], p, model)      /loss_obj.norm['idn']* loss_obj.fract['idn']

    loss = abs.mean() + idn.mean() + grd.mean()

    loss_obj.adjust_loss('tot', loss.item())
    loss_obj.adjust_loss('abs', abs.mean().item())
    loss_obj.adjust_loss('grd', grd.mean().item())
    loss_obj.adjust_loss('idn', idn.mean().item())


    return loss



def run_epoch(data_loader, model, loss_obj, DEVICE, optimizer, training, nb_evol = 0):
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
        loss = calc_losses(n, n_hat, z_hat, p, model, loss_obj)
        status += modstatus.sum().item()

        if training == True:
            ## Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return i+1, status




# def validate_one_epoch(test_loader, model, loss_obj, DEVICE):
#     '''
#     Function to validate 1 epoch.

#     Input:
#     - test_loader   = data, torchtensor
#     - model         = ML architecture to be trained
#     - loss_obj      = object that stores the losses

#     Method:
#     1. initialise loss for this epoch
#     2. get data
#     3. pass it through the model, x_hat = result
#     4. calculate loss (difference between x & x_hat), according to loss function defined in loss_function()

#     Returns 
#     - total number of samples
#     - status of the solver
#     '''

#     loss_obj.init_loss()

#     status = 0

#     with torch.no_grad():
#         for i, (n,p,dt) in enumerate(test_loader):
#             # print('\tbatch',i+1,'/',len(test_loader),end="\r")

#             n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
#             p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
#             dt = dt.view(dt.shape[1]).to(DEVICE)
    
#             n_hat, z_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 

#             ## Calculate losses
#             loss = calc_losses(n, n_hat, z_hat,p, model, loss_obj)
#             status += modstatus.sum().item()

    
#     return i+1, status

  


# def train(model, lr, data_loader, test_loader, path, end_epochs, DEVICE, trainloss, testloss, start_epochs = 0, plot = False, log = True, show = True, start_time = 0.):
#     '''
#     Train the model for a number of epochs in the local way (for details; see paper).
    
#     Input:
#         - model         = ML architecture to be trained    
#         - lr            = learning rate
#         - data_loader   = training data, torch tensor
#         - test_loader   = validation data, torch tensor
#         - path          = path to save the model
#         - end_epochs    = number of epochs to train
#         - DEVICE        = device to train on (CPU or GPU)
#         - trainloss     = object that stores the losses of the training
#         - testloss      = object that stores the losses of the validation
#         - start_epochs  = epoch to start from
#         - plot          = plot the losses (boolean)
#         - log           = use logscale in plots (boolean)
#         - show          = show the plots (boolean)
#         - start_time    = time to start from (default = 0)

#     Process:
#         1. initialise the optimizer --> Adam optimiser 
#         Per epoch:
#         2. train the model
#         3. validate the model
#         4. save the model every 10 epochs
#             - save the losses
#             - save the status
#         5. plot the losses if plot == True

#     Returns:
#         - optimizer
#     '''
#     optimizer = Adam(model.parameters(), lr=lr)

#     ## initialise lists for statistics of training

#     print('Model:         ')
#     print('learning rate: '+str(lr))
#     print('loss type:     '+str(trainloss.type))
#     print('\n>>> Training model...')
#     for epoch in range(start_epochs, end_epochs):

#         ## --- Training ---
        
#         model.train()
#         nb, status = train_one_epoch(data_loader, model, trainloss, DEVICE, optimizer)
#         ## save status
#         model.set_status(status/4, 'train')

#         ## ---- Validating ----

#         model.eval() ## zelfde als torch.no_grad
#         nb, status = validate_one_epoch(test_loader, model, testloss, DEVICE)
#         ## save status
#         model.set_status(status/4, 'test')
        
#         ## --- save model every 10 epochs ---
#         if (start_epochs+epoch)%10 == 0 and path != None:
#             ## nn
#             torch.save(model.state_dict(),path+'/nn/nn'+str(int((epoch)/10))+'.pt')
#             trainpath = path+'/train'
#             testpath  = path+'/test'
#             ## losses
#             trainloss.save(trainpath)
#             testloss.save(testpath)
#             ## plot
#             plotting.plot_loss(trainloss, testloss, log = log, show = show)
#             plt.savefig(path+'/loss.png')
        
#         print("Epoch", epoch + 1, "complete!", "\tAverage loss train: ", trainloss.get_loss('tot')[epoch], "\tAverage loss test: ", testloss.get_loss('tot')[epoch])
#         print("              time [hours]: ", (time()-start_time)/(60*60))
    
#     trainloss.normalise_loss(nb)
#     testloss.normalise_loss(nb)

#     print('\n \tDONE!')

#     if plot == True:
#         print('\n >>> Plotting...')
#         plotting.plot_loss(trainloss, testloss, log = log, show = show)

#     return optimizer






