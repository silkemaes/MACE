import loss as ls
import torch



def evaluate(n,p,dt, model, nb_evol,  loss_obj, status,  DEVICE):
    '''
    < Core of the integrated training scheme >
    The loop in this function makes that the absolute loss becomes integrated.

    Function to evaluate the model on a given data sample.

    Input:
    - n         = abundances
    - p         = parameters
    - dt        = time steps
    - model     = ML architecture to be trained
    - nb_evol   = number of evolution steps
    - loss_obj  = loss object to store losses of training
    - status    = status of the solver

    Method:
    1. Extract the first "hair" of the evolution, depending on nb_evol (see more in paper)
    2. Calculate the first step of the evolution
    3. Loop through the input data and 
        calculate the subsequent steps of the evolution
        Every step is stored in a list, n as well as n_hat
    4. Transform outcome in such a way that it can be compared with the real data
    5. Calculate the losses
    6. Pass losses and solver status back to training function

    Returns:
    '''

    ## 1.
    n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     
    p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
    dt = dt.view(dt.shape[1]).to(DEVICE)

    n0  = n[0:-nb_evol]
    p0  = p[0:-nb_evol]
    dt0 = dt[0:-nb_evol]

    nhat_evol = list()  ## predicted evolution
    n_evol    = list()  ## list to split up real abundances to compare with predicted ones
    
    ## 2. Calculate the first step of the evolution
    n_hat, z_hat, modstatus = model(n0[:-1],p0,dt0)   
    n_hat = n_hat.view(-1, 468)                 ## remove unnecessary dimension (i.e. batch size = 1)
    nhat_evol.append(n_hat)                     ## store the first step of the predicted evolution
    n_evol.append(n[0:-nb_evol+0][:-1])         ## store the first step of the evolution
    status += modstatus.sum().item()

    ## 3. Subsequent steps of the evolution
    for i in range(1,nb_evol):
        n_hat,z_hat, modstatus = model(n_hat,p[i:-nb_evol+i],dt[i:-nb_evol+i])   
        n_hat = n_hat.view(-1, 468) 
        nhat_evol.append(n_hat)
        n_evol.append(n[i:-nb_evol+i][:-1])
        status += modstatus.sum().item()

    ## 4. 
    nhat_evol = torch.stack(nhat_evol).permute(1,0,2)
    n_evol = torch.stack(n_evol).permute(1,0,2)

    ## 5.
    loss = get_loss(n, n_evol, nhat_evol, p, model, loss_obj)

    ## 6.
    return loss, status


def get_loss(n, n_evol, nhat_evol, p, model, loss_obj):
    '''
    Function to calculate the losses of the model.

    Input:
    - n         = abundances
    - n_evol    = real evolution
    - nhat_evol = predicted evolution
    - p         = physical parameters
    - model     = ML architecture to be trained
    - loss_obj  = loss object to store losses of training

    Returns:
    - mse of the abs and idn losses
    '''
    abs = ls.abs_loss(n_evol, nhat_evol)   /loss_obj.norm['abs']* loss_obj.fract['abs']
    idn = ls.idn_loss(n[:-1], p, model)    /loss_obj.norm['idn']* loss_obj.fract['idn']

    loss = abs.mean() + idn.mean()
    # print(loss)
    loss_obj.adjust_loss('tot', loss.item())
    loss_obj.adjust_loss('abs', abs.mean().item())
    loss_obj.adjust_loss('idn', idn.mean().item())

    return loss

def run_epoch(data_loader, model, loss_obj, DEVICE, optimizer, training, nb_evol):
    '''
    Function to train 1 epoch.

    - data_loader   = data, torchtensor
    - model         = ML architecture to be trained
    - nb_evol       = number of evolution steps
    - loss_obj      = loss object to store losses of training
    - DEVICE        = device to run the model on
    - optimizer     = type of optimizer to update the weights of the model
    - training      = boolean to indicate if the model is training
    - nb_evol       = number of evolution steps

    Returns 
    - number of data samples
    - status of the solver
    '''    
        
    loss_obj.init_loss()

    status = 0

    for i, (n,p,dt) in enumerate(data_loader):

        loss, status = evaluate(n, p, dt, model, nb_evol, loss_obj, status, DEVICE)

        if training == True:
            ## Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return i+1, status

# def validate_epoch(test_loader, model, nb_evol, loss_obj, DEVICE):
#     '''
#     Function to validate the model on the test data.

#     - test_loader   = validation data, torchtensor
#     - model         = ML architecture to be trained
#     - nb_evol       = number of evolution steps
#     - loss_obj      = loss object to store losses of training
#     - DEVICE        = device to run the model on

#     '''

#     loss_obj.init_loss()

#     status = 0

#     with torch.no_grad():
#         for i, (n,p,dt) in enumerate(test_loader):
#             loss = evaluate(n,p,dt, model, nb_evol, loss_obj, status, DEVICE)

#         return i+1, status



# def train(model, lr, data_loader, test_loader, nb_evol, path, end_epochs, DEVICE, trainloss, testloss, start_epochs = 0, start_time = 0., plot = False, log = True, show = True):
#     '''
    
#     '''
#     optimizer = Adam(model.parameters(), lr=lr)

#     ## initialise lists for statistics of training

#     print('Model:         ')
#     print('learning rate: '+str(lr))
#     print('loss type:     '+str(trainloss.type))
#     print('\n>>> Training model...\n')
#     for epoch in range(start_epochs, end_epochs):

#         ## --- Training ---
        
#         model.train()
#         nb, status = train_epoch(data_loader, model, nb_evol, trainloss, DEVICE, optimizer)
#         ## save status
#         model.set_status(status/4, 'train')

#         ## ---- Validating ----
#         model.eval() 
#         nb, status = validate_epoch(test_loader, model, nb_evol,testloss, DEVICE)
#         ## save status
#         model.set_status(status/4, 'test')

        
#         ## --- save model every epoch temporarily ---
#         if path != None:
#             torch.save(model.state_dict(),path+'/nn/nn-1'+'.pt')
#             np.save(path+'/nb_epoch.npy',np.array([epoch+1]))
#             ## losses
#             trainpath = path+'/train'
#             testpath  = path+'/test'
#             trainloss.save(trainpath)
#             testloss.save(testpath)

#         ## --- save model every 10 epochs ---
#         if (start_epochs+epoch)%10 == 0 and path != None:
#             ## nn
#             torch.save(model.state_dict(),path+'/nn/nn'+str(int((epoch)/10))+'.pt')
#             ## losses
#             trainpath = path+'/train'
#             testpath  = path+'/test'
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
#         plotting.plot_loss(trainloss, testloss, log = log, show = show)

#     return optimizer


