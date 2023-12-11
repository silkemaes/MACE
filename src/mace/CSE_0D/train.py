from time       import time

import torch
from torch.optim  import Adam

## own scripts
import plotting 
from CSE_0D.loss import loss_function, get_loss, Loss





def train_one_epoch(data_loader, model, loss_obj, DEVICE, optimizer):
    '''
    Function to train 1 epoch.

    - data_loader   = data, torchtensor
    - model         = ML architecture to be trained
    - loss_obj      = object that stores the losses

    Method:
    1. get data
    2. push it through the model, x_hat = result
    3. calculate loss (difference between x & x_hat), according to loss function defined in loss_function()
    4. with optimiser, get the gradients and update weights using back propagation.
    
    Returns 
    - losses
    '''    
    overall_loss = 0
    overall_mse_loss = 0
    overall_rel_loss = 0
    overall_evo_loss = 0
    # idv_mse_loss = torch.zeros(466)
    # idv_rel_loss = torch.zeros(466)
    # idv_evo_loss = torch.zeros(466)
    status = 0

    for i, (n,p,dt) in enumerate(data_loader):

        # print('\tbatch',i+1,'/',len(data_loader),end="\r")
        
        n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
        p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
        dt = dt.view(dt.shape[1]).to(DEVICE)

        n_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 


        # print(modstatus.shape)
        # if modstatus.item() == 4:
        #     status += modstatus.item()

        ## Calculate losses
            ## n[:,1:] = abundances[1:k+1] with k=last-1; In this way we can compare if the predicted abundances are correct
            ## the whole array n in passed along, since this is needed to compute the evo loss
        mse_loss, rel_loss, evo_loss  = loss_function(loss_obj, n, n_hat) 
        ## The total loss depends upon the type of losses, set in the loss_obj
        loss = get_loss(mse_loss, rel_loss, evo_loss, loss_obj.type)

        overall_loss += loss.item()
        overall_mse_loss += mse_loss.mean().item()
        overall_evo_loss += evo_loss.mean().item()
        overall_rel_loss += rel_loss.mean().item()
        # idv_mse_loss += mse_loss[:,-1,:].view(-1)       ## only take the loss on the final abundances (last timestep)
        # idv_rel_loss += rel_loss[:,-1,:].view(-1)
        # idv_evo_loss += evo_loss[:,-1,:].view(-1)


        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

    return (overall_loss)/(i+1), overall_mse_loss/(i+1), overall_rel_loss/(i+1), overall_evo_loss/(i+1), status#, idv_mse_loss/(i+1), idv_rel_loss/(i+1), idv_evo_loss/(i+1), status  ## save losses
             



def validate_one_epoch(test_loader, model, loss_obj, DEVICE):

    overall_loss = 0
    overall_mse_loss = 0
    overall_rel_loss = 0
    overall_evo_loss = 0
    # idv_mse_loss = torch.zeros(466)
    # idv_rel_loss = torch.zeros(466)
    # idv_evo_loss = torch.zeros(466)
    status = 0

    with torch.no_grad():
        for i, (n,p,dt) in enumerate(test_loader):
            # print('\tbatch',i+1,'/',len(test_loader),end="\r")

            n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
            dt = dt.view(dt.shape[1]).to(DEVICE)
    
            n_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 

            # if modstatus is not None and modstatus.item() == 4:
            #     status += modstatus.item()

            ## Calculate losses
                ## n[:,1:] = abundances[1:k+1] with k=last-1; In this way we can compare if the predicted abundances are correct
                ## the whole array n in passed along, since this is needed to compute the evo loss
            mse_loss, rel_loss, evo_loss  = loss_function(loss_obj, n, n_hat) 
            ## The total loss depends upon the type of losses, set in the loss_obj
            loss = get_loss(mse_loss, rel_loss, evo_loss, loss_obj.type)

            overall_loss     += loss.item()
            overall_mse_loss += mse_loss.mean().item()
            overall_rel_loss += rel_loss.mean().item()
            overall_evo_loss += evo_loss.mean().item()
            # idv_mse_loss += mse_loss[:,-1,:].view(-1) 
            # idv_rel_loss += rel_loss[:,-1,:].view(-1) 
            # idv_evo_loss += evo_loss[:,-1,:].view(-1)

        return (overall_loss)/(i+1), overall_mse_loss/(i+1), overall_rel_loss/(i+1),overall_evo_loss/(i+1), status#, idv_mse_loss/(i+1), idv_rel_loss/(i+1), idv_evo_loss/(i+1), status  ## save losseses

  


def train(model, lr, data_loader, test_loader, path, end_epochs, DEVICE, trainloss, testloss, start_epochs = 0, plot = False, log = True, show = True):
    optimizer = Adam(model.parameters(), lr=lr)

    ## initialise lists for statistics of training

    print('Model:         ')
    print('learning rate: '+str(lr))
    print('\n>>> Training model...')
    for epoch in range(start_epochs, end_epochs):

        ## Training
        
        model.train()
        print('')
        train_loss, train_mse_loss, train_rel_loss, train_evo_loss, status = train_one_epoch(data_loader, model, trainloss, DEVICE, optimizer)
        ## save losses
        trainloss.set_tot_loss(train_loss)
        trainloss.set_loss(train_mse_loss,'mse')
        trainloss.set_loss(train_rel_loss,'rel')
        trainloss.set_loss(train_evo_loss, 'evo')
        # trainloss.set_idv_loss(train_idv_mse_loss.detach().cpu().numpy(),'mse')
        # trainloss.set_idv_loss(train_idv_rel_loss.detach().cpu().numpy(),'rel')
        # trainloss.set_idv_loss(train_idv_evo_loss.detach().cpu().numpy(), 'evo')
        ## save status
        model.set_status(status/4, 'train')

        ## Validating

        print('\n')
        model.eval() ## zelfde als torch.no_grad

        test_loss, test_mse_loss, test_rel_loss, test_evo_loss, status = validate_one_epoch(test_loader, model, testloss, DEVICE)
        ## save losses
        testloss.set_tot_loss(test_loss)
        testloss.set_loss(test_mse_loss,'mse')
        testloss.set_loss(test_rel_loss,'rel')
        testloss.set_loss(test_evo_loss, 'evo')
        # testloss.set_idv_loss(test_idv_mse_loss.detach().cpu().numpy(),'mse')
        # testloss.set_idv_loss(test_idv_rel_loss.detach().cpu().numpy(),'rel')
        # testloss.set_idv_loss(test_idv_evo_loss.detach().cpu().numpy(), 'evo')
        ## save status
        model.set_status(status/4, 'test')
        
        ## save model
        if (start_epochs+epoch)%10 == 0 and path != None:
            torch.save(model.state_dict(),path+'/nn/nn'+str(int((epoch)/10))+'.pt')
        
        print("\nEpoch", epoch + 1, "complete!", "\tAverage loss train: ", train_loss, "\tAverage loss test: ", test_loss)
    print('\n \tDONE!')

    if plot == True:
        plotting.plot_loss(trainloss, testloss, log = log, show = show)

    return optimizer




def test(model, input,  loss_obj):

    losses = Loss(None,None)

    mace_time = list()
    overall_loss = 0
    idv_mse_loss = []
    idv_rel_loss = []
    print('>>> Testing model...')

    model.eval()
    n     = input[0]
    p     = input[1]
    dt    = input[2]

    print(n.shape, p.shape,dt.shape)

    # n  = n.view(n.shape[1], n.shape[2])     ## op een niet-CPU berekenen als dat er is op de device
    # p  = p.view(p.shape[1], p.shape[2])
    # dt = dt.view(dt.shape[1])

    
    tic = time()
    n_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 
    toc = time()

    # if status.item() == 4:
    #     print('ERROR: neuralODE could not be solved!')
        # break

    ## Calculate losses
    mse_loss, rel_loss, evo_loss  = loss_function(loss_obj,n,n_hat)

    loss = get_loss(mse_loss, rel_loss, evo_loss, loss_obj.type)

    ## overall summed loss of test set
    overall_loss     += loss.item()

    ## individual losses of test set
    losses.set_idv_loss(mse_loss[:,-1,:].view(-1) .detach().cpu().numpy(), 'mse')
    losses.set_idv_loss(rel_loss[:,-1,:].view(-1) .detach().cpu().numpy(), 'rel')
    losses.set_idv_loss(evo_loss[:,-1,:].view(-1) .detach().cpu().numpy(), 'evo')

    losses.set_tot_loss(overall_loss)
    losses.set_loss(mse_loss.mean().item(),'mse')
    losses.set_loss(rel_loss.mean().item(),'rel')
    losses.set_loss(evo_loss.mean().item(),'evo')

    solve_time = toc-tic
    mace_time.append(solve_time)


    print('\nTest loss       :',(overall_loss))
    print('\nSolving time [s]:', solve_time)

    return n, n_hat[0].detach().numpy(), dt, losses, mace_time