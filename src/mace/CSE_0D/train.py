from time       import time
from tqdm   import tqdm
import numpy as np

import torch
from torch.optim  import Adam

## own scripts
import chempy_0D.plotting as plotting 
from CSE_0D.loss import loss_function, get_loss, Loss


def process_loss_one_epoch(loss_dict, n, n_hat, z_hat, p, model, loss_obj):
    ## Calculate losses
            ## n[:,1:] = abundances[1:k+1] with k=last-1; In this way we can compare if the predicted abundances are correct
            ## the whole array n in passed along, since this is needed to compute the evo loss
    mse_loss, rel_loss, evo_loss, idn_loss, elm_loss  = loss_function(loss_obj, model, n, n_hat,z_hat, p) 
    ## The total loss depends upon the type of losses, set in the loss_obj
    loss = get_loss(mse_loss, rel_loss, evo_loss,idn_loss,elm_loss, loss_obj.type)

    loss_dict['tot']  += loss.item()
    loss_dict['mse']  += mse_loss.mean().item()
    loss_dict['rel']  += rel_loss.mean().item()
    loss_dict['evo']  += evo_loss.mean().item()
    loss_dict['idn']  += idn_loss.mean().item()
    loss_dict['elm']  += elm_loss.mean().item()

    return loss,loss_dict

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
        
    loss_dict = dict()
    loss_dict['mse'] = 0
    loss_dict['rel'] = 0
    loss_dict['evo'] = 0
    loss_dict['idn'] = 0
    loss_dict['elm'] = 0
    # loss_dict['grd'] = 0
    loss_dict['tot'] = 0

    status = 0

    for i, (n,p,dt) in enumerate(data_loader):

        # print('\tbatch',i+1,'/',len(data_loader),end="\r")
        
        n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
        p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
        dt = dt.view(dt.shape[1]).to(DEVICE)

        n_hat, z_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 


        # print(modstatus.shape)
        # if modstatus.item() == 4:
        #     status += modstatus.item()

        loss,loss_dict = process_loss_one_epoch(loss_dict, n, n_hat, z_hat, p, model, loss_obj)

        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    

    return loss_dict,i+1, status



def validate_one_epoch(test_loader, model, loss_obj, DEVICE):

    loss_dict = dict()
    loss_dict['mse'] = 0
    loss_dict['rel'] = 0
    loss_dict['evo'] = 0
    loss_dict['idn'] = 0
    loss_dict['elm'] = 0
    loss_dict['tot'] = 0

    status = 0

    with torch.no_grad():
        for i, (n,p,dt) in enumerate(test_loader):
            # print('\tbatch',i+1,'/',len(test_loader),end="\r")

            n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     ## op een niet-CPU berekenen als dat er is op de device
            p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
            dt = dt.view(dt.shape[1]).to(DEVICE)
    
            n_hat, z_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 

            # if modstatus is not None and modstatus.item() == 4:
            #     status += modstatus.item()
            
            loss,loss_dict = process_loss_one_epoch(loss_dict, n, n_hat, z_hat,p, model, loss_obj)


        return loss_dict,i+1, status

  


def train(model, lr, data_loader, test_loader, path, end_epochs, DEVICE, trainloss, testloss, start_epochs = 0, plot = False, log = True, show = True):
    optimizer = Adam(model.parameters(), lr=lr)

    ## initialise lists for statistics of training

    print('Model:         ')
    print('learning rate: '+str(lr))
    print('loss type:     '+str(trainloss.type))
    print('\n>>> Training model...')
    for epoch in range(start_epochs, end_epochs):

        ## Training
        
        model.train()
        print('')
        trainloss_dict,nb, status = train_one_epoch(data_loader, model, trainloss, DEVICE, optimizer)
        ## save losses
        trainloss.set_loss_all(trainloss_dict, nb)

        ## save status
        model.set_status(status/4, 'train')

        ## Validating

        print('\n')
        model.eval() ## zelfde als torch.no_grad

        testloss_dict,nb, status = validate_one_epoch(test_loader, model, testloss, DEVICE)
        ## save losses
        testloss.set_loss_all(testloss_dict,nb)
        ## save status
        model.set_status(status/4, 'test')
        
        ## save model
        if (start_epochs+epoch)%10 == 0 and path != None:
            torch.save(model.state_dict(),path+'/nn/nn'+str(int((epoch)/10))+'.pt')
        
        print("Epoch", epoch + 1, "complete!", "\tAverage loss train: ", trainloss_dict['tot'], "\tAverage loss test: ", testloss_dict['tot'])
    print('\n \tDONE!')

    if plot == True:
        plotting.plot_loss(trainloss, testloss, log = log, show = show)

    return optimizer




def test(model, input,  loss_obj):

    losses = Loss(None,None)

    mace_time = list()
    loss_dict = dict()
    loss_dict['mse'] = 0
    loss_dict['rel'] = 0
    loss_dict['evo'] = 0
    loss_dict['idn'] = 0
    loss_dict['elm'] = 0
    loss_dict['tot'] = 0
    print('>>> Testing model...')

    model.eval()
    n     = input[0]
    p     = input[1]
    dt    = input[2]

    # print(n.shape, p.shape,dt.shape)
    
    tic = time()
    n_hat,z_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 
    toc = time()

    loss,loss_dict = process_loss_one_epoch(loss_dict, n, n_hat, z_hat,p, model, loss_obj)
    losses.set_loss_all(loss_dict, 1)

    solve_time = toc-tic
    mace_time.append(solve_time)

    print('\nTest loss     :',loss.item())
    print('Solving time [s]:', solve_time)

    return n.detach().numpy(), n_hat[0].detach().numpy(), dt, losses, mace_time

def test_evolution(model, input, start_idx):

    losses = Loss(None,None)

    mace_time = list()
    n_evo = list()

    print('>>> Testing model...')

    model.eval()
    n     = input[0][start_idx]
    p     = input[1]
    dt    = input[2]

    ## first step of the evolution
    tic = time()
    n_hat, z_hat,modstatus = model(n.view(1, -1),p[start_idx].view(1, -1),dt[start_idx].view(-1))    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 
    toc = time()
    n_evo.append(n_hat.detach().numpy())
    solve_time = toc-tic
    mace_time.append(solve_time)

    ## subsequent steps of the evolution
    for i in tqdm(range(start_idx+1,len(dt))):
        n_hat,z_hat, modstatus = model(n_hat.view(1, -1),p[i].view(1, -1),dt[i].view(-1))    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches
        n_evo.append(n_hat.detach().numpy())
        solve_time = toc-tic
        mace_time.append(solve_time)


    return np.array(n_evo).reshape(-1,468), mace_time