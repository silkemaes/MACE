from time   import time
from tqdm   import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim  import Adam

## own scripts
import chempy_0D.plotting as plotting 
import CSE_0D.loss as loss_scipt



def evaluate(n,p,dt, model, nb_evol, loss_dict, loss_obj, DEVICE):
    n  = n.view(n.shape[1], n.shape[2]).to(DEVICE)     
    p  = p.view(p.shape[1], p.shape[2]).to(DEVICE) 
    dt = dt.view(dt.shape[1]).to(DEVICE)

    n0  = n[0:-nb_evol]
    p0  = p[0:-nb_evol]
    dt0 = dt[0:-nb_evol]

    nhat_evol = list()
    n_evol    = list()
    
    ## Calculate the first step of the evolution
    n_hat, z_hat, modstatus = model(n0[:-1],p0,dt0)   
    n_hat = n_hat.view(-1, 468)                 ## remove unnecessary dimension (i.e. batch size = 1)
    nhat_evol.append(n_hat)                     ## store the first step of the predicted evolution
    n_evol.append(n[0:-nb_evol+0][:-1])      ## store the first step of the evolution

    ## subsequent steps of the evolution
    for i in range(1,nb_evol):
        n_hat,z_hat, modstatus = model(n_hat,p[i:-nb_evol+i],dt[i:-nb_evol+i])   
        n_hat = n_hat.view(-1, 468) 
        nhat_evol.append(n_hat)
        n_evol.append(n[i:-nb_evol+i][:-1])

    nhat_evol = torch.stack(nhat_evol).permute(1,0,2)
    n_evol = torch.stack(n_evol).permute(1,0,2)

    loss, loss_dict = get_loss(loss_dict,n, n_evol, nhat_evol, p, model, loss_obj)

    return loss, loss_dict


def get_loss(loss_dict,n, n_evol, nhat_evol, p, model, loss_obj):
    mse = loss_scipt.mse_loss(n_evol, nhat_evol)   /loss_obj.norm['mse']* loss_obj.fract['mse']
    idn = loss_scipt.idn_loss(n[:-1], p, model)    /loss_obj.norm['idn']* loss_obj.fract['idn']

    loss = mse.mean() + idn.mean()
    # print(loss)
    loss_dict['tot']  += loss.item()
    loss_dict['mse']  += mse.mean().item()
    loss_dict['idn']  += idn.mean().item()

    return loss,loss_dict

def train_epoch(data_loader, model, nb_evol, loss_obj,DEVICE, optimizer):
    '''
    Function to train 1 epoch.

    - data_loader   = data, torchtensor
    - model         = ML architecture to be trained
    
    Returns 
    - losses
    '''    
        
    loss_dict = dict()
    loss_dict['mse'] = 0
    loss_dict['rel'] = 0
    loss_dict['grd'] = 0
    loss_dict['idn'] = 0
    loss_dict['elm'] = 0
    loss_dict['tot'] = 0

    status = 0

    for i, (n,p,dt) in enumerate(data_loader):
        # print('batch: ', i)

        loss, loss_dict = evaluate(n,p,dt, model, nb_evol, loss_dict, loss_obj, DEVICE)

        ## Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return loss_dict,i+1, status

def validate_epoch(test_loader, model, nb_evol, loss_obj, DEVICE):

    loss_dict = dict()
    loss_dict['mse'] = 0
    loss_dict['rel'] = 0
    loss_dict['grd'] = 0
    loss_dict['idn'] = 0
    loss_dict['elm'] = 0
    loss_dict['tot'] = 0

    status = 0

    with torch.no_grad():
        for i, (n,p,dt) in enumerate(test_loader):
            # print('batch: ', i)
            loss, loss_dict = evaluate(n,p,dt, model, nb_evol, loss_dict, loss_obj, DEVICE)

        return loss_dict,i+1, status



def train(model, lr, data_loader, test_loader,nb_evol, path, end_epochs, DEVICE, trainloss, testloss, start_epochs = 0, start_time = 0.):
    optimizer = Adam(model.parameters(), lr=lr)

    ## initialise lists for statistics of training

    log = True
    show = False
    plot = False

    print('Model:         ')
    print('learning rate: '+str(lr))
    print('loss type:     '+str(trainloss.type))
    print('\n>>> Training model...\n')
    for epoch in range(start_epochs, end_epochs):

        ## Training
        
        model.train()
        trainloss_dict,nb, status = train_epoch(data_loader, model, nb_evol, trainloss, DEVICE, optimizer)
        ## save losses
        trainloss.set_loss_all(trainloss_dict, nb)

        ## Validating
        model.eval() 

        testloss_dict,nb, status = validate_epoch(test_loader, model, nb_evol,testloss, DEVICE)
        ## save losses
        testloss.set_loss_all(testloss_dict,nb)

        
        ## save model every epoch temporarily
        torch.save(model.state_dict(),path+'/nn/nn-1'+'.pt')
        np.save(path+'/nb_epoch.npy',np.array([epoch+1]))
        ## losses
        trainpath = path+'/train'
        testpath  = path+'/test'
        trainloss.save(trainpath)
        testloss.save(testpath)
        ## plot
        # plotting.plot_loss(trainloss, testloss, log = log, show = show)
        # plt.savefig(path+'/loss.png')

        ## save model every 10 epochs

        if (start_epochs+epoch)%10 == 0 and path != None:
            ## nn
            torch.save(model.state_dict(),path+'/nn/nn'+str(int((epoch)/10))+'.pt')
            ## losses
            trainpath = path+'/train'
            testpath  = path+'/test'
            trainloss.save(trainpath)
            testloss.save(testpath)
            ## plot
            plotting.plot_loss(trainloss, testloss, log = log, show = show)
            plt.savefig(path+'/loss.png')
        
        print("Epoch", epoch + 1, "complete!", "\tAverage loss train: ", trainloss_dict['tot'], "\tAverage loss test: ", testloss_dict['tot'])
        print("              time [hours]: ", (time()-start_time)/(60*60))
    print('\n \tDONE!')

    if plot == True:
        plotting.plot_loss(trainloss, testloss, log = log, show = show)

    return optimizer


def test(model, input):


    mace_time = list()
    # print('     >>> Testing model per time step...')

    model.eval()
    n     = input[0]
    p     = input[1]
    dt    = input[2]

    # print(n.shape, p.shape,dt.shape)
    
    tic = time()
    n_hat, z_hat, modstatus = model(n[:-1],p,dt)    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 
    toc = time()

    solve_time = toc-tic
    # mace_time.append(solve_time)

    # print('             Solving time [s]:', solve_time)

    return n.detach().numpy(), n_hat[0].detach().numpy(), dt, solve_time



def test_evolution(model, input, start_idx):

    mace_time = list()
    n_evol = list()
    # print('     >>> Testing evolution model...')


    model.eval()
    n     = input[0][start_idx]
    p     = input[1]
    dt    = input[2]

    tic_tot = time()
    ## first step of the evolution
    tic = time()
    n_hat, z_hat,modstatus = model(n.view(1, -1),p[start_idx].view(1, -1),dt[start_idx].view(-1))    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches 
    toc = time()
    n_evol.append(n_hat.detach().numpy())
    solve_time = toc-tic
    mace_time.append(solve_time)

    
    ## subsequent steps of the evolution
    for i in range(start_idx+1,len(dt)):
        tic = time()
        n_hat,z_hat, modstatus = model(n_hat.view(1, -1),p[i].view(1, -1),dt[i].view(-1))    ## Give to the solver abundances[0:k] with k=last-1, without disturbing the batches
        toc = time()
        n_evol.append(n_hat.detach().numpy())
        solve_time = toc-tic
        mace_time.append(solve_time)
    toc_tot = time()

    print('             Solving time [s]:', np.array(mace_time).sum())
    # print('         Total   time [s]:', toc_tot-tic_tot)

    return np.array(n_evol).reshape(-1,468), np.array(mace_time).sum()