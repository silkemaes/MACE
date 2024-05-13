from time   import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.optim  import Adam

## own scripts
# import CSE_0D.plotting       as plotting 
import loss


def train(model, lr, data_loader, test_loader, path, end_epochs, DEVICE, trainloss, testloss, start_epochs = 0, plot = False, log = True, show = True, start_time = 0.):
    '''
    Train the model for a number of epochs in the local way (for details; see paper).
    
    Input:
        - model         = ML architecture to be trained    
        - lr            = learning rate
        - data_loader   = training data, torch tensor
        - test_loader   = validation data, torch tensor
        - nb_evol       = number of evolution steps
            == 0: local training
             > 0: integrated training
        - path          = path to save the model
        - end_epochs    = number of epochs to train
        - DEVICE        = device to train on (CPU or GPU)
        - trainloss     = object that stores the losses of the training
        - testloss      = object that stores the losses of the validation
        - start_epochs  = epoch to start from
        - plot          = plot the losses (boolean)
        - log           = use logscale in plots (boolean)
        - show          = show the plots (boolean)
        - start_time    = time to start from (default = 0)

    Process:
        1. initialise the optimizer --> Adam optimiser 
        Per epoch:
        2. train the model
        3. validate the model
        4. save the model every 10 epochs
            - save the losses
            - save the status
        5. plot the losses if plot == True

    Returns:
        - optimizer
    '''
    optimizer = Adam(model.parameters(), lr=lr)

    ## initialise lists for statistics of training

    print('Model:         ')
    print('learning rate: '+str(lr))
    print('loss type:     '+str(trainloss.type))

    if model.scheme == 'loc':
        print('\nLocal training scheme in use.')
        from local import run_epoch
    elif model.scheme == 'int':
        print('\nIntegrated training scheme in use.')
        from integrated import run_epoch
    else:
        print('\nInvalid training scheme input. Please choose either "loc" or "int".')
        

    print('\n>>> Training model...')


    for epoch in range(start_epochs, end_epochs):

        ## --- Training ---
        
        model.train()
        nb, status = run_epoch(data_loader, model, trainloss, DEVICE, optimizer, training=True)
        
        ## save status
        model.set_status(status/4, 'train')

        ## ---- Validating ----

        model.eval() ## zelfde als torch.no_grad
        nb, status = run_epoch(test_loader, model, testloss, DEVICE, optimizer, training=False)
        
        ## save status
        model.set_status(status/4, 'test')
        
        ## --- save model every 10 epochs ---
        if (start_epochs+epoch)%10 == 0 and path != None:
            ## nn
            torch.save(model.state_dict(),path+'/nn/nn'+str(int((epoch)/10))+'.pt')
            trainpath = path+'/train'
            testpath  = path+'/test'
            ## losses
            trainloss.save(trainpath)
            testloss.save(testpath)
            ## plot
            loss.plot(trainloss, testloss, log = log, show = show)
            plt.savefig(path+'/loss.png')
        
        print("Epoch", epoch + 1, "complete!", "\tAverage loss train: ", np.round(trainloss.get_loss('tot')[epoch], 5), "\tAverage loss test: ", np.round(testloss.get_loss('tot')[epoch],5))
        print("              time [hours]: ", np.round((time()-start_time)/(60*60),5))
    
    trainloss.normalise_loss(nb)
    testloss.normalise_loss(nb)

    print('\n \tDONE!')

    if plot == True:
        print('\n >>> Plotting...')
        loss.plot(trainloss, testloss, ylim = False, log = log, show = show)

    return optimizer