from time   import time
import matplotlib.pyplot as plt
import numpy as np

import torch

import src.mace.loss as loss


def train(model,
          data_loader, test_loader, 
          end_epochs, 
          trainloss, testloss, 
          start_epochs = 0,                 ## option to restart training from a certain epoch
          plot = False, log = True, show = True, save_epoch = 10,
          start_time = 0.):
    '''
    Train the model for a number of epochs in the local way (for details; see paper).
    
    Input:
        - model         = ML architecture to be trained    
        - data_loader   = training data, torch tensor
        - test_loader   = validation data, torch tensor
        - end_epochs    = total number of epochs to train
        - trainloss     = object that stores the losses of the training
        - testloss      = object that stores the losses of the validation
        - start_epochs  = epoch to start from (default = 0)
        - plot          = plot the losses (boolean)
        - log           = use logscale in plots (boolean)
        - show          = show the plots (boolean)
        - save_epoch    = save the model every "save_epoch" epochs (default = 10)
        - start_time    = time to start from (default = 0)

    Process:
        1. initialise the optimiser --> Adam optimiser 
        Per epoch:
        2. train the model
        3. validate the model
        4. save the model every 10 epochs
            - save the losses
            - save the status
        5. plot the losses if plot == True

    Returns:
        - optimiser
    '''
    model.set_optimiser()
    path = model.path
    
    ## initialise lists for statistics of training

    print('Model:         ')
    print('--------------')
    print('     # epochs: '+str(end_epochs))
    print('learning rate: '+str(model.lr))
    print('    loss type: '+str(trainloss.losstype))

    if model.scheme == 'loc':
        print('\nLocal training scheme in use.')
        from src.mace.local import run_epoch
    elif model.scheme == 'int':
        print('\nIntegrated training scheme in use.')
        from src.mace.integrated import run_epoch
    else:
        print('\nInvalid training scheme input. Please choose either "loc" or "int".')
        

    print('\n>>> Training model...')


    for epoch in range(start_epochs, end_epochs):

        ## --- Training ---
        
        model.train()
        nb, status = run_epoch(data_loader, model, trainloss, training=True)
        
        ## save status
        model.set_status(status/4, 'train')

        ## ---- Validating ----

        model.eval() ## zelfde als torch.no_grad
        nb, status = run_epoch(test_loader, model, testloss, training=False)
        
        ## save status
        model.set_status(status/4, 'test')
        
        ## --- save model every "save_epoch" epochs ---
        if (start_epochs+epoch)%save_epoch == 0 and path != None:
            ## nn
            torch.save(model.state_dict(),path+'/nn/nn'+str(int((epoch)/save_epoch))+'.pt')
            trainpath = path+'/train'
            testpath  = path+'/test'
            ## losses
            trainloss.save(trainpath)
            testloss.save(testpath)
            ## plot, every save this fig is updated
            loss.plot(trainloss, testloss, log = log, show = show)
            plt.savefig(path+'/loss.png')
        
        print("Epoch", epoch + 1, "complete!", "\tAverage loss train: ", np.round(trainloss.get_loss('tot')[epoch], 5), "\tAverage loss test: ", np.round(testloss.get_loss('tot')[epoch],5))
        
        calc_time = (time()-start_time)     ## in seconds
        if calc_time < 60:
            print("              time [secs]: ", np.round(calc_time,5))
        elif calc_time >= 60:
            print("              time [mins]: ", np.round(calc_time/60,5))
        elif calc_time > 3600:
            print("              time [hours]: ", np.round((time()-start_time)/(60*60),5))
    
    trainloss.normalise_loss(nb)
    testloss.normalise_loss(nb)

    print('\n \tDONE!')

    if plot == True:
        print('\n >>> Plotting...')
        loss.plot(trainloss, testloss, ylim = False, log = log, show = show)

    return 