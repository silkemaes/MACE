import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import matplotlib.lines     as mlines
from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})
mpl.rcParams.update({'font.size': 10})
plt.rcParams['figure.dpi'] = 150

## own scripts
import utils



def plot_hist(df):

    fig = plt.figure(figsize = (10,4))
    ax1 = fig.add_subplot((111))

    for i in range(df.shape[1]):
        ax1.hist(df[:,i].ravel(), bins = 100, histtype='step')

    # ax1.set_yscale('log')
    ax1.set_xlabel('normalised log abundance')
    ax1.set_ylabel('#')

    plt.show()

    return


def plot_loss(train, test, log = True, ylim = False, show = False):

    fig = plt.figure(figsize = (10,4))
    ax1 = fig.add_subplot((111))

    lw = 1
    a = 0.8
    lw2 = 4
    ## ------------ legend ----------------

    l_train = mlines.Line2D([],[], color = 'grey', ls = '-' , marker = '.', label='train',lw = lw, alpha = 1)
    l_test  = mlines.Line2D([],[], color = 'grey', ls = '--', marker = 'x', label='test' ,lw = lw, alpha = 1)
    l_tot   = mlines.Line2D([],[], color = 'k'   , ls = '-' , label='total',lw = lw, alpha = 1)
    
    handles = [l_train, l_test, l_tot]

    ## ------------- TOTAL ------------
    ax1.plot(test.get_tot_loss(), ls = '--', marker = 'None', lw = lw, c='k')
    ax1.plot(train.get_tot_loss(), ls = '-', marker = 'None', lw = lw, c='k')

    ## ------------- MSE -------------
    if (train.type == 'mse' 
        or train.type == 'mse_evo' or train.type == 'evo_mse' 
        or train.type == 'mse_rel' or train.type == 'rel_mse'
        or train.type == 'mse_rel_evo' or train.type == 'mse_evo_rel' or train.type == 'rel_mse_evo' or train.type == 'rel_evo_mse' or train.type == 'evo_mse_rel' or train.type == 'evo_rel_mse'):

        ax1.plot(test.get_loss('mse'), ls = '--', marker = 'x', lw = lw, c='firebrick', alpha = a)
        ax1.plot(train.get_loss('mse'), ls = '-', marker = '.', lw = lw, c='firebrick', alpha = a)
        l_mse   = mlines.Line2D([],[], color = 'firebrick', ls = '-',label='mse',lw = lw2, alpha = 1)
        handles.append(l_mse)
    
    ## ------------- REL -------------
    if (train.type == 'rel'
        or train.type == 'rel_evo' or train.type == 'evo_rel'
        or train.type == 'rel_mse' or train.type == 'mse_rel'
        or train.type == 'rel_mse_evo' or train.type == 'rel_evo_mse' or train.type == 'mse_rel_evo' or train.type == 'mse_evo_rel' or train.type == 'evo_rel_mse' or train.type == 'evo_mse_rel'):

        ax1.plot(test.get_loss('rel'), ls = '--', marker = 'x', lw = lw, c='royalblue', alpha = a)
        ax1.plot(train.get_loss('rel'), ls = '-', marker = '.', lw = lw, c='royalblue', alpha = a)
        l_rel = mlines.Line2D([],[], color = 'royalblue', ls = '-', label='rel',lw = lw2, alpha = 1)
        handles.append(l_rel)

    ## ------------- EVO -------------
    if (train.type == 'evo'
        or train.type == 'evo_rel' or train.type == 'rel_evo'
        or train.type == 'evo_mse' or train.type == 'mse_evo'
        or train.type == 'evo_mse_rel' or train.type == 'evo_rel_mse' or train.type == 'mse_evo_rel' or train.type == 'mse_rel_evo' or train.type == 'rel_evo_mse' or train.type == 'rel_mse_evo'):

        ax1.plot(test.get_loss('evo'), ls = '--', marker = 'x', lw = lw, c='goldenrod', alpha = a)
        ax1.plot(train.get_loss('evo'), ls = '-', marker = '.', lw = lw, c='goldenrod', alpha = a)
        l_evo = mlines.Line2D([],[], color = 'goldenrod', ls = '-', label='evo',lw = lw2, alpha = 1)
        handles.append(l_evo)

    ## ------------ settings --------------
    if log == True:
        ax1.set_yscale('log') 

    if ylim == True:
        ax1.set_ylim([1e-2,1e0])

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend(handles=handles,loc = 'upper right')
    
    plt.tight_layout()

    if show == True:
        plt.show()


    return fig







def plot_compare(real, preds, models, molecs, spec, scale = 'norm'):

    colors = mpl.cm.Set3(np.linspace(0, 1, len(models)))

    fig = plt.figure(figsize = (3,3))
    ax1 = fig.add_subplot((111))

    ax1.set_title(spec, fontsize = 7)

    for i, pred in enumerate(preds): 
        ax1.scatter(real[:,molecs[spec]] ,pred[:,molecs[spec]], marker = '.', label = models[i].name, alpha = 0.6, color = colors[i])

    if scale == 'norm':
        line = [-3,2]
    if scale == 'minmax':
        line = [0,1]
        
    ax1.plot(line,line, '--k', lw = 0.5)

    ax1.set_xlabel('real')
    ax1.set_ylabel('predicted')

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend(fontsize = 5)

    plt.show()


'''
preds = list(predictions)
models = list(models)
'''
def plot_fracs_profile(rad, real, preds, models, molecs, spec, lw = 1):
        
    colors = mpl.cm.Set3(np.linspace(0, 1, len(models)))
  
    
    fig, ax = plt.subplots(3,1, gridspec_kw={'height_ratios': [5,2,2]},figsize=(5,6))
    ## first row
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    axs = [ax1,ax2, ax3]

    ax1.set_title(spec, fontsize = 7)

    idx = molecs[spec]

    ax1.plot(rad,real[:,idx], label = 'real' , lw = lw, c = 'k')
    for i, pred in enumerate(preds): 
        ax1.plot(rad,pred[:,idx], label = models[i].name  , lw = lw, c = colors[i])
        ax2.plot(rad,np.abs(real[:,idx]-pred[:,idx])/max(real[:,idx])      , lw = lw, c = colors[i],ls = '--')
        ## absolute residuals
        res = utils.get_absolute_residuals(real, preds[i])
        ax3.plot(rad, res, lw = lw, c = colors[i],ls = '--')
   
    for ax in axs:
        ax.set_xscale('log')
        ax.grid(True, linestyle = '--', linewidth = 0.2)
        ax.set_yscale('log')
    ax1.set(xticklabels=[])
    ax2.set(xticklabels=[])

    ax1.set_ylim([1e-12,3e-3])
    ax2.set_ylim([1e-7,1e1])
    ax3.set_xlabel('Radius (cm)')
    ax1.set_ylabel('Fractional abundance w.r.t. H')
    ax2.set_ylabel('Relative residuals')
    ax3.set_ylabel('Total residual')

    ax1.legend(loc = 'upper right', fontsize = 5)

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.07)

    return

'''
preds = dict()
'''
def plot_fracs_profile_lr(rad, real, preds, molecs, spec, lw = 0.8):
    
    colors = mpl.cm.viridis(np.linspace(0, 1, len(preds)))
      
    fig, ax = plt.subplots(3,1, gridspec_kw={'height_ratios': [5,2,2]},figsize=(5,6))
    ## first row
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    axs = [ax1,ax2, ax3]

    ax1.set_title(spec, fontsize = 7)

    idx = molecs[spec]
    alpha = 0.8

    ax1.plot(rad,real[:,idx], label = 'real' , lw = 1.1, c = 'k')
    for i,lr in enumerate(preds): 
        ax1.plot(rad,preds[lr][:,idx], label = 'lr = '+str(lr) , lw = lw, c = colors[i], alpha = alpha)
        ax2.plot(rad,np.abs(real[:,idx]-preds[lr][:,idx])/max(real[:,idx])      , lw = lw, c = colors[i], alpha = alpha,ls = '--')
        ## absolute residuals
        res = utils.get_absolute_residuals(real, preds[lr])
        ax3.plot(rad, res, lw = lw, c = colors[i], alpha = alpha,ls = '--')
   
    for ax in axs:
        ax.set_xscale('log')
        ax.grid(True, linestyle = '--', linewidth = 0.2)
        ax.set_yscale('log')
    ax1.set(xticklabels=[])
    ax2.set(xticklabels=[])

    ax1.set_ylim([1e-12,3e-3])
    ax2.set_ylim([1e-7,1e1])
    ax3.set_xlabel('Radius (cm)')
    ax1.set_ylabel('Fractional abundance w.r.t. H')
    ax2.set_ylabel('Relative residuals')
    ax3.set_ylabel('Total residual')

    ax1.legend(loc = 'upper right', fontsize = 5)

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.07)

    plt.show()
    return