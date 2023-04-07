import numpy             as np
import matplotlib.pyplot as plt
import matplotlib        as mpl




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

def plot_loss(train, test, log = True):
    fig = plt.figure(figsize = (10,4))
    ax1 = fig.add_subplot((111))

    ax1.plot(train, ls = '-'    , c='k', lw = 1)
    ax1.plot(train, ls = None, marker = '.', c='royalblue', label = 'train')

    ax1.plot(test, ls = '-'    , c='k', lw = 1)
    ax1.plot(test, ls = None, marker = '.', c='firebrick', label = 'test')

    if log == True:
        ax1.set_yscale('log')

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('MSE')

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend(loc = 'upper right')

    plt.show()

    return

def plot_compare(real,pred, molecs, specs = None, scale = 'norm'):

    fig = plt.figure(figsize = (5,5))
    ax1 = fig.add_subplot((111))

    if specs == None:
        for spec in molecs:
            ax1.scatter(real[:,molecs[spec]] ,pred[:,molecs[spec]]  , marker = '.', label = spec, alpha = 0.6)
    else:
        for spec in specs:
            ax1.scatter(real[:,molecs[spec]] ,pred[:,molecs[spec]]  , marker = '.', label = spec, alpha = 0.6)

    if scale == 'norm':
        line = [-3,2]
    if scale == 'minmax':
        line = [0,1]
        
    ax1.plot(line,line, '--k', lw = 1)

    ax1.set_xlabel('real')
    ax1.set_ylabel('predicted')

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend()

    plt.show()

def plot_fracs_profile(rad, real, pred, molecs, spec, lw = 1):
    fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [5,2]},figsize=(5,5))
    ## first row
    ax1 = ax[0]
    ax2 = ax[1]
    axs = [ax1,ax2]

    ax1.set_title(spec)

    idx = molecs[spec]

    ax1.plot(rad,real[:,idx], label = 'real'     , lw = lw, c = 'royalblue')
    ax1.plot(rad,pred[:,idx], label = 'predicted by autoencoder', lw = lw, c = 'firebrick')
    ax2.plot(rad,np.abs(real[:,idx]-pred[:,idx]),'--k', lw=lw)
    for ax in axs:
        ax.set_xscale('log')
        ax.grid(True, linestyle = '--', linewidth = 0.2)
        ax.set_yscale('log')
    ax1.set(xticklabels=[])

    ax2.set_xlabel('Radius (cm)')
    ax1.set_ylabel('Fractional abundance w.r.t. H')
    ax2.set_ylabel('Residual')

    ax1.legend(loc = 'lower left')

    fig.tight_layout()
    fig.subplots_adjust(hspace = 0.07)

    plt.show()
    return
