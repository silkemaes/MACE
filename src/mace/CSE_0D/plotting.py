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

specs_dict, idx_specs = utils.get_specs()



def plot_loss(train, test, log = True, ylim = False, limits = None, show = False):

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
    c_mse = 'deepskyblue'
    if 'mse' in train.type:
        ax1.plot(test.get_loss('mse'), ls = '--', marker = 'x', lw = lw, c=c_mse, alpha = a)
        ax1.plot(train.get_loss('mse'), ls = '-', marker = '.', lw = lw, c=c_mse, alpha = a)
        l_mse   = mlines.Line2D([],[], color = c_mse, ls = '-',label='mse',lw = lw2, alpha = 1)
        handles.append(l_mse)
    
    ## ------------- REL -------------
    c_rel = 'firebrick'
    if 'rel' in train.type:
        ax1.plot(test.get_loss('rel'), ls = '--', marker = 'x', lw = lw, c=c_rel, alpha = a)
        ax1.plot(train.get_loss('rel'), ls = '-', marker = '.', lw = lw, c=c_rel, alpha = a)
        l_rel = mlines.Line2D([],[], color = c_rel, ls = '-', label='rel',lw = lw2, alpha = 1)
        handles.append(l_rel)

    ## ------------- GRD -------------
    c_grd = 'forestgreen'
    if 'grd' in train.type:
        ax1.plot(test.get_loss('grd'), ls = '--', marker = 'x', lw = lw, c=c_grd, alpha = a)
        ax1.plot(train.get_loss('grd'), ls = '-', marker = '.', lw = lw, c=c_grd, alpha = a)
        l_grd = mlines.Line2D([],[], color = c_grd, ls = '-', label='grd',lw = lw2, alpha = 1)
        handles.append(l_grd)

    ## ------------- IDN -------------
    c_idn = 'salmon'
    if 'idn' in train.type:
        ax1.plot(test.get_loss('idn'), ls = '--', marker = 'x', lw = lw, c=c_idn, alpha = a)
        ax1.plot(train.get_loss('idn'), ls = '-', marker = '.', lw = lw, c=c_idn, alpha = a)
        l_idn = mlines.Line2D([],[], color = c_idn, ls = '-', label='idn',lw = lw2, alpha = 1)
        handles.append(l_idn)

    ## ------------- ELM -------------
    c_elm = 'darkorchid'
    if 'elm' in train.type:
        ax1.plot(test.get_loss('elm'), ls = '--', marker = 'x', lw = lw, c=c_elm, alpha = a)
        ax1.plot(train.get_loss('elm'), ls = '-', marker = '.', lw = lw, c=c_elm, alpha = a)
        l_elm = mlines.Line2D([],[], color = c_elm, ls = '-', label='elm',lw = lw2, alpha = 1)
        handles.append(l_elm)



    ## ------------ settings --------------
    if log == True:
        ax1.set_yscale('log') 

    if ylim == True:
        if limits == None:
            ax1.set_ylim([1e-2,1e0])
        else:
            ax1.set_ylim(limits)

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    # ax1.set_xlim([5.5,7.5])

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend(handles=handles,loc = 'center left')
    
    plt.tight_layout()

    if show == True:
        plt.show()


    return fig


def get_grd(n, n_hat):
    n0 = n[:-1]

    Δn = np.abs((n[1:]-n0))
    Δn_hat = np.abs((n_hat-n0))

    # print(Δn[:,0].shape)

    # for i in range(len(Δn_un)):
    #     if Δn_un[i] == 0:
    #         Δn_un[i] = 1e-30

    limits = [5e-32,5e-1]
    x = np.linspace(1e-32,1e-2,100)

    return Δn, Δn_hat, limits, x


def plot_mse(n, n_hat,ax1, color, alpha, title = None, j = -1):

    x = np.linspace(1e-20,1e-1,100)

    if j == -1:
        colors = mpl.cm.brg(np.linspace(0, 1, n.shape[0]-1))
        alpha = 0.2
        step = 5
        ax1.set_title(title)
        for i in range(0,n.shape[0]-1, step):
            ax1.scatter(n[i+1],n_hat[i],marker = 'o', color = colors[i], alpha = alpha, label = i) # type: ignore

    else:
        ax1.scatter(n[j],n_hat[j],marker = 'o', color = color, alpha = alpha, label = title+' - step '+str(j)) # type: ignore

    ax1.plot(x,x, '--k', lw = 0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlabel('real abundance')
    ax1.set_ylabel('predicted abundance')

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    # ax1.legend(fontsize = 10)

    return


def plot_grd(n, n_hat,ax1, colors, alpha, title = None, j = -1):
    Δn, Δnhat, limits,x = get_grd(n, n_hat)

    if j == -1:
        
        alpha = 0.2
        step = 5
        ax1.set_title(title)
        for i in range(0,n.shape[0]-1, step):
            ax1.scatter(Δn[i],Δnhat[i],marker = 'o', color = colors[i], alpha = alpha, label = i) # type: ignore

    else:
        ax1.scatter(Δn[j],Δnhat[j],marker = 'o', color = colors, alpha = alpha, label = title) # type: ignore

    ax1.plot(x,x, '--k', lw = 0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(limits)
    ax1.set_ylim(limits)

    ax1.set_xlabel('real evolution')
    ax1.set_ylabel('predicted evolution')

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend(fontsize = 10)

    return

def plot_compare(n, n_hat, plots_path,  title, alpha = 0.5, j = -1, save=True):
    colors = mpl.cm.brg(np.linspace(0, 1, n.shape[0]-1))
    
    fig, axs = plt.subplots(1,2,figsize=(13,6))
    ax1 = axs[0]
    ax2 = axs[1]

    j=-1
    plot_mse(n, n_hat, ax1, colors, alpha = 0.5, title = "metric for mse loss", j = j)
    plot_grd(n, n_hat, ax2, colors, alpha = 0.5, title = "metric for gradient loss", j = j)
    if save:
        plt.savefig(plots_path+title+'_comparison.png')

    return

def plot_mse_specs(n, n_hat, ax1,specs, alpha, title = None):
    n, n_hat,x = get_abs(n, n_hat)

    ax1.set_title(title)

    i=-1
    for spec in specs:
        idx = specs_dict[spec]
        ax1.scatter(n[0][i][idx],n_hat[0][i][idx],marker = 'o', alpha = alpha, label = spec) # type: ignore

    ax1.plot(x,x, '--k', lw = 0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('real log abundance')
    ax1.set_ylabel('predicted log abundance')
    

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend(fontsize = 8)

    return


def plot_evol_specs(n, n_hat, ax1,specs, alpha, title = None):
    Δn, Δnhat, limits, x = get_grd(n, n_hat)

    for spec in specs:
        idx = specs_dict[spec]
        ax1.scatter(Δn[idx],Δnhat[idx],marker = 'o', alpha = alpha, label = spec) # type: ignore

    ax1.plot(x,x, '--k', lw = 0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlim(limits)
    ax1.set_ylim(limits)

    ax1.set_xlabel('real evolution')
    ax1.set_ylabel('predicted evolution')

    ax1.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.legend(fontsize = 8)

    return


def plot_abs(n, n_hat, plots_path,title = '', specs = [], save = True):

    a = 0.5
    ms = 1.5
    lw = 1

    fig, axs = plt.subplots(2,1, gridspec_kw={'height_ratios': [4,4]},figsize=(10, 8))
    ax1 = axs[0]
    ax2 = axs[1]

    ax1.set_title(title) 

    if len(n_hat) == 0:
        n_hat = n[1:]

    ## ------------------- plot abundance profile -------------------
    ## plot individual species
    if len(specs) != 0:
        for spec in specs:
            idx = specs_dict[spec]
            line, = ax1.plot(n_hat[:,idx], '-', label = spec, ms = ms,  lw = lw)
            ax1.plot(n[1:,idx], '--',  lw = lw, color = line.get_color())
            ## relative error
            ax2.plot(np.abs((n[1:]-n_hat)[:,idx]/n[1:][:,idx]), '-', label = spec, ms = ms, lw = lw, color = line.get_color())
            ax1.legend(fontsize = 6,loc = 'upper right')
    ## plot all species
    else:
        for i in range(n_hat.shape[1]):
            line, = ax1.plot(n_hat[:,i], '-',  ms = ms,  lw = lw)
            ax1.plot(n[1:,i], '--',  lw = lw, color = line.get_color())
            ## relative error
            ax2.plot(np.abs((n[1:]-n_hat)[:,i]/n[1:][:,i]), '-', ms = ms, lw = lw, color = line.get_color())
       
    ax2.plot([0,n_hat.shape[0]],[1,1], '--k', lw = 0.5)


    ## ------------------- settings -------------------

    ax1.xaxis.set_ticklabels([])

    ax1.set_ylabel('abundance') 
    ax2.set_ylabel('relative error')
    ax2.set_xlabel('step')

    ax2.set_yscale('log')
    ax1.set_yscale('log') 
    ax1.grid(True, linestyle = '--', linewidth = 0.2)
    ax2.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.set_ylim([1e-20, 1e1])

    plt.subplots_adjust(hspace = 0.07)

    plt.tight_layout()

    if save == True:
        if len(specs) != 0:
            plt.savefig(plots_path+title+'_abs_specs.png')
        else:
            plt.savefig(plots_path+title+'_abs.png')

    return 




