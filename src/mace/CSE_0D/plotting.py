import numpy             as np
import sys
import matplotlib.pyplot as plt
import matplotlib        as mpl
import matplotlib.lines     as mlines
from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})
# mpl.rcParams.update({'font.size': 10})
# plt.rcParams['figure.dpi'] = 150

## own scripts
import utils
sys.path.insert(1, '/STER/silkem/MACE/src/mace')
import loss             as loss_script

specs_dict, idx_specs = utils.get_specs()




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


def plot_abs(n, n_hat,ax1, color, alpha, title = None, j = -1):

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
    plot_abs(n, n_hat, ax1, colors, alpha = 0.5, title = "metric for abs loss", j = j)
    plot_grd(n, n_hat, ax2, colors, alpha = 0.5, title = "metric for gradient loss", j = j)
    if save:
        plt.savefig(plots_path+title+'_comparison.png')

    return

def plot_abs_specs(n, n_hat, ax1,specs, alpha, title = None):
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


def plot_abs_old(r,n, n_hat, plots_path, rho,T,title = '',specs_lg=dict(), specs = [], save = True, step = False):

    a = 0.7
    ms = 1
    lw = 1

    fig, axs = plt.subplots(3,1, gridspec_kw={'height_ratios': [1,4,1.5]},figsize=(6, 5))
    ax1 = axs[1]
    ax2 = axs[2]
    ax3 = axs[0]

    ax4 = ax3.twinx()

    # axs = np.array([ax1,ax2,ax3,ax4])

    # ax3.set_title(title) 

    

    if len(n_hat) == 0:
        n_hat = n

    ## ------------------- plot abundance profile -------------------
    ## plot individual species
    if len(specs) != 0:
        for spec in specs:
            idx = specs_dict[spec]
            if step == True:
                line, = ax1.plot(r,n_hat[:,idx], ls ='none',  marker = 'o', label = specs_lg[spec], ms = ms,  lw = lw)
            else:
                line, = ax1.plot(r,n_hat[:,idx], ls ='-', label = specs_lg[spec], ms = ms,  lw = lw)
            
            ax1.plot(r,n[:,idx], '--',  lw = lw, color = line.get_color(), alpha = a)
            ## relative error
            # abs = loss_script.abs_loss(n[1:], n_hat)
            # ax2.plot(r[1:], abs[:,idx], '-', label = spec, ms = ms, lw = lw, color = line.get_color())
            # ax2.plot(r[1:],np.abs((n[1:]-n_hat)[:,idx]/n[1:][:,idx]), '-', label = spec, ms = ms, lw = lw, color = line.get_color())
            ax2.plot(r,((np.log10(n[:])-np.log10(n_hat))[:,idx]/np.log10(n[:][:,idx])), '-', label = specs_lg[spec], ms = ms, lw = lw, color = line.get_color())
            # ax2.plot(r[1:],np.abs((n[1:]-n_hat)[:,idx]), '-', label = spec, ms = ms, lw = lw, color = line.get_color())
            # ax2.plot(r[1:],((n[1:]-n_hat)[:,idx]/n[1:][:,idx]), '-', label = spec, ms = ms, lw = lw, color = line.get_color())
            ax1.legend(fontsize = 12,loc = 'lower left')
    ## plot all species
    else:
        for i in range(n_hat.shape[1]):
            line, = ax1.plot(n_hat[:,i], '-',  ms = ms,  lw = lw)
            ax1.plot(r[1:],n[1:,i], '--',  lw = lw, color = line.get_color())
            ## relative error
            ax2.plot(r[1:],np.abs((n[1:]-n_hat)[:,i]/n[1:][:,i]), '-', ms = ms, lw = lw, color = line.get_color())
       
    ax2.plot([1e14,1e18],[0,0], '--k', lw = 0.5)
    ax3.plot(r,rho, 'k-.', lw =lw)
    tempc = 'darkgrey'
    ax4.plot(r, T, ls='-', c=tempc, lw=lw)


    ## ------------------- settings -------------------

    ax1.xaxis.set_ticklabels([])

    fs = 16

    

    ax1.set_ylabel('abundance relative to H$_2$', fontsize = fs) 
    ax2.set_ylabel('error', fontsize = fs)
    ax2.set_xlabel('radius [cm]', fontsize = fs)
    ax3.set_ylabel('$\\rho$ [cm$^{-3}$]', fontsize = fs)
    ax4.set_ylabel('$T$ [K]', color = tempc, fontsize = fs)

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([1e14, 1e18])
        ax.tick_params(labelsize = 14)
    ax4.set_xscale('log')
    ax3.set_yscale('linear')

    ax2.set_yscale('linear')

    for ax in [ax1,ax3,ax4]:
        ax.set_xticklabels([])

    ax1.grid(True, linestyle = '--', linewidth = 0.2)
    ax2.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.set_ylim([1e-20, 1e-2])
    # ax2.set_ylim([-2,2])
         

    plt.subplots_adjust(hspace = 0.00001)

    plt.tight_layout()

    if save == True:
        if len(specs) != 0:
            plt.savefig(plots_path+title+'_abs_specs.png', dpi=300)
        else:
            plt.savefig(plots_path+title+'_abs.png', dpi=300)

    return 

def plot_abs(model1D, n, n_hat, specs, step = False):
    '''
    Function to plot the 1D abundance profiles of a model (middle panel, ax1), 
    together with the density (rho) and temperature (T) profile (upper panel, ax3 & ax4, respectively).
    Also the error between the real and predicted abundances is plotted (lower panel, ax2).
        The error is defined as 
            error = ( log10(n) - log10(n_hat) ) / log10(n) in an element-wise way. 
        See Maes et al. (2024), Eq. (23) for more information.

    The real model will be plotted in dashed lines, 
    the predicted step model by MACE in dotted lines,
    the predicted evolution model by MACE in solid lines.

    Input:
        - model1D: 1D model
        - n: real abundances
        - n_hat: predicted abundances
        - specs: list of species to plot
        - step: boolean that indicated which type of MACE prediction is plotted.
            - False (default) = evolution
            - True = step
    '''

    r = model1D.radius
    rho = model1D.get_dens()
    T = model1D.get_temp()

    a = 0.7
    ms = 1
    lw = 1

    fig, axs = plt.subplots(3,1, gridspec_kw={'height_ratios': [1,4,1.5]},figsize=(6, 5))
    ax1 = axs[1]
    ax2 = axs[2]
    ax3 = axs[0]

    ax4 = ax3.twinx()

    if len(n_hat) == 0:
        n_hat = n

    ## ------------------- plot abundance profile -------------------

    for spec in specs:
        idx = specs_dict[spec]
        if step == True:
            ls = 'none'
            marker = 'o'
        else:
            ls = '-'
            marker = 'none'

        line, = ax1.plot(r,n_hat[:,idx], ls =ls, marker = marker, label = spec, ms = ms,  lw = lw)
        
        ax1.plot(r,n[:,idx], '--',  lw = lw, color = line.get_color(), alpha = a)
        ## relative error
        ax2.plot(r,((np.log10(n[:])-np.log10(n_hat))[:,idx]/np.log10(n[:][:,idx])), '-', label = spec, ms = ms, lw = lw, color = line.get_color())
       
    ax2.plot([1e14,1e18],[0,0], '--k', lw = 0.5)
    ax3.plot(r,rho, 'k-.', lw =lw)
    tempc = 'darkgrey'
    ax4.plot(r, T, ls='-', c=tempc, lw=lw)

    ## ------------------- settings -------------------

    ax1.xaxis.set_ticklabels([])

    fs = 14

    ax1.set_ylabel('abundance relative to H$_2$', fontsize = fs) 
    ax2.set_ylabel('error', fontsize = fs)
    ax2.set_xlabel('radius [cm]', fontsize = fs)
    ax3.set_ylabel('$\\rho$ [cm$^{-3}$]', fontsize = fs)
    ax4.set_ylabel('$T$ [K]', color = tempc, fontsize = fs)

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([1e14, 1e18])
        ax.tick_params(labelsize = 14)
    ax4.set_xscale('log')
    ax3.set_yscale('linear')

    ax2.set_yscale('linear')

    for ax in [ax1,ax3,ax4]:
        ax.set_xticklabels([])

    ax1.grid(True, linestyle = '--', linewidth = 0.2)
    ax2.grid(True, linestyle = '--', linewidth = 0.2)

    ax1.set_ylim([1e-20, 1e-2])
    # ax2.set_ylim([-2,2])

    ax1.legend(fontsize = 12,loc = 'lower left')

    plt.subplots_adjust(hspace = 0.00001)

    plt.tight_layout()


    return fig




