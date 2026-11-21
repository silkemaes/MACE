'''
This script contains a function to plot the 1D abundance profiles of a model,
together with the predictions of MACE for this model.

'''

import torch
import matplotlib.pyplot as plt
from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})

import src.mace.utils           as utils
import src.mace.dataset         as ds
import src.mace.train           as train
specs_dict = utils.get_specs()


def plot_n(model1D, modeltype, n, n_hat, specs, step=False):
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
    if modeltype == 'Phantom':
        specs_dict = utils.get_specs('Phantom')
    else:
        specs_dict = utils.get_specs()

    r = model1D.radius
    rho = model1D.get_dens()
    T = model1D.get_temp()

    a = 0.7
    ms = 1
    lw = 1

    fig, axs = plt.subplots(3,
                            1,
                            gridspec_kw={'height_ratios': [1, 4, 1.5]},
                            figsize=(6, 5))
    ax1 = axs[1]
    ax2 = axs[2]
    ax3 = axs[0]

    ax4 = ax3.twinx()

    if len(n_hat) == 0:
        n_hat = n

    ## ------------------- plot abundance profile -------------------

    err, err_mean = utils.error(n, n_hat)

    for spec in specs:
        idx = specs_dict[spec]
        if step == True:
            ls = 'none'
            marker = 'o'
        else:
            ls = '-'
            marker = '.'
        ## predicted abundances
        line, = ax1.plot(r,
                         n_hat[:, idx],
                         ls=ls,
                         marker=marker,
                         label=spec,
                         ms=ms,
                         lw=lw)
        ## real abundances
        ax1.plot(r, n[:, idx], '--', lw=lw, color=line.get_color(), alpha=a)
        ## relative error
        ax2.plot(r,
                 err[:, idx],
                 '-',
                 label=spec,
                 ms=ms,
                 lw=lw,
                 color=line.get_color())
    ## indicate where 0 is on the error plot
    ax2.plot([1e14, 1e18], [0, 0], '--k', lw=0.5)

    ## plot the physical parameters (density, rho, and temperature, T)
    ax3.plot(r, rho, 'k-.', lw=lw)
    tempc = 'darkgrey'
    ax4.plot(r, T, ls='-', c=tempc, lw=lw)

    ## ------------------- settings -------------------

    ax1.xaxis.set_ticklabels([])

    fs = 14

    ax1.set_ylabel('abundance relative to H$_2$', fontsize=fs)
    ax2.set_ylabel('error', fontsize=fs)
    ax2.set_xlabel('radius [cm]', fontsize=fs)
    ax3.set_ylabel('$\\rho$ [cm$^{-3}$]', fontsize=fs)
    ax4.set_ylabel('$T$ [K]', color=tempc, fontsize=fs)

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlim([1e14, 1e18])
        ax.tick_params(labelsize=14)
    ax4.set_xscale('log')
    ax3.set_yscale('linear')

    ax2.set_yscale('linear')

    for ax in [ax1, ax3, ax4]:
        ax.set_xticklabels([])

    ax1.grid(True, linestyle='--', linewidth=0.2)
    ax2.grid(True, linestyle='--', linewidth=0.2)

    ax1.set_ylim([1e-20, 1e-2])
    ax2.set_ylim([-2,2])

    ax1.legend(fontsize=10, loc='lower left')

    plt.subplots_adjust(hspace=0.00001)

    plt.tight_layout()

    return fig, [ax3,ax1,ax2]



def plot_hairs(model, meta, test_path, nb_evol:int, 
               specs:list=['CO', 'H2O', 'OH', 'C2H2', 'C2H', 'CH3C5NH+', 'C10H2+'], 
               interval:int=1):
    '''
    ZEvolves model for nb_evol steps on the test data at test_path. 
    Plots this evolution for each datapoint with interval between.
    '''
    model1D, (n, p, dt), info = ds.get_test_data(test_path, meta) #TODO: figure out a better name for model1D
    r  = model1D.radius

    n0, p_seq, dt_seq, _ = train.make_windows(n, p, dt, nb_evol)

    # Make hairs (but better)
    with torch.no_grad():
        hairs, status = model.rollout(n0[::interval], # Slice with step
                                         p_seq[::interval], 
                                         dt_seq[::interval])
        
    hairs = torch.concatenate((n0[::interval,None], hairs), dim=1) # Plop n0 to the front of each hair

    # Plot every species in separate panel
    n_figs = len(specs)
    fig, axes = plt.subplots(n_figs, figsize=(8, 2*n_figs), sharex=True ,squeeze=False)
    for spec, ax in zip(specs, axes.flat):

        idx = specs_dict[spec]
        ax.loglog(r, n[:,idx], label=spec) # Species over r

        for i, hair in enumerate(hairs):
            r_hair = r[i*interval : i*interval+nb_evol+1]
            ax.loglog(r_hair, hair[:,idx], lw=.5) # Hair over subset r

    #fig.legend()
    return fig, axes