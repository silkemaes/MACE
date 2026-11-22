'''
This script contains a function to plot the 1D abundance profiles of a model,
together with the predictions of MACE for this model.

Version 2.1

#TODO: Make hairplot and identityplot return scaled values 
'''

import torch
import matplotlib.pyplot as plt
from matplotlib          import rcParams
rcParams.update({'figure.dpi': 200})

import src.mace.utils           as utils
import src.mace.dataset         as ds
import src.mace.train           as train
from src.mace.mace import MACE
from src.mace.loss import Loss
from src.mace.meta import Meta
from pathlib import Path
import os


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



def plot_hairs(model:MACE, meta:Meta, test_path:str|Path, nb_evol:int, 
               specs:list[str]=['CO', 'H2O', 'OH', 'C2H2', 'C2H', 'CH3C5NH+', 'C10H2+'], 
               interval:int=1):
    '''
    ZEvolves model for nb_evol steps on the test data at test_path. 

    Plots this evolution for each datapoint with interval between.
    '''
    specs_dict = utils.get_specs(meta.data_type)

    model1D, (n, p, dt), info = ds.get_test_data(test_path, meta) #TODO: figure out a better name for model1D
    r  = model1D.radius

    n0, p_seq, dt_seq, _ = train.make_windows(n, p, dt, nb_evol)

    # Make hairs
    with torch.no_grad():
        hairs, status = model.rollout(n0[::interval], # Slice with step
                                         p_seq[::interval], 
                                         dt_seq[::interval])
        
    hairs = torch.concatenate((n0[::interval,None], hairs), dim=1) # Plop n0 to the front of each hair
    n = ds.get_n(n)          # denormalise
    hairs  = ds.get_n(hairs)

    # Plot every species in separate panel
    n_figs = len(specs)
    fig, axes = plt.subplots(n_figs, figsize=(8, 2*n_figs), sharex=True ,squeeze=False, layout='constrained')
    fig.get_layout_engine().set(h_pad=0, hspace=0)   # panels touch
    for spec, ax in zip(specs, axes.flat):

        idx = specs_dict[spec]
        ax.loglog(r, n[:,idx], 'black') # Species over r
        ax.set_ylabel(f"{spec} abundance")

        for i, hair in enumerate(hairs):
            r_hair = r[i*interval : i*interval+nb_evol+1]
            ax.loglog(r_hair, hair[:,idx], lw=.5) # Hair over subset r

    ax.set_xlabel('radius [cm]')
    fig.suptitle(f"Evolutions over {nb_evol} steps on {test_path}")
    return fig, axes



def plot_identity(model:MACE, meta:Meta, test_path:str|Path, 
                 specs:list[str]=['CO', 'H2O', 'OH', 'C2H2', 'C2H', 'CH3C5NH+', 'C10H2+']):
    '''
    Plots the autoencoded forms of the given specs. 
    
    If model.p_out, plots reconstruction of p (dashed). Everything is denormalised.
    '''
    specs_dict = utils.get_specs(meta.data_type)

    # Not ds.get_test_data: we need the dataset object to denormalise p
    data    = ds.get_type_class(meta.data_type, ds.DATA_CLASSES)(meta=meta, train=False)
    model1D = data.load_model(test_path)
    n, p, _ = data.preprocess(model1D)
    r  = model1D.radius

    # n has one row more than p: n[j] with p[j] is the state at r[j]
    with torch.no_grad():
        n_hat, p_hat = model.autoencode(n[:-1], p)
    n     = ds.get_n(n)               # denormalise
    n_hat = ds.get_n(n_hat)
    p     = ds.get_phys(p, data)
    if model.p_out:
        p_hat = ds.get_phys(p_hat, data)

    fig, axes = plt.subplots(2, figsize=(8, 8), sharex=True ,squeeze=False, layout='constrained')
    axp, axn = axes.ravel()[0], axes.ravel()[1]
    fig.get_layout_engine().set(h_pad=0, hspace=0)   # panels touch

    # Physical state
    p_names = ['ρ', 'T', 'ξ', 'A_V']
    for k, param in enumerate(p_names):
        color = f"C{k}"
        axp.loglog(r[:-1], p[:,k], lw=1, color=color, label=param)
        if model.p_out and (p_hat is not None):
            axp.loglog(r[:-1], p_hat[:,k], lw=1, ls='--', color=color)
    axp.set_ylabel("Physical state")
    axp.legend()

    # Abundances
    for k, spec in enumerate(specs):
        idx = specs_dict[spec]
        color = f"C{k}"
        axn.loglog(r, n[:,idx], lw=1, color=color, label=spec) # Species over r
        axn.loglog(r[:-1], n_hat[:,idx], lw=1, ls='--', color=color) # Autoencoded, same rows as p
    axn.set_ylabel(r"Abundances relative to $H_2$")
    axn.set_xlabel('radius [cm]')
    axn.legend()

    fig.suptitle(f"Identity loss of autoencoder on {test_path}")
    return fig, axes



def diagnose(model:MACE, meta:Meta, test_path:str|Path, nb_evol:int, 
             specs:list[str]=['CO', 'H2O', 'OH', 'C2H2', 'C2H', 'CH3C5NH+', 'C10H2+'], 
             interval:int=1):
    # Ensure model folder exists
    if not hasattr(meta, 'path'):
        print("Model does not have a directory! Use meta.makeMeta('path'). Defaulting to /temp")
        meta.makeMeta('/temp') #TODO: implement os.cwd

    # Bigplot
    model1D, (n, p, dt), info = ds.get_test_data(test_path, meta)
    with torch.no_grad():
        n_hat, status = model.rollout(n[0:1], p[None], dt[None])
    n_evol = torch.cat((n[0:1], n_hat[0]), dim=0)      ## (N, 468)
    n_real = ds.get_n(n)
    n_evol = ds.get_n(n_evol)
    fig, axes = plot_n(model1D, '1DCSE', n_real, n_evol, specs=specs, step=False)
    fig.suptitle(f'MACE v2, {test_path}', y=1.0)
    fig.savefig(os.path.join(meta.path,'figs/preds.png'))
    plt.show()

    # HAIRPLOT
    fig, axes = plot_hairs(model, meta, test_path, nb_evol, specs, interval)
    fig.savefig(os.path.join(meta.path,'figs/hairs.png'))
    plt.show()

    # IDENTITY PLOT
    fig, axes = plot_identity(model, meta, test_path, specs)
    fig.savefig(os.path.join(meta.path,'figs/identity.png'))
    plt.show()