'''
Training functionality for MACE. Supports local and integrated schemes,
and autoencoder-only training (train_ae).

Version 2.0
'''
from time                   import time
import torch

def make_windows(n, p, dt, e):
    '''
    Cut one 1D entry into windows with length e

    For use in rollout: (n[r], p[r], dt[r])--> n[r+1].

    - n:  (N,   n_dim) abundances
    - p:  (N-1, p_dim) physical parameters
    - dt: (N-1,)       time steps
    - e:  window length (1 = local scheme, nb_evol = integrated scheme)

    Returns (with M =  #windows, window j starts at row j)
    - n0:     (M,    n_dim) starting abundances,  n0[j]       = n[j]
    - p_seq:  (M, e, p_dim) physics per step,     p_seq[j, i] = p[j+i]
    - dt_seq: (M, e)        time step per step,   dt_seq[j,i] = dt[j+i]
    - n_e:    (M, e, n_dim) targets,              n_e[j, i]   = n[j+i+1]

    @Claude (discovering torch.unfold)
    '''
    N      = n.shape[0]
    if e > N:
        raise RuntimeError(f"Woah there parther! This town of {N} datapoints ain't big enough for windows of length {e}.")
    n0     = n[:N-e]
    p_seq  = p.unfold(0, e, 1).transpose(1, 2)
    dt_seq = dt.unfold(0, e, 1)
    n_e    = n[1:].unfold(0, e, 1).transpose(1, 2)
    return n0, p_seq, dt_seq, n_e



#####################
### Full training ###
#####################

def run_epoch(model,
              loader,
              loss,
              nb_evol,
              device,
              optimiser=None,
              batch_size=None,
              scheduler=None):
    '''
    Single data pass over an epoch.

    - model:        MACE
    - loader:       DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - loss:         src.mace.loss.Loss
    - device        device on which MACE is stored
    - optimiser:    if None, does a validation epoch (requires no_grad())
    - batch_size:   Amount of windows passed per loss computation.
    '''
    nb_failed, nb_solves = 0, 0

    for n, p, dt in loader:
        # drop the batch dimension (batch_size = 1)
        n, p, dt = n[0].to(device), p[0].to(device), dt[0].to(device)
        n0, p_seq, dt_seq, n_e = make_windows(n, p, dt, nb_evol)

        M = n0.shape[0]
        windowBatches = torch.arange(M, device=device) # valid: linearly pass windows
        if optimiser is not None:
            windowBatches = torch.randperm(M, device=device) # train: mix windows

        for m in windowBatches.split(batch_size or M):
            n_e_hat, status = model.rollout(n0[m], p_seq[m], dt_seq[m])

            loss_value, _ = loss(n_e=n_e[m], 
                            n_e_hat=n_e_hat, 
                            n=n0[m], 
                            p=p_seq[m,0], 
                            model=model)

            if optimiser is not None: # train: refresh optim and clear autograd
                optimiser.zero_grad()
                loss_value.backward()
                optimiser.step()
                if scheduler is not None:
                    scheduler.step()

            nb_failed += int((status!=0).sum())
            nb_solves += status.numel()

    loss.endEpoch()
    return nb_failed, nb_solves

def train(model,
          train_loader, valid_loader,
          train_loss, valid_loss, 
          optimiser, 
          nb_epochs, 
          nb_evol=1,
          batch_size=None,
          scheduler=None,
          print_every=1):
    '''
    Training and validation loop.

    - model:        MACE
    - train_loader: DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - valid_loader: DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - train_loss:   src.mace.loss.Loss (optimised over)
    - valid_loss:   src.mace.loss.Loss (no gradient)
    - optimiser:    torch optimiser over model.parameters()
    - nb_epochs:    number of epochs
    - nb_evol:      window length e
    - batch_size:   Amount of windows passed per train loss computation.
    - scheduler:    Learning rate updater # TODO I don't trust this one
    - print_every:  Epoch status printing cadence. <1 disables printing.

    Returns:
    exit_code:      Summed nODE solver status. Should be 0.
    '''
    device = next(model.parameters()).device
    exit_code = 0.

    tic = time()
    for epoch in range(nb_epochs):

        model.train()
        nb_failed, nb_solves = run_epoch(model, train_loader, train_loss, 
                                         nb_evol, device, 
                                         optimiser, batch_size, scheduler)
        model.eval()
        with torch.no_grad():
            run_epoch(model, valid_loader, valid_loss, 
                      nb_evol, device)

        exit_code += nb_failed

        if print_every < 1:
            continue
        if (epoch+1) % print_every == 0 or epoch+1 == nb_epochs:
            print(f'Epoch {epoch+1}/{nb_epochs}:\n'
                  f' - Train Loss {train_loss.getTotHistory()[-1]:.3e} | '
                  f'Valid Loss {valid_loss.getTotHistory()[-1]:.3e},\n'
                  f' - Failed solves {nb_failed}/{nb_solves} | {time()-tic:.1f} s,\n'
                  f' - lr {optimiser.param_groups[0]["lr"]:.1e}')

    # Rescale the validation loss to match the training loss
    for name, term in valid_loss.terms.items(): 
        term.setScale(train_loss.terms[name].scale)

    return exit_code



#############################
### Autoencoder-only mode ###
#############################

def run_epoch_autoencoder(model,
                          loader,
                          loss,
                          device,
                          optimiser=None,
                          batch_size=None,
                          scheduler=None):
    '''
    Single data pass over an epoch.

    Losses can only contain AE losses (IDN, pIDN).

    - model:        MACE
    - loader:       DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - loss:         src.mace.loss.Loss
    - device        device on which MACE is stored
    - optimiser:    if None, does a validation epoch (requires no_grad())
    - scheduler:    Learning rate updater
    '''

    for n, p, _ in loader:

        # drop the batch dimension (batch_size = 1)
        n, p = n[0, :-1].to(device), p[0].to(device) # last point redundant: no p
        
        R = n.shape[0]
        rows = torch.randperm(R, device=n.device) if optimiser is not None else torch.arange(R, device=n.device)

        for m in rows.split(batch_size or R):
            ae = model.autoencode(n[m], p[m]) # shared by idn and pidn
            loss_value, _ = loss(n=n[m], p=p[m], model=model, ae=ae)

            if optimiser is not None: # train: refresh optim and clear autograd
                optimiser.zero_grad()
                loss_value.backward()
                optimiser.step()
                if scheduler is not None:
                    scheduler.step()

    loss.endEpoch()

def train_autoencoder(model,
                      train_loader, valid_loader,
                      train_loss, valid_loss, 
                      optimiser, 
                      nb_epochs,
                      batch_size=None,
                      scheduler=None,
                      print_every=1):
    '''
    Training and validation loop.

    Losses can only contain AE losses (IDN, pIDN).

    - model:        MACE
    - train_loader: DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - valid_loader: DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - train_loss:   src.mace.loss.Loss (optimised over)
    - valid_loss:   src.mace.loss.Loss (no gradient)
    - device        device on which MACE is stored
    - optimiser:    if None, does a validation epoch (requires no_grad())
    - scheduler:    Learning rate updater
    - print_every:  Epoch status printing cadence. <1 disables printing.
    '''
    device = next(model.parameters()).device

    tic = time()
    for epoch in range(nb_epochs):

        model.train()
        run_epoch_autoencoder(model, train_loader, train_loss, device, 
                              optimiser, batch_size, scheduler)
        model.eval()
        with torch.no_grad():
            run_epoch_autoencoder(model, valid_loader, valid_loss, device, batch_size=batch_size)

        if print_every < 1:
            continue
        if (epoch+1) % print_every == 0 or epoch+1 == nb_epochs:
            print(f'Epoch {epoch+1}/{nb_epochs}:\n'
                  f' - Train Loss {train_loss.getTotHistory()[-1]:.3e} | '
                  f'Valid Loss {valid_loss.getTotHistory()[-1]:.3e},\n'
                  f' - lr {optimiser.param_groups[0]["lr"]:.1e} | {time()-tic:.1f} s')

    # Rescale the validation loss to match the training loss
    for name, term in valid_loss.terms.items(): 
        term.setScale(train_loss.terms[name].scale)


########################################################
### Autoencoder training, full memory loading @CLaude###
########################################################

def stack_rows(dataset):
    '''
    Parse every 1D model of a dataset once and stack its rows.

    Row i of a 1D model is (n[i], p[i]); the last abundance row has no p and is dropped.

    - dataset:  ChemData (e.g. CSEdata), item = (n (N, n_dim), p (N-1, p_dim), dt (N-1,))

    Returns
    - n: (R, n_dim) abundances of all models
    - p: (R, p_dim) physical parameters of all models

    @Claude
    '''
    ns, ps = [], []
    for i in range(len(dataset)):
        n, p, _ = dataset[i]
        ns.append(n[:-1])
        ps.append(p)
    return torch.cat(ns), torch.cat(ps)

def run_ae_epoch(model, n, p, loss, optimiser=None, batch_size=None, scheduler=None):
    '''
    Single autoencoder-only pass over rows (n, p): no ODE solve.

    - model:        MACE
    - n, p:         (R, n_dim), (R, p_dim) rows, from stack_rows()
    - loss:         src.mace.loss.Loss with idn and/or pidn terms only
    - optimiser:    if None, does a validation epoch (requires no_grad())
    - batch_size:   rows per optimiser step (None = all rows)
    - scheduler:    optional LR scheduler, stepped after every optimiser step

    @Claude
    '''
    R = n.shape[0]
    rows = torch.randperm(R, device=n.device) if optimiser is not None else torch.arange(R, device=n.device)

    for m in rows.split(batch_size or R):
        ae = model.autoencode(n[m], p[m]) # shared by idn and pidn
        loss_value, _ = loss(n=n[m], p=p[m], model=model, ae=ae)

        if optimiser is not None:
            optimiser.zero_grad()
            loss_value.backward()
            optimiser.step()
            if scheduler is not None:
                scheduler.step()

    loss.endEpoch()

def train_ae(model,
             train_rows, valid_rows,
             train_loss, valid_loss,
             optimiser,
             nb_epochs,
             batch_size=64,
             scheduler=None,
             print_every=10):
    '''
    Autoencoder-only training (stage A of the training recipe in CLAUDE.md):
    rows shuffled over all 1D models, no ODE, so G is not trained.

    - model:        MACE
    - train_rows:   (n, p) from stack_rows(train dataset)
    - valid_rows:   (n, p) from stack_rows(valid dataset)
    - train_loss:   src.mace.loss.Loss with idn and/or pidn terms (optimised over)
    - valid_loss:   same terms (no gradient)
    - optimiser:    torch optimiser
    - nb_epochs:    number of passes over the training rows
    - batch_size:   rows per optimiser step
    - scheduler:    optional LR scheduler, stepped per optimiser step
                    (steps per epoch = ceil(#train rows / batch_size))
    - print_every:  print the losses every this many epochs

    @Claude
    '''
    device = next(model.parameters()).device
    n_tr, p_tr = (x.to(device) for x in train_rows)
    n_va, p_va = (x.to(device) for x in valid_rows)

    tic = time()
    for epoch in range(nb_epochs):
        model.train()
        run_ae_epoch(model, n_tr, p_tr, train_loss, optimiser, batch_size, scheduler)
        model.eval()
        with torch.no_grad():
            run_ae_epoch(model, n_va, p_va, valid_loss)

        if print_every < 1:
            continue
        if (epoch+1) % print_every == 0 or epoch+1 == nb_epochs:
            print(f'Epoch {epoch+1}/{nb_epochs}: '
                  f'Train Loss {train_loss.getTotHistory()[-1]:.3e} | '
                  f'Valid Loss {valid_loss.getTotHistory()[-1]:.3e} | '
                  f'lr {optimiser.param_groups[0]["lr"]:.1e} | {time()-tic:.1f} s')

    for name, term in valid_loss.terms.items():
        term.setScale(train_loss.terms[name].scale)
