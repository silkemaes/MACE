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
    '''
    N      = n.shape[0]
    if e > N:
        raise RuntimeError(f"Can't evolve {e} datapoints in dataset with lenght {N}.")
    n0     = n[:N-e]
    p_seq  = p.unfold(0, e, 1).transpose(1, 2)
    dt_seq = dt.unfold(0, e, 1)
    n_e    = n[1:].unfold(0, e, 1).transpose(1, 2)
    return n0, p_seq, dt_seq, n_e

def run_epoch(model,
              loader,
              loss,
              nb_evol,
              device,
              optimiser=None):
    '''
    Single data pass over an epoch.

    - model:        MACE
    - loader:       DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - loss:         src.mace.loss.Loss
    - device        device on which MACE is stored
    - optimiser:    if None, does a validation epoch (requires no_grad())
    '''
    nb_failed, nb_solves = 0, 0

    for n, p, dt in loader:
        # drop the batch dimension (batch_size = 1)
        n, p, dt = n[0].to(device), p[0].to(device), dt[0].to(device)

        n0, p_seq, dt_seq, n_e = make_windows(n, p, dt, nb_evol)
        n_e_hat, status = model.rollout(n0, p_seq, dt_seq)

        loss_value, _ = loss(n_e=n_e, 
                       n_e_hat=n_e_hat, 
                       n=n[:-1], 
                       p=p, 
                       model=model)

        if optimiser is not None:
            optimiser.zero_grad()
            loss_value.backward()
            optimiser.step()

        nb_failed += int((status!=0).sum())
        nb_solves += status.numel()

    loss.endEpoch()
    return nb_failed, nb_solves

def train(model,
          train_loader, valid_loader,
          train_loss, valid_loss, 
          optimiser, 
          nb_epochs, 
          nb_evol=1):
    '''
    Training loop.

    - model:        MACE
    - train_loader: DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - valid_loader: DataLoader yielding (n, p, dt) for one 1D model, batch_size = 1
    - train_loss:   src.mace.loss.Loss (optimised over)
    - valid_loss:   src.mace.loss.Loss (no gradient)
    - optimiser:    torch optimiser over model.parameters()
    - nb_epochs:    number of epochs
    - nb_evol:      window length e
    '''
    device = next(model.parameters()).device

    for epoch in range(nb_epochs):
        tic = time()

        model.train()
        nb_failed, nb_solves = run_epoch(model, train_loader, train_loss, 
                                         nb_evol, device, optimiser)
        model.eval()
        with torch.no_grad():
            run_epoch(model, valid_loader, valid_loss, 
                      nb_evol, device)

        print(f'Epoch {epoch+1}/{nb_epochs}:\n'
              f' - Train Loss {train_loss.getTotHistory()[-1]:.3e} | '
              f'Valid Loss {valid_loss.getTotHistory()[-1]:.3e},\n'
              f' - Failed solves {nb_failed}/{nb_solves} | {time()-tic:.1f} s')

    # Rescale the validation loss to match the training loss
    for name, term in valid_loss.terms.items(): 
        term.setScale(train_loss.terms[name].scale)

    return train_loss
