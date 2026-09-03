'''
This script contains the MACE model architecture, called Solver.
It combines the autoencoder with the latent ODE, and includes solving the ODE in the forward pass.

Also an old version of the solver is available still, which is compatible with an older version of the autoencoder.
'''



import torch.nn             as nn
import torch
import numpy                as np
import torchode             as to      # Lienen, M., & Günnemann, S. 2022, in The Symbiosis of Deep Learning and Differential Equations II, NeurIPS. https://openreview.net/forum?id=uiKVKTiUYB0
import src.mace.autoencoder as ae
import src.mace.latentODE   as lODE
from time                   import time
import resource
import os

def _mem(tag=""):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KB
    print(f"[MEM] {tag} maxrss={rss_kb/1024/1024:.3f} GB", flush=True)


def _tensor_stats(name, x, eps=1e-20):
    x_det = x.detach()
    x_cpu = x_det.to("cpu")
    flat = x_cpu.reshape(-1)

    total = flat.numel()
    finite_mask = torch.isfinite(flat)
    finite_n = int(finite_mask.sum().item())
    nan_n = int(torch.isnan(flat).sum().item())
    inf_n = int(torch.isinf(flat).sum().item())

    if finite_n > 0:
        f = flat[finite_mask]
        vmin = float(f.min().item())
        vmax = float(f.max().item())
        vmean = float(f.mean().item())
        vstd = float(f.std(unbiased=False).item()) if f.numel() > 1 else 0.0
        p01 = float(torch.quantile(f, 0.01).item())
        p50 = float(torch.quantile(f, 0.50).item())
        p99 = float(torch.quantile(f, 0.99).item())
        abs_small = int((f.abs() < eps).sum().item())
        nonpos = int((f <= 0).sum().item())
    else:
        vmin = vmax = vmean = vstd = float("nan")
        p01 = p50 = p99 = float("nan")
        abs_small = 0
        nonpos = 0

    print(
        f"[ODE-IN] {name}: "
        f"shape={tuple(x_det.shape)} dtype={x_det.dtype} device={x_det.device} "
        f"numel={total} finite={finite_n} nan={nan_n} inf={inf_n} "
        f"min={vmin:.6e} p01={p01:.6e} p50={p50:.6e} p99={p99:.6e} max={vmax:.6e} "
        f"mean={vmean:.6e} std={vstd:.6e} "
        f"|x|<eps({eps:.1e})={abs_small} <=0={nonpos}",
        flush=True,
    )


def _log_ode_inputs(y0, t_eval, t_start, t_end, p_args=None):
    print("[ODE-IN] ----- begin pre-solve diagnostics -----", flush=True)
    _tensor_stats("y0", y0)
    _tensor_stats("t_eval", t_eval)
    _tensor_stats("t_start", t_start)
    _tensor_stats("t_end", t_end)

    if p_args is not None:
        _tensor_stats("p_args", p_args)

    # Extra dt diagnostics for your case
    dt = t_end - t_start
    _tensor_stats("dt=t_end-t_start", dt)

    dt_cpu = dt.detach().to("cpu").reshape(-1)
    finite_dt = dt_cpu[torch.isfinite(dt_cpu)]
    if finite_dt.numel() > 0:
        near_zero = int((finite_dt.abs() < 1e-14).sum().item())
        nonpos = int((finite_dt <= 0).sum().item())
        print(
            f"[ODE-IN] dt checks: finite={finite_dt.numel()} near_zero(<1e-14)={near_zero} nonpositive={nonpos}",
            flush=True,
        )
    print("[ODE-IN] ----- end pre-solve diagnostics -----", flush=True)

class Solver(nn.Module):
    '''
    The Solver class presents the full architecture of MACE.
    Components:
        1) Encoder; neural network with adjustable amount of nodes and layers
        2) Latent ODE; ODE given by function g, with trainable elements 
        3) Decoder; neural network with adjustable amount of nodes and layers

    '''

    def __init__(self,
                 n_dim,
                 p_dim,
                 z_dim,
                 nb_hidden,
                 ae_type,
                 scheme,
                 nb_evol,
                 lr,
                 path,
                 DEVICE,
                 g_nn=False,
                 atol=1e-5,
                 rtol=1e-25):
        # def __init__(self,  p_dim, z_dim, DEVICE, n_dim, nb_hidden, ae_type, g_nn = False, atol = 1e-5, rtol = 1e-25):
        '''
        Initialising the Solver class with the hyperparameters.

        - n_dim: number of dimensions of the physical output
        - p_dim: number of dimension of the physical input
        - z_dim: number of dimension of the latent space
        
        - nb_hidden: number of hidden layers in the encoder and decoder
        - ae_type: type of autoencoder used

        - scheme: type of scheme used to train the model 
            - 'loc': local training scheme
            - 'int': integrated training scheme
            (see Maes et al., 2024 for more details)
        - nb_evol: number of evolutions used during the integrated training scheme

        - lr: learning rate of the training optimiser 

        - path: path to the model. Here the model will be saved as well as its test results

        - DEVICE: device to run the model on (cuda or cpu)
        
        - g_nn: boolean:
            True: use a neural network version of G (Gnn class)
            False: use the G class (default)

        - atol: absolute tolerance of the ODE solver, default 1e-5
        - rtol: relative tolerance of the ODE solver, default 1e-2

        This class builds the architecture of MACE; with the encoder, decoder and latent ODE.
        It also sets up the ODE solver, with the adjoint method from the torchode package (Lienen & Günnemann 2022).
        '''
        super(Solver, self).__init__()

        self.scheme = scheme
        self.nb_evol = nb_evol

        self.lr = lr

        self.path = path

        self.status_train = list()
        self.status_test = list()

        self.p_dim = p_dim
        self.z_dim = z_dim
        self.n_dim = n_dim

        self.DEVICE = DEVICE
        self.g_nn = g_nn

        ## Setting the neural ODE
        input_ae_dim = n_dim
        if not self.g_nn:
            self.g = lODE.G(z_dim)
            input_ae_dim = input_ae_dim + p_dim
            self.odeterm = to.ODETerm(self.g, with_args=False)
        if self.g_nn:
            self.g = lODE.Gnn(p_dim, z_dim)
            self.odeterm = to.ODETerm(self.g, with_args=True)

        self.step_method = to.Dopri5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol,
                                                          rtol=rtol,
                                                          term=self.odeterm)
        self.adjoint = to.AutoDiffAdjoint(self.step_method,
                                          self.step_size_controller).to(
                                              self.DEVICE)  # type: ignore

        self.jit_solver = torch.compile(self.adjoint)

        ## Setting the autoencoder (encoder + decoder)
        self.encoder = ae.Encoder(input_dim=input_ae_dim,
                                  latent_dim=z_dim,
                                  nb_hidden=nb_hidden,
                                  ae_type=ae_type)
        self.decoder = ae.Decoder(latent_dim=z_dim,
                                  output_dim=n_dim,
                                  nb_hidden=nb_hidden,
                                  ae_type=ae_type)

    def set_status(self, status, type):
        '''
        Sets the status of the torchode solver. 
        (See torchode code/documentation for more information on the status of the solver)
        '''
        if type == 'train':
            self.status_train.append(status)
        elif type == 'test':
            self.status_test.append(status)

    def get_status(self, type):
        '''
        Returns the status of the torchode solver.
        (See torchode code/documentation for more information on the status of the solver)'''
        if type == 'train':
            return np.array(self.status_train)
        elif type == 'test':
            return np.array(self.status_test)

    def set_optimiser(self):
        '''
        Sets the optimiser for the model for its training.
        '''
        self.optimiser = torch.optim.Adam(self.parameters(), lr=self.lr)

        return

    def forward(self, n_0, p, tstep):
        '''
        Forward function giving the workflow of the MACE architecture.
            - n_0: abundances input
            - p: physical input
            - tstep: time steps to solve the ODE

        Currently, this forward pass is written in such a way that it can only
        handle a batch size of 1. Therefore, some transformations on the data
        are done to make sure the batch size is 1.
            >> In a later version, MACE will be made compatible with batch sizes > 1.
        There is code in place to time the encoder, solver and decoder, but this is commented out.
        

        Returns:
            - n_s: the predicted abundances at the time steps tstep
            - z_s: the predicted latent space at the time steps tstep
            - solution.status: the status of the ODE solver
        '''

        # If data is unbatched, batch it to make sure the batch size is 1
        if n_0.dim() == 2:
            n_0 = n_0.unsqueeze(0)
        if p.dim() == 2:
            p = p.unsqueeze(0)
        if tstep.dim() == 1:
            tstep = tstep.unsqueeze(0)
        
        # Grab dimensions of the tensors, should be [B, T, _] where B = batch size, T = time steps, _ = abundances or physical input
        B, T, _ = n_0.shape
        if p.shape[0] != B or p.shape[1] != T:
            raise ValueError(f"Physical input tensor p must have shape [B, T, p_dim] matching abundance tensor n_0 {n_0.shape}, but got {p.shape}")
        if tstep.shape[0] != B or tstep.shape[1] != T:
            raise ValueError(f"Timestep tensor tstep must have shape [B, T] matching abundance tensor n_0 {n_0.shape}, but got {tstep.shape}")

        # Build encoder input
        _mem("Building encoder input with shape")
        if self.g_nn:
            x_0 = n_0  ## use NN version of G
            p = p.to(self.DEVICE)
        else:
            # Concatenate the abundances n_0 and physical input p to x_0, with shape [B, T, n_dim + p_dim]
            x_0 = torch.cat((p, n_0), axis=-1)  # type: ignore
            p = p.to(self.DEVICE)

        x_0 = x_0.to(self.DEVICE)
        tstep = tstep.to(self.DEVICE)

        _mem("flattening input tensors for encoder...")
        # Flatten the batch and time dimensions for the encoder
        x_0_flat = x_0.reshape(B * T, -1)
        p_flat = p.reshape(B * T, -1)
        tstep_flat = tstep.reshape(B * T, 1)
        t_start = torch.zeros_like(tstep_flat[:,0])
        t_end = tstep_flat[:, 0]

        # Encode x_0, returning the encoded z_0 in latent space
        _mem("encoding input...")
        tic = time()
        z_0 = self.encoder(x_0_flat)
        toc = time()
        enc_time = toc - tic

        # Create initial value problem
        y0 = z_0.to(self.DEVICE)
        problem = to.InitialValueProblem(
            y0=y0,
            t_eval=tstep_flat,
            t_start=t_start,
            t_end=t_end,
        )
        # check the health of the input data

        _mem("solving initial value problem...")
        # Solve initial value problem. Details are set in the __init__() of this class.
        tic = time()
        solution = self.jit_solver.solve(problem, args=p_flat) if self.g_nn else self.jit_solver.solve(problem)
        toc = time()
        solve_time = toc - tic
        # the resulting ys has shape [BT, 1, z_dim] so we need to put it back to [BT, z_dim]
        z_s = solution.ys.reshape(B * T, self.z_dim)
        _mem(f"solution obtained in {solve_time} seconds")

        # Decode the resulting values from latent space z_s back to physical space
        tic = time()
        n_s_flat = self.decoder(z_s)
        toc = time()
        dec_time = toc - tic

        ## Reshape to initial batch tensor
        n_s = n_s_flat.reshape(B, T, self.n_dim)
        z_s = z_s.reshape(B, T, self.z_dim)
        status = solution.status.reshape(B, T)
        #print('\nencoder time:', enc_time)
        #print('solver  time:', solve_time)
        #print('decoder time:', dec_time)

        return n_s, z_s, status



## ---------- OLD VERSION OF THE SOLVER CLASS ---------- ##
## This class is compatible with an older version of the autoencoder


# from scipy.stats            import gmean

# class Solver_old(nn.Module):
#     '''
#     The Solver class presents the architecture of MACE.
#     Components:
#         1) Encoder; neural network with adjustable amount of nodes and layers
#         2) Neural ODE; ODE given by function g, with trainable elements
#         3) Decoder; neural network with adjustable amount of nodes and layers

#     '''
#     def __init__(self, p_dim, z_dim, DEVICE,  n_dim=466, g_nn = False, atol = 1e-5, rtol = 1e-2):
#         super(Solver_old, self).__init__() # type: ignore

#         self.status_train = list()
#         self.status_test = list()

#         self.z_dim = z_dim
#         self.n_dim = n_dim
#         self.DEVICE = DEVICE
#         self.g_nn = g_nn

#         ## Setting the neural ODE
#         input_ae_dim  = n_dim
#         if not self.g_nn:
#             self.g = lODE.G(z_dim)
#             input_ae_dim  = input_ae_dim+p_dim
#             self.odeterm = to.ODETerm(self.g, with_args=False)
#         if self.g_nn:
#             self.g = lODE.Gnn(p_dim, z_dim)
#             self.odeterm = to.ODETerm(self.g, with_args=True)

#         self.step_method          = to.Dopri5(term=self.odeterm)
#         self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
#         self.adjoint              = to.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE) # type: ignore

#         self.jit_solver = torch.compile(self.adjoint)

#         ## Setting the autoencoder (enocder + decoder)
#         hidden_ae_dim = int(gmean([input_ae_dim, z_dim]))
#         self.encoder = ae.Encoder_old(input_dim=input_ae_dim, hidden_dim=hidden_ae_dim, latent_dim=z_dim)
#         self.decoder = ae.Decoder_old(latent_dim=z_dim      , hidden_dim=hidden_ae_dim, output_dim=n_dim)

#     def set_status(self, status, phase):
#         if phase == 'train':
#             self.status_train.append(status)
#         elif phase == 'test':
#             self.status_test.append(status)

#     def get_status(self, phase):
#         if phase == 'train':
#             return np.array(self.status_train)
#         elif phase == 'test':
#             return np.array(self.status_test)


#     def forward(self, n_0, p, tstep):
#         '''
#         Forward function giving the workflow of the MACE architecture.
#         '''

#         x_0 = n_0               ## use NN version of G
#         if not self.g_nn:       ## DON'T use NN version of G
#             ## Ravel the abundances n_0 and physical input p to x_0
#             x_0 = torch.cat((p, n_0), axis=-1) # type: ignore

#         ## Encode x_0, returning the encoded z_0 in latent space
#         z_0 = self.encoder(x_0)

#         ## Create initial value problem
#         problem = to.InitialValueProblem(
#             y0     = z_0.to(self.DEVICE),  ## "view" is om met de batches om te gaan
#             t_eval = tstep.view(z_0.shape[0],-1).to(self.DEVICE),
#         )

#         ## Solve initial value problem. Details are set in the __init__() of this class.
#         solution = self.jit_solver.solve(problem, args=p)
#         z_s = solution.ys.view(-1, self.z_dim)  ## want batches

#         ## Decode the resulting values from latent space z_s back to physical space
#         n_s_ravel = self.decoder(z_s)

#         ## Reshape correctly
#         n_s = n_s_ravel.reshape(1,tstep.shape[-1], self.n_dim)

#         return n_s, z_s, solution.status
