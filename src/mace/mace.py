'''
MACE architecture.

Reworked from the ground up with love :D

Version 2.1
'''

import torch.nn             as nn
import torch
import torchode             as to      # Lienen, M., & Günnemann, S. 2022, in The Symbiosis of Deep Learning and Differential Equations II, NeurIPS. https://openreview.net/forum?id=uiKVKTiUYB0
import src.mace.autoencoder as ae
import src.mace.latentODE   as lODE



class MACE(nn.Module):
    '''
    Main MACE architecture. 
    Contains an encoder, decoder, and nODE solver.

    - n_dim:    Amount of chemical abundances
    - p_dim:    Amount of physical parameters
    - z_dim:    Amount of latent dimensions
    - ae_type:  'simple' or 'linear'
    - p_out:    Have AE return p (for separate idn loss)
    - atol:     Absolute tolerance of nODE solutions
    - rtol:     Relative tolerance of nODE solutions
    '''
    # ODE helpers, unregistered as submodules (see setNeuralODE)
    # Kind of nasty but better on memory
    _term:   to.ODETerm
    _solver: to.AutoDiffAdjoint

    def __init__(self, 
                 n_dim,
                 p_dim,
                 z_dim,
                 ae_type='simple',
                 p_out= False,
                 atol=1e-5,
                 rtol=1e-2):
        '''
        Main MACE architecture. Contains all updateable weights.
        '''
        super().__init__()
        # Data dimensions
        self.n_dim, self.p_dim, self.z_dim = n_dim, p_dim, z_dim
        # AE dimensions
        self.p_out = p_out
        self.in_dim = p_dim + n_dim
        self.out_dim = n_dim
        if self.p_out: self.out_dim = p_dim + n_dim
        # Autoencoder (self.encoder/decoder)
        self.setAutoencoder(ae_type)
        # Neural ODE (self.g, self._term, self._solver)
        self.setNeuralODE(atol, rtol)

    def setAutoencoder(self, ae_type):
        self.encoder = ae.Encoder(input_dim=self.in_dim,
                                  latent_dim=self.z_dim,
                                  ae_type=ae_type)
        self.decoder = ae.Decoder(output_dim=self.out_dim,
                                  latent_dim=self.z_dim,
                                  ae_type=ae_type)

    def encode(self, n, p):
        '''
        Encode the real state (n)

        - n: (B, n_dim) real states
        - p: (B, p_dim) physical parameters

        Returns
        - z: (B, z_dim) corresponding latent states
        '''
        x = torch.cat((p, n), dim=-1)
        z = self.encoder(x)
        return z

    def decode(self, z):
        '''
        Decode the latent state (z)

        - z: (B, z_dim) latent states

        Returns
        - n: (B, n_dim) corresponding real states
        - p: (B, p_dim) corresponding physical parameters
        '''
        x = self.decoder(z)
        if not self.p_out:
            return x, None
        p, n = x.split([self.p_dim, self.n_dim], dim=-1)
        return n, p

    def autoencode(self, n, p):
        '''
        Auto-encodes n. Returns D(E(n)).

        - n: (B, n_dim) real states
        - p: (B, p_dim) physical parameters

        Returns:
        - n_hat: (B, n_dim) corresponding real states
        - p_hat: (B, p_dim) corresponding physical parameters
        '''
        z = self.encode(n, p)
        n_hat, p_hat = self.decode(z)
        return n_hat, p_hat

    def setNeuralODE(self, atol, rtol):
        '''
        Build the latent ODE dz/dt = g(z)

        Only g is registered as a submodule.

        @Claude (avoid double gradient)
        '''
        # set g and tolerances
        self.g = lODE.G(self.z_dim)
        self.atol, self.rtol = atol, rtol

        # adapt g to torchode's calling convention f(t, z).
        term = to.ODETerm(self.g, with_args=False, with_stats=False)
        # setup solver with custom backprop (adjoint)
        solver = to.AutoDiffAdjoint(to.Dopri5(), #type:ignore all around
                                    to.IntegralController(atol=atol, rtol=rtol)) #type:ignore
        # object.__setattr__ bypasses nn.Module's automatic submodule registration
        object.__setattr__(self, '_term', term)
        object.__setattr__(self, '_solver', solver)

    def solveNeuralODE(self, z0, dt):
        '''
        Integrate the latent ODE from t = 0 to t = dt, independently per batch row.

        - z0: (B, z_dim) initial latent states
        - dt: (B,)       time steps

        Returns
        - z1:    (B, z_dim) latent states at t = dt
        - status:(B,)       torchode status codes (0 = success, see torchode.status_codes.Status)
        '''
        problem = to.InitialValueProblem(y0=z0,
                                         t_start=torch.zeros_like(dt), #type:ignore
                                         t_end=dt,
                                         t_eval=dt[:, None]) # TODO:remove this (will slightly break old models)
        solution = self._solver.solve(problem, term=self._term)
        return solution.ys[:, -1], solution.status

    def forward(self, n0, p, dt):
        '''
        Evolves n0 over a timestep dt using physical state p.
        
        - n0: (B, n_dim) initial real states
        - p:  (B, p_dim) physical parameters
        - dt: (B,)       time steps

        Returns
        - n1:    (B, n_dim) evolved real states
        - z1:    (B, z_dim) evolved latent states
        - status:(B,)       ODE solver error codes
        '''
        z0 = self.encode(n0, p)
        z1, status = self.solveNeuralODE(z0, dt)
        n1, _ = self.decode(z1)
        return n1, z1, status

    def rollout(self, n0, p_seq, dt_seq):
        '''
        Repeatedly evolves a starting state n0 
        over a timestep dt, injecting p at every step e.

        - n0:    (B,    n_dim) initial real states
        - p_seq: (B, e, p_dim) physical parameters per step
        - dt_seq:(B, e)        time step sizes

        Returns
        - n_e_hat:(B, e, n_dim) predicted real states after each step
        - status: (B, e)        torchode status codes (0 = success, see torchode.status_codes.Status)
        '''
        n, n_e_hat, status = n0, [], []
        for i in range(p_seq.shape[1]):
            n, _, s = self(n, p_seq[:, i], dt_seq[:, i])
            n_e_hat.append(n)
            status.append(s)
        return torch.stack(n_e_hat, dim=1), torch.stack(status, dim=1)
