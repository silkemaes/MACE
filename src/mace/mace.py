'''
MACE architecture.

Reworked from the ground up with love :D

Version 2.0
'''

import torch.nn             as nn
import torch
import torchode             as to      # Lienen, M., & Günnemann, S. 2022, in The Symbiosis of Deep Learning and Differential Equations II, NeurIPS. https://openreview.net/forum?id=uiKVKTiUYB0
import src.mace.autoencoder as ae
import src.mace.latentODE   as lODE



class AutoEncoder(nn.Module):
    '''
    Parent class of MACE
    Builds an autoencoder with the passed arguments
    Idk seems nice
    '''
    def __init__(self, 
                 in_dim,
                 z_dim,
                 out_dim,
                 ae_type='simple'):
        super().__init__()
        # Dimensions
        self.in_dim = in_dim
        self.z_dim = z_dim
        self.out_dim = out_dim
        # Autoencoder (self.encoder/decoder)
        self.setAutoencoder(ae_type)

    def setAutoencoder(self, ae_type):
        self.encoder = ae.Encoder(input_dim=self.in_dim,
                                  latent_dim=self.z_dim,
                                  ae_type=ae_type)
        self.decoder = ae.Decoder(output_dim=self.out_dim,
                                  latent_dim=self.z_dim,
                                  ae_type=ae_type)

    def encode(self, n):
        '''
        Encode the real state (n)

        - n: (B, in_dim) real states

        Returns
        - z: (B, z_dim) corresponding latent states
        '''
        z = self.encoder(n)
        return z

    def decode(self, z):
        '''
        Decode the latent state (z)

        - z: (B, z_dim) latent states

        Returns
        - n: (B, out_dim) corresponding real states
        '''
        n = self.decoder(z)
        return n

    def forward(self, n):
        '''
        Auto-encodes n. Returns D(E(n)) and E(n).

        - n: (B, in_dim) real states

        Returns:
        - n_hat: (B, out_dim) corresponding real states
        - z:     (B, z_dim) 
        '''
        z = self.encode(n)
        n_hat = self.decode(z)
        return n_hat, z
    
        
class MACE(AutoEncoder):
    '''
    Rework of MACE.py
    will fix documentation later 
    glory to mamdanistan
    '''
    # ODE helpers, unregistered as submodules (see setNeuralODE)
    # ts some nasty ass code but who cares
    _term:   to.ODETerm
    _solver: to.AutoDiffAdjoint

    def __init__(self, 
                 n_dim,
                 p_dim,
                 z_dim,
                 ae_type='simple',
                 g_nn=False,
                 atol=1e-5,
                 rtol=1e-2):
        '''
        Main MACE architecture. Contains all updateable weights.
        
        Planned to include self propagation as _rollout_()
        '''
        in_dim = p_dim + n_dim
        if g_nn: in_dim = n_dim
        super().__init__(in_dim, z_dim, n_dim, ae_type)
        # Dimensions (in_dim, z_dim, out_dim are set by AutoEncoder)
        self.n_dim, self.p_dim = n_dim, p_dim
        # Keep track of version for loading old weights
        self.version = '2'
        # Neural version flag
        self.g_nn = g_nn
        # Neural ODE (self.g, self._term, self._solver)
        self.setNeuralODE(atol, rtol)

    def setNeuralODE(self, atol, rtol):
        '''
        Build the latent ODE dz/dt = g(z)

        Only g is registered as a submodule.
        '''
        # set G and tolerances
        if self.g_nn:
            self.g = lODE.Gnn(self.p_dim, self.z_dim)
        else:
            self.g = lODE.G(self.z_dim)
        self.atol, self.rtol = atol, rtol

        # adapt g to torchode's calling convention f(t, z[, args]).
        term = to.ODETerm(self.g, with_args=self.g_nn, with_stats=False)
        # setup solver with custom backprop (adjoint)
        solver = to.AutoDiffAdjoint(to.Dopri5(),
                                    to.IntegralController(atol=atol, rtol=rtol))
        # object.__setattr__ bypasses nn.Module's automatic submodule registration
        object.__setattr__(self, '_term', term)
        object.__setattr__(self, '_solver', solver)

    def solveNeuralODE(self, z0, p, dt):
        '''
        Integrate the latent ODE from t = 0 to t = dt, independently per batch row.

        - z0: (B, z_dim) initial latent states
        - p:  (B, p_dim) physical parameters (only used by Gnn)
        - dt: (B,)       time steps

        Returns
        - z1:    (B, z_dim) latent states at t = dt
        - status:(B,)       torchode status codes (0 = success, see torchode.status_codes.Status)
        '''
        problem = to.InitialValueProblem(y0=z0,
                                         t_start=torch.zeros_like(dt),
                                         t_end=dt,
                                         t_eval=dt[:, None]) # TODO:remove this (will slightly break old models)
        solution = self._solver.solve(problem, term=self._term, args=p) # Args is ignored if not self.g_nn
        return solution.ys[:, -1], solution.status

    def encodeState(self, n, p):
        '''
        Creates the correct input for self.encode,
        then runs self.encode()
        
        - n: (B, n_dim) real states
        - p: (B, p_dim) physical parameters

        Returns
        - z: (B, z_dim) latent states
        '''
        x = n
        if not self.g_nn: # Feed p into encode
            x = torch.cat((p, n), dim=-1)
        return self.encode(x)

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
        z0 = self.encodeState(n0, p)
        z1, status = self.solveNeuralODE(z0, p, dt)
        n1 = self.decode(z1)
        return n1, z1, status

    def rollout(self, n0, p_seq, dt):
        '''
        Repeatedly evolves a starting state n0 
        over a timestep dt, injecting p at every step i.

        - n0:    (B,    n_dim) initial real states
        - p_seq: (B, i, p_dim) physical parameters per step
        - dt:    (B, i)        time step sizes

        Returns
        - n_e_hat:(B, i, n_dim) predicted real states after each step
        - status: (B, i)        torchode status codes (0 = success, see torchode.status_codes.Status)
        '''
        n, n_e_hat, status = n0, [], []
        for i in range(p_seq.shape[1]):
            n, _, s = self(n, p_seq[:, i], dt[:, i])
            n_e_hat.append(n)
            status.append(s)
        return torch.stack(n_e_hat, dim=1), torch.stack(status, dim=1)
