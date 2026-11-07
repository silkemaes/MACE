import torch.nn             as nn
import torch
import torchode             as to      # Lienen, M., & Günnemann, S. 2022, in The Symbiosis of Deep Learning and Differential Equations II, NeurIPS. https://openreview.net/forum?id=uiKVKTiUYB0
import src.mace.autoencoder as ae
import src.mace.latentODE   as lODE

'''
MACE architecture.

Reworked from the ground up with love :D

Version 2.0
'''


class MACE(nn.Module):
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
        super().__init__()
        # Keep track of version for loading old weights
        self.version = '2'
        # Dimensions
        self.n_dim, self.p_dim, self.z_dim = n_dim, p_dim, z_dim
        # Neural version flag
        self.g_nn = g_nn

        # Autoencoder (self.encoder/decoder)
        self.setAutoencoder(ae_type)

        # Neural ODE (self.g, self._term, self._solver)
        self.setNeuralODE(atol, rtol)

    def setAutoencoder(self, ae_type):
        ae_in_dim = self.n_dim + self.p_dim
        if self.g_nn: ae_in_dim = self.n_dim

        self.encoder = ae.Encoder(input_dim=ae_in_dim,
                                  latent_dim=self.z_dim,
                                  ae_type=ae_type)
        self.decoder = ae.Decoder(output_dim=self.n_dim,
                                  latent_dim=self.z_dim,
                                  ae_type=ae_type)

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

    def encode(self, n, p):
        '''
        Encode the physical state (n0) or (p0, n0) 

        - n: (B, n_dim) real states
        - p: (B, p_dim) physical parameters

        Returns
        - z: (B, z_dim) corresponding latent states
        '''
        x = n  ## use NN version of G
        if not self.g_nn:  ## DON'T use NN version of G
            x = torch.cat((p, n), dim=-1) 
        ## Encode x_0, returning the encoded z_0 in latent space
        z = self.encoder(x)
        return z

    def solve(self, z0, p, dt):
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

    def decode(self, z1):
        '''
        Decode the physical state (z)

        - z: (B, z_dim) latent states

        Returns
        - n: (B, n_dim) corresponding real states
        '''
        n1 = self.decoder(z1)
        return n1

    def forward(self, n0, p, dt):
        '''
        Evolves n0 over a timestep dt using physical state p.
        
        - n0: (B, n_dim) initial real states
        - p:  (B, p_dim) physical parameters
        - dt: (B,)       time steps

        Returns
        - n1:    (B, n_dim) evolved real states
        - z0:    (B, z_dim) initial latent states
        - status:(B,)       ODE solver error codes
        '''
        z0 = self.encode(n0, p)
        z1, status = self.solve(z0, p, dt)
        n1 = self.decode(z1)
        #n1 = n1_ravel.reshape(1, dt.shape[-1], self.n_dim)
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
