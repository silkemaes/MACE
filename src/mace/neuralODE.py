import torch.nn     as nn
import torch    
import torchode     as to
import autoencoder  as ae
from scipy.stats    import gmean



class G(nn.Module):
    '''
    The G class gives the evolution in latent space.
        
        g(z:t)_i = A_ij * z_j(t) + B_ijk * z_j(t) * z_k(t)
        
            Here 
                - z(t) are the encoded species + physical parameters
                - A is a matrix with adjustable/trainable elements (2D)
                - B is a tensor with adjustable/trainable elements (3D)
    '''
    def __init__(self, z_dim):
        super(G, self).__init__()
        self.A = nn.Parameter(torch.randn(z_dim, z_dim).requires_grad_(True))
        self.B = nn.Parameter(torch.randn(z_dim, z_dim, z_dim).requires_grad_(True))

    def forward(self,t, z):     ## t has also be given to the forward function, in order that the ODE solver can read it properly

        return torch.einsum("ij, bj -> bi", self.A, z) + torch.einsum("ijk, bj, bk -> bi", self.B, z, z)  ## b is the index of the batchsize


class Solver(nn.Module):
    '''
    The Solver class presents the architecture of MACE.
    Components:
        1) Encoder; neural network with adjustable amount of nodes and layers
        2) Neural ODE; ODE given by function g, with trainable elements 
        3) Decoder; neural network with adjustable amount of nodes and layers

    '''
    def __init__(self, p_dim, z_dim, DEVICE,  n_dim=466, atol = 1e-5, rtol = 1e-2):
        super(Solver, self).__init__()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.DEVICE = DEVICE

        ## Setting the neural ODE
        self.g       = G(z_dim)
        self.odeterm = to.ODETerm(self.g, with_args=False)

        self.step_method          = to.Dopri5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
        self.adjoint              = to.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE) # type: ignore

        self.jit_solver = torch.compile(self.adjoint)

        ## Setting the autoencoder (enocder + decoder)
        input_ae_dim  = n_dim+p_dim
        hidden_ae_dim = int(gmean([input_ae_dim, z_dim]))
        self.encoder = ae.Encoder(input_dim=input_ae_dim, hidden_dim=hidden_ae_dim, latent_dim=z_dim       )
        self.decoder = ae.Decoder(latent_dim=z_dim      , hidden_dim=hidden_ae_dim, output_dim=n_dim)

        # for p in self.encoder.parameters():
        #     p.requires_grad_(False)
        
        # for p in self.decoder.parameters():
        #     p.requires_grad_(False)


    def forward(self, n_0, p, tstep):
        '''
        Forward function giving the workflow of the MACE architecture.
        '''
        ## Ravel the abundances n_0 and physical input p to x_0
        x_0 = torch.cat((p, n_0), axis=1) # type: ignore

        ## Encode x_0, returning the encoded z_0 in latent space
        z_0 = self.encoder(x_0)

        ## Create initial value problem
        problem = to.InitialValueProblem(
            y0     = z_0.view((1,-1)).to(self.DEVICE),  ## "view" is om met de batches om te gaan
            t_eval = tstep.view((1,-1)).to(self.DEVICE),
        )

        ## Solve initial value problem. Details are set in the __init__() of this class.
        solution = self.jit_solver.solve(problem)
        z_s = solution.ys.view(-1, self.z_dim)  ## want batches 

        ## Decode the resulting values from latent space z_s back to physical space
        n_s_ravel = self.decoder(z_s)

        ## Reshape correctly
        n_s = n_s_ravel.reshape(1,tstep.shape[1], self.n_dim)

        return n_s, solution.status

        
        

    