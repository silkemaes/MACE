import torch.nn     as nn
import torch    
import numpy as np
import torchode     as to
import autoencoder  as ae
from scipy.stats    import gmean


class A(nn.Module):
    """
    Neural network that constructs a matrix A from the output layer, 
    starting from the physical input of the chemistry model.
    """
    def __init__(self, input_dim, z_dim):
        super(A, self).__init__()

        self.z_dim = z_dim

        hidden_dim1 = z_dim
        out_dim = z_dim**2
        hidden_dim2 = out_dim//2

        self.layer_in = nn.Linear( input_dim, hidden_dim1)
        self.layer_hidden = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer_out = nn.Linear(hidden_dim2, out_dim)

        self.layer_out.weight.data = torch.zeros_like(self.layer_out.weight)
        # print(self.layer_out.bias.data.shape)
        bias = torch.diag(-torch.ones(z_dim))
        self.layer_out.bias.data = bias.ravel()
        self.layer_hidden.weight.requires_grad_(True)
        self.layer_hidden.bias.requires_grad_(True)

        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.LeakyReLU(self.layer_hidden(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h.reshape(self.z_dim,self.z_dim) ## vierkant
    



class B(nn.Module):
    """
    Neural network that constructs a tensor B from the output layer, 
    starting from the physical input of the chemistry model.
    """
    def __init__(self, input_dim, z_dim):
        super(B, self).__init__()

        self.z_dim = z_dim

        hidden_dim1 = z_dim
        out_dim = z_dim**3
        hidden_dim2 = int(np.sqrt(out_dim))
        hidden_dim3 = out_dim//2

        self.layer_in = nn.Linear( input_dim, hidden_dim1)
        self.layer_hidden1 = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer_hidden2 = nn.Linear(hidden_dim2,hidden_dim3)
        self.layer_out = nn.Linear(hidden_dim3, out_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.LeakyReLU(self.layer_hidden1(h))
        h = self.LeakyReLU(self.layer_hidden2(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h.reshape(self.z_dim,self.z_dim,self.z_dim)      ## kubus
    

class Gnn(nn.Module):
    '''
    g(z,p) is a function. 
    '''
    def __init__(self, p_dim, z_dim):
        super(Gnn, self).__init__()
        self.a = A(p_dim, z_dim)  
        self.b = B(p_dim, z_dim)   

    def forward(self,t, z, p: torch.Tensor):     ## volgorde specifiek voor torchode solver 
        A = self.a(p)       
        B = self.b(p)

        return torch.einsum("ij, bj -> bi", A, z) + torch.einsum("ijk, bj, bk -> bi", B, z, z)  ## b is de index vd batchsize


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
    def __init__(self, p_dim, z_dim, DEVICE,  n_dim=466, g_nn = False, atol = 1e-5, rtol = 1e-2):
        super(Solver, self).__init__()

        self.status_train = list()
        self.status_test = list()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.DEVICE = DEVICE
        self.g_nn = g_nn

        ## Setting the neural ODE
        input_ae_dim  = n_dim
        if not self.g_nn:
            self.g = G(z_dim)
            input_ae_dim  = input_ae_dim+p_dim
            self.odeterm = to.ODETerm(self.g, with_args=False)
        if self.g_nn:
            self.g = Gnn(p_dim, z_dim)
            self.odeterm = to.ODETerm(self.g, with_args=True)

        self.step_method          = to.Dopri5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
        self.adjoint              = to.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE) # type: ignore

        self.jit_solver = torch.compile(self.adjoint)

        ## Setting the autoencoder (enocder + decoder)
        hidden_ae_dim = int(gmean([input_ae_dim, z_dim]))
        self.encoder = ae.Encoder(input_dim=input_ae_dim, hidden_dim=hidden_ae_dim, latent_dim=z_dim)
        self.decoder = ae.Decoder(latent_dim=z_dim      , hidden_dim=hidden_ae_dim, output_dim=n_dim)

    def set_status(self, status, type):
        if type == 'train':
            self.status_train.append(status)
        elif type == 'test':
            self.status_test.append(status)

    def get_status(self, type):
        if type == 'train':
            return np.array(self.status_train)
        elif type == 'test':
            return np.array(self.status_test)


    def forward(self, n_0, p, tstep):
        '''
        Forward function giving the workflow of the MACE architecture.
        '''

        x_0 = n_0               ## use NN version of G
        if not self.g_nn:       ## DON'T use NN version of G
            ## Ravel the abundances n_0 and physical input p to x_0
            x_0 = torch.cat((p, n_0), axis=-1) # type: ignore

        ## Encode x_0, returning the encoded z_0 in latent space
        z_0 = self.encoder(x_0)
        
        ## Create initial value problem
        problem = to.InitialValueProblem(
            y0     = z_0.to(self.DEVICE),  ## "view" is om met de batches om te gaan
            t_eval = tstep.view(z_0.shape[0],-1).to(self.DEVICE),
        )

        ## Solve initial value problem. Details are set in the __init__() of this class.
        solution = self.jit_solver.solve(problem, args=p)
        z_s = solution.ys.view(-1, self.z_dim)  ## want batches 

        ## Decode the resulting values from latent space z_s back to physical space
        n_s_ravel = self.decoder(z_s)

        ## Reshape correctly
        n_s = n_s_ravel.reshape(1,tstep.shape[-1], self.n_dim)

        return n_s, solution.status

        
        

    