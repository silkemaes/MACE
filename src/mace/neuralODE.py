import torch.nn     as nn
import numpy        as np
import torch    
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

        hidden_dim1 = out_dim
        out_dim = z_dim**2
        hidden_dim2 = out_dim/2

        self.layer_in = nn.Linear( input_dim, hidden_dim1)
        self.layer_hidden = nn.Linear(hidden_dim1,hidden_dim2)
        self.layer_out = nn.Linear(hidden_dim2, out_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.LeakyReLU(self.layer_hidden(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h.reshape(self.z_dim,self.z_dim) ## vierkant
    

    
    # def convert(self):


class B(nn.Module):
    """
    Neural network that constructs a tensor B from the output layer, 
    starting from the physical input of the chemistry model.
    """
    def __init__(self, input_dim, z_dim):
        super(B, self).__init__()

        self.z_dim = z_dim

        hidden_dim1 = out_dim
        out_dim = z_dim**3
        hidden_dim2 = np.sqrt(out_dim)
        hidden_dim3 = out_dim/2

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
    

class G(nn.Module):
    '''
    g(z,p) is a function. 
    '''
    def __init__(self, p_dim, z_dim):
        super(G, self).__init__()
        self.a = A(p_dim, z_dim)  
        self.b = B(p_dim, z_dim)   

    def forward(self, t, z, p):     ## volgorde specifiek voor torchode solver 
        A = self.a(p)       ## hier wordt de forward() uitgevoerd, normaal
        B = self.b(p)
        return torch.einsum("ij, j -> i", A, z) + torch.einsum("ijk, j, k -> i", B, z, z)
    

class Solver(nn.Module):
    def __init__(self, p_dim, z_dim, n_dim=466, atol = 1e-20, rtol = 1e-6):
        super(Solver, self).__init__()
        self.z_dim = z_dim

        self.g       = G(p_dim, z_dim)
        self.odeterm = to.ODETerm(self.g, with_args=True)

        self.step_method          = to.Tsit5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
        self.solver               = to.AutoDiffAdjoint(self.step_method, self.step_size_controller)

        self.jit_solver = torch.compile(self.solver)

        input_ae_dim  = n_dim+p_dim
        hidden_ae_dim = gmean([input_ae_dim, z_dim])
        self.encoder = ae.Encoder(input_dim=input_ae_dim, hidden_dim=hidden_ae_dim, output_dim=z_dim       )
        self.decoder = ae.Decoder(input_dim=z_dim       , hidden_dim=hidden_ae_dim, output_dim=input_ae_dim)

        self.g = G(p_dim=p_dim, z_dim=z_dim)

    def forward(self, n_0, p, tstep):
        z_0 = self.encoder(n_0)

        problem = to.InitialValueProblem(
            y0     = torch.from_numpy(z_0  ).view((1,-1)),  ## "view" is om met de batches om te gaan
            t_eval = torch.from_numpy(tstep).view((1,-1)),
        )

        solution = self.jit_solver.solve(problem, args=p)

        z_s = solution.ys.view(len(tstep), self.z_dim)  ## want batches 

        y_s_unravel = self.decoder(z_s)

        y_s = y_s_unravel.reshape(1,len(tstep), self.z_dim)

        return y_s

        
        

    