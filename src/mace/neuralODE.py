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

        hidden_dim1 = z_dim
        out_dim = z_dim**2
        hidden_dim2 = out_dim//2

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
    

class G(nn.Module):
    '''
    g(z,p) is a function. 
    '''
    def __init__(self, p_dim, z_dim):
        super(G, self).__init__()
        self.a = A(p_dim, z_dim)  
        self.b = B(p_dim, z_dim)   

        # print(self.a.shape)

    def forward(self, t, z, p: torch.Tensor):     ## volgorde specifiek voor torchode solver 
        A = self.a(p)       ## hier wordt de forward() uitgevoerd, normaal
        B = self.b(p)
        # print(A.shape, B.shape)
        # print(z.shape)
        return torch.einsum("ij, bj -> bi", A, z) + torch.einsum("ijk, bj, bk -> bi", B, z, z)  ## b is de index vd batchsize
    

class Solver(nn.Module):
    def __init__(self, p_dim, z_dim, DEVICE,  n_dim=466, atol = 1e-20, rtol = 1e-5):
        super(Solver, self).__init__()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.DEVICE = DEVICE

        self.g       = G(p_dim, z_dim)
        self.odeterm = to.ODETerm(self.g, with_args=True)

        self.step_method          = to.Dopri5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
        self.adjoint              = to.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE)

        self.jit_solver = torch.compile(self.adjoint)

        input_ae_dim  = n_dim#+p_dim
        hidden_ae_dim = int(gmean([input_ae_dim, z_dim]))
        self.encoder = ae.Encoder(input_dim=input_ae_dim, hidden_dim=hidden_ae_dim, latent_dim=z_dim       )
        self.decoder = ae.Decoder(latent_dim=z_dim      , hidden_dim=hidden_ae_dim, output_dim=input_ae_dim)

        self.g = G(p_dim=p_dim, z_dim=z_dim)

    def forward(self, n_0, p, tstep):
        z_0 = self.encoder(n_0)

        # problem = to.InitialValueProblem(
        #     y0     = z_0.view((1,-1)).to(self.DEVICE),  ## "view" is om met de batches om te gaan
        #     t_eval = tstep.view((1,-1)).to(self.DEVICE),
        # )

        problem = to.InitialValueProblem(
            y0     = z_0.to(self.DEVICE),  ## "view" is om met de batches om te gaan
            t_eval = tstep.to(self.DEVICE),
        )

        solution = self.jit_solver.solve(problem, args=p)

        print('len tstep    ',tstep.shape)
        print('stats        ',solution.stats)
        print('status       ',solution.status)

        # # print('solution',solution.ys.shape)
        z_s = solution.ys#.view(-1, self.z_dim)  ## want batches 

        print(z_s.shape)

        # n_0 = self.decoder(z_0.view(-1, self.z_dim))

        n_s_ravel = self.decoder(z_s)
        # n_s = n_s_ravel.reshape(1,tstep.shape[1], self.n_dim)

        # print('shape ns', n_s.shape)

        return n_s_ravel

        
        

    