import torch.nn     as nn
import numpy        as np
import torch    
import torchode     as to
import autoencoder  as ae
from scipy.stats    import gmean



    

class G(nn.Module):
    '''
    g(z,p) is a function. 
    '''
    def __init__(self, z_dim):
        super(G, self).__init__()
        self.A = nn.Parameter(torch.randn(z_dim, z_dim).requires_grad_(True))
        self.B = nn.Parameter(torch.randn(z_dim, z_dim, z_dim).requires_grad_(True))
        # self.reset_parameters()
        # self.B = B(p_dim, z_dim)   

        # print(self.a.shape)


    def forward(self,t, z):     ## volgorde specifiek voor torchode solver 

        return torch.einsum("ij, bj -> bi", self.A, z) + torch.einsum("ijk, bj, bk -> bi", self.B, z, z)  ## b is de index vd batchsize
    
    


class Solver(nn.Module):
    def __init__(self, p_dim, z_dim, DEVICE,  n_dim=466, atol = 1e-5, rtol = 1e-2):
        super(Solver, self).__init__()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.DEVICE = DEVICE

        self.g       = G(z_dim)
        self.odeterm = to.ODETerm(self.g, with_args=False)

        self.step_method          = to.Dopri5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
        self.adjoint              = to.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE) # type: ignore

        self.jit_solver = torch.compile(self.adjoint)

        input_ae_dim  = n_dim+p_dim
        hidden_ae_dim = int(gmean([input_ae_dim, z_dim]))
        self.encoder = ae.Encoder(input_dim=input_ae_dim, hidden_dim=hidden_ae_dim, latent_dim=z_dim       )
        self.decoder = ae.Decoder(latent_dim=z_dim      , hidden_dim=hidden_ae_dim, output_dim=n_dim)

        # for p in self.encoder.parameters():
        #     p.requires_grad_(False)
        
        # for p in self.decoder.parameters():
        #     p.requires_grad_(False)


    def forward(self, n_0, p, tstep):

        # print(n_0.shape)
        # print(p.shape)

        x_0 = torch.cat((p, n_0), axis=1) # type: ignore

        # print(x_0.shape)

        z_0 = self.encoder(x_0)

        problem = to.InitialValueProblem(
            y0     = z_0.view((1,-1)).to(self.DEVICE),  ## "view" is om met de batches om te gaan
            t_eval = tstep.view((1,-1)).to(self.DEVICE),
        )

        solution = self.jit_solver.solve(problem)

        # print('len tstep    ',tstep.shape)
        # print('stats        ',solution.stats)
        # print('status       ',solution.status.item())

        z_s = solution.ys.view(-1, self.z_dim)  ## want batches 


        n_s_ravel = self.decoder(z_s)
        n_s = n_s_ravel.reshape(1,tstep.shape[1], self.n_dim)

        # print('shape ns', n_s.shape)

        return n_s, solution.status

        
        

    