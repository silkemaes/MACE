'''
This script builds the latent ordinary differential equation (ODE).

Two different options exist:
    - G: ODE without neural network, but with adjustable/trainable elements (C, A, B)
    - Gnn: ODE with neural networks that constructs the tensors C, A and B from the output layer, starting from the physical input of the chemistry model.
'''


import torch.nn     as nn
import torch    
import numpy        as np


class C(nn.Module):
    """
    Neural network that constructs a vector C (constant term) from the output layer,
    starting from the physical input of the chemistry model.
    """
    def __init__(self, input_dim, z_dim):
        super(C, self).__init__()

        self.z_dim = z_dim

        hidden_dim1 = z_dim
        out_dim = z_dim

        self.layer_in = nn.Linear( input_dim, hidden_dim1)
        self.layer_out = nn.Linear(hidden_dim1, out_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.layer_out(h)                                   ## no activation: entries of C can be any sign
        return h.reshape(-1, self.z_dim)                        ## (batch, z), one C per IVP


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

        #self.layer_out.weight.data = torch.zeros_like(self.layer_out.weight)
        # print(self.layer_out.bias.data.shape)
        #bias = torch.diag(-torch.ones(z_dim))
        #self.layer_out.bias.data = bias.ravel()
        #self.layer_hidden.weight.requires_grad_(True)
        #self.layer_hidden.bias.requires_grad_(True)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, p):
        h = self.LeakyReLU(self.layer_in(p))
        h = self.LeakyReLU(self.layer_hidden(h))
        h = self.layer_out(h)                                   ## no activation: entries of A can be any sign
        return h.reshape(-1, self.z_dim, self.z_dim)            ## (batch, z, z), one A per IVP
    



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
        h = self.layer_out(h)                                   ## no activation: entries of B can be any sign
        return h.reshape(-1, self.z_dim, self.z_dim, self.z_dim)    ## (batch, z, z, z), one B per IVP
    

class Gnn(nn.Module):
    '''
    The G_nn class gives the evolution in latent space.
            
        g_nn(z:t)_i = C_i(p)  + A_ij(p) * z_j(t) + B_ijk(p) * z_j(t) * z_k(t)
        with einstein summation.
            
            Here 
                - z(t) are the encoded species + physical parameters
                - C is a vector with adjustable/trainable elements (1D), constant term
                - A is a matrix with adjustable/trainable elements (2D)
                - B is a tensor with adjustable/trainable elements (3D)
    '''
    def __init__(self, p_dim, z_dim):
        super(Gnn, self).__init__()
        self.c = C(p_dim, z_dim)
        self.a = A(p_dim, z_dim)  
        self.b = B(p_dim, z_dim)   

    def forward(self,t, z, p: torch.Tensor):     ## volgorde specifiek voor torchode solver 
        ## p: (b, p_dim), z: (b, z_dim); b = number of independent IVPs (time steps) solved at once
        C = self.c(p)
        A = self.a(p)       
        B = self.b(p)

        return C + torch.einsum("bij, bj -> bi", A, z) + torch.einsum("bijk, bj, bk -> bi", B, z, z)  ## b is de index vd batchsize


class G(nn.Module):
    '''
    The G class gives the evolution in latent space.
        
        g(z:t)_i = C_i  + A_ij * z_j(t) + B_ijk * z_j(t) * z_k(t)
        with einstein summation.
        
            Here 
                - z(t) are the encoded species + physical parameters
                - C is a vector with adjustable/trainable elements (1D), constant term
                - A is a matrix with adjustable/trainable elements (2D)
                - B is a tensor with adjustable/trainable elements (3D)
    '''
    def __init__(self, z_dim):
        '''
        Initialising the tensors C, A and B.
        '''
        super(G, self).__init__()
        self.C = nn.Parameter(torch.randn(z_dim).requires_grad_(True))
        self.A = nn.Parameter(torch.randn(z_dim, z_dim).requires_grad_(True))
        self.B = nn.Parameter(torch.randn(z_dim, z_dim, z_dim).requires_grad_(True))

    def forward(self,t, z):     ## t has also be given to the forward function, in order that the ODE solver can read it properly
        '''
        Forward function of the G class, einstein summations over indices.
        '''
        return self.C + torch.einsum("ij, bj -> bi", self.A, z) + torch.einsum("ijk, bj, bk -> bi", self.B, z, z)  ## b is the index of the batchsize
        



