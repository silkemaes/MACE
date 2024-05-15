import torch.nn     as nn
import torch    
import numpy as np
import torchode     as to      # Lienen, M., & Günnemann, S. 2022, in The Symbiosis of Deep Learning and Differential Equations II, NeurIPS. https://openreview.net/forum?id=uiKVKTiUYB0
import autoencoder  as ae
import latentODE    as lODE
from scipy.stats    import gmean
from time           import time


class Solver(nn.Module):
    '''
    The Solver class presents the full architecture of MACE.
    Components:
        1) Encoder; neural network with adjustable amount of nodes and layers
        2) Latent ODE; ODE given by function g, with trainable elements 
        3) Decoder; neural network with adjustable amount of nodes and layers

    '''
    def __init__(self, 
                 n_dim, p_dim, z_dim,  
                 nb_hidden, ae_type, 
                 scheme, nb_evol, 
                 lr,
                 path,
                 DEVICE,
                 g_nn = False, 
                 atol = 1e-5, rtol = 1e-2):
        # def __init__(self,  p_dim, z_dim, DEVICE, n_dim, nb_hidden, ae_type, g_nn = False, atol = 1e-5, rtol = 1e-2):
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
        input_ae_dim  = n_dim
        if not self.g_nn:
            self.g = lODE.G(z_dim)
            input_ae_dim  = input_ae_dim+p_dim
            self.odeterm = to.ODETerm(self.g, with_args=False)
        if self.g_nn:
            self.g = lODE.Gnn(p_dim, z_dim)
            self.odeterm = to.ODETerm(self.g, with_args=True)

        self.step_method          = to.Dopri5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
        self.adjoint              = to.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE) # type: ignore

        self.jit_solver = torch.compile(self.adjoint)

        ## Setting the autoencoder (encoder + decoder)
        self.encoder = ae.Encoder(input_dim=input_ae_dim, latent_dim=z_dim, nb_hidden=nb_hidden, ae_type=ae_type)
        self.decoder = ae.Decoder(latent_dim=z_dim      , output_dim=n_dim, nb_hidden=nb_hidden, ae_type=ae_type)

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

        x_0 = n_0               ## use NN version of G
        if not self.g_nn:       ## DON'T use NN version of G
            ## Ravel the abundances n_0 and physical input p to x_0
            x_0 = torch.cat((p, n_0), axis=-1) # type: ignore

        ## Encode x_0, returning the encoded z_0 in latent space
        tic = time()
        z_0 = self.encoder(x_0)
        toc = time()
        enc_time = toc-tic
        
        ## Create initial value problem
        problem = to.InitialValueProblem(
            y0     = z_0.to(self.DEVICE),  
            t_eval = tstep.view(z_0.shape[0],-1).to(self.DEVICE),
        )

        ## Solve initial value problem. Details are set in the __init__() of this class.
        tic = time()
        solution = self.jit_solver.solve(problem, args=p)
        toc = time()
        solve_time = toc-tic
        z_s = solution.ys.view(-1, self.z_dim)  ## want batches 

        ## Decode the resulting values from latent space z_s back to physical space
        tic = time()
        n_s_ravel = self.decoder(z_s)
        toc = time()
        dec_time = toc-tic

        ## Reshape correctly
        n_s = n_s_ravel.reshape(1,tstep.shape[-1], self.n_dim)

        # print('\nencoder time:', enc_time)
        # print('solver  time:', solve_time)
        # print('decoder time:', dec_time)

        return n_s, z_s, solution.status

        
        
## ---------- OLD VERSION OF THE SOLVER CLASS ---------- ##
## This class is compatible with an older version of the autoencoder

class Solver_old(nn.Module):
    '''
    The Solver class presents the architecture of MACE.
    Components:
        1) Encoder; neural network with adjustable amount of nodes and layers
        2) Neural ODE; ODE given by function g, with trainable elements 
        3) Decoder; neural network with adjustable amount of nodes and layers

    '''
    def __init__(self, p_dim, z_dim, DEVICE,  n_dim=466, g_nn = False, atol = 1e-5, rtol = 1e-2):
        super(Solver_old, self).__init__() # type: ignore

        self.status_train = list()
        self.status_test = list()

        self.z_dim = z_dim
        self.n_dim = n_dim
        self.DEVICE = DEVICE
        self.g_nn = g_nn

        ## Setting the neural ODE
        input_ae_dim  = n_dim
        if not self.g_nn:
            self.g = lODE.G(z_dim)
            input_ae_dim  = input_ae_dim+p_dim
            self.odeterm = to.ODETerm(self.g, with_args=False)
        if self.g_nn:
            self.g = lODE.Gnn(p_dim, z_dim)
            self.odeterm = to.ODETerm(self.g, with_args=True)

        self.step_method          = to.Dopri5(term=self.odeterm)
        self.step_size_controller = to.IntegralController(atol=atol, rtol=rtol, term=self.odeterm)
        self.adjoint              = to.AutoDiffAdjoint(self.step_method, self.step_size_controller).to(self.DEVICE) # type: ignore

        self.jit_solver = torch.compile(self.adjoint)

        ## Setting the autoencoder (enocder + decoder)
        hidden_ae_dim = int(gmean([input_ae_dim, z_dim]))
        self.encoder = ae.Encoder_old(input_dim=input_ae_dim, hidden_dim=hidden_ae_dim, latent_dim=z_dim)
        self.decoder = ae.Decoder_old(latent_dim=z_dim      , hidden_dim=hidden_ae_dim, output_dim=n_dim)

    def set_status(self, status, phase):
        if phase == 'train':
            self.status_train.append(status)
        elif phase == 'test':
            self.status_test.append(status)

    def get_status(self, phase):
        if phase == 'train':
            return np.array(self.status_train)
        elif phase == 'test':
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

        return n_s, z_s, solution.status