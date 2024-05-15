'''
This script contains the neural network architecture for the autoencoder.

Classes:
    - Encoder: neural network for the encoder
    - Decoder: neural network for the decoder
    - Autoencoder: neural network for the autoencoder
'''



import torch.nn          as nn
import torch

torch.set_default_dtype(torch.float64)

class Encoder(nn.Module):
    """
    Encoder neural network.

    Architecture:
    - input_dim: number of input nodes = number of chemical species + physical parameters
    - latent_dim: number of output nodes = number of latent variables
    - nb_hidden: number of hidden layers; options are '1' or '2'
    - ae_type: type of autoencoder; options are 'simple' or 'complex'
        The complex autoencoder has more nodes in the hidden layers, 
        and this more trainable parameters.

    """
    def __init__(self, input_dim, latent_dim, nb_hidden = 1, ae_type = 'complex'):
        '''
        if nb_hidden == 1:
            input_dim --> 264 --> 64 --> latent_dim
        if nb_hidden == 2:
            if ae_type == 'simple': 
                input_dim --> 256 --> 128 --> 64 --> latent_dim
            if ae_type == 'complex':
                input_dim --> 512 --> 256 --> 64 --> latent_dim        
        '''
        super(Encoder, self).__init__()
        hidden_out_dim = 64

        self.hidden = nn.ModuleList()

        if nb_hidden == 1:
            hidden1_dim = 264
            self.layer_in = nn.Linear(input_dim, hidden1_dim)
            layer = nn.Linear(hidden1_dim, hidden_out_dim)
            self.hidden.append(layer)
        if nb_hidden == 2:
            # print('in hidden')
            if ae_type == 'simple':
                # print('in simple')
                hidden1_dim = 256
                hidden2_dim = 128

                self.layer_in = nn.Linear(input_dim, hidden1_dim)
                
                layer = nn.Linear(hidden1_dim, hidden2_dim)
                self.hidden.append(layer)
                layer = nn.Linear(hidden2_dim, hidden_out_dim)
                self.hidden.append(layer)
            if ae_type == 'complex':
                hidden1_dim = 512
                hidden2_dim = 256

                self.layer_in = nn.Linear(input_dim, hidden1_dim)
                
                layer = nn.Linear(hidden1_dim, hidden2_dim)
                self.hidden.append(layer)
                layer = nn.Linear(hidden2_dim, hidden_out_dim)
                self.hidden.append(layer)

        self.layer_out = nn.Linear(hidden_out_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Tanh = nn.Tanh()
        
    def forward(self, x):
        '''
        Forward pass of the encoder.

        activation function: leaky ReLU, except for last layer: tanh
        '''

        h = self.LeakyReLU(self.layer_in(x))
        for layer in self.hidden:
            h = self.LeakyReLU(layer(h))
        h = self.Tanh(self.layer_out(h))
        return h




class Decoder(nn.Module):
    """
    Decoder neural network.

    Architecture:
    - latent_dim: number of input nodes = number of latent variables   
    - output_put: number of output nodes = number of chemical species
    - nb_hidden: number of hidden layers; options are '1' or '2'
    - ae_type: type of autoencoder; options are 'simple' or 'complex'
        The complex autoencoder has more nodes in the hidden layers,
        and this more trainable parameters.

    """
    def __init__(self, latent_dim, output_dim, nb_hidden = 2, ae_type = 'complex'):
        ''' 
        if nb_hidden == 1:
            latent_dim --> 264 --> 64 --> output_dim
        if nb_hidden == 2:
            if ae_type == 'simple':
                latent_dim --> 128 --> 256 --> output_dim
            if ae_type == 'complex':
                latent_dim --> 256 --> 512 --> output_dim
        '''
        super(Decoder, self).__init__()

        self.hidden = nn.ModuleList()
        hidden_in_dim = 64

        self.layer_in = nn.Linear(latent_dim, hidden_in_dim)

        if nb_hidden == 1:
            hidden1_dim = 264
            layer = nn.Linear(hidden_in_dim, hidden1_dim)
            self.hidden.append(layer)
            self.layer_out = nn.Linear(hidden1_dim, output_dim)

        if nb_hidden == 2:
            if ae_type == 'simple':
                hidden2_dim = 256
                hidden1_dim = 128

                layer = nn.Linear(hidden_in_dim, hidden1_dim)
                self.hidden.append(layer)
                layer = nn.Linear(hidden1_dim, hidden2_dim)
                self.hidden.append(layer)
                self.layer_out = nn.Linear(hidden2_dim, output_dim)

            if ae_type == 'complex':
                hidden2_dim = 512
                hidden1_dim = 256

                layer = nn.Linear(hidden_in_dim, hidden1_dim)
                self.hidden.append(layer)
                layer = nn.Linear(hidden1_dim, hidden2_dim)
                self.hidden.append(layer)
                self.layer_out = nn.Linear(hidden2_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        ''' 
        Forward pass of the decoder.

        activation function: leaky ReLU
        '''
        h = self.LeakyReLU(self.layer_in(z))
        for layer in self.hidden:
            h = self.LeakyReLU(layer(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h

    
    
class Autoencoder(nn.Module):
    """
    Autoencoder.

    Combines the Encoder and Decoder.
    """
    def __init__(self, Encoder, Decoder):
        super(Autoencoder, self).__init__()
        
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        '''
        Forward pass of the autoencoder.
        '''
        h = self.Encoder(x)
        h = self.Decoder(h)
        return h


    


def get_overview(coder):
    '''
    Retrieve information on a coder.

    Input:
        - coder: neural network (encoder or decoder)
    
    Returns:
        - the number of nodes in each layer
        - the number of parameters in each layer
    '''
    input_nodes = coder.layer_in.in_features
    hidden_nodes = list()
    for layer in coder.hidden:
        hidden_nodes.append(layer.in_features)
    hidden_nodes.append(coder.layer_out.in_features)
    output_nodes = coder.layer_out.out_features

    ## Calculate the number of parameters to be trained in each layer
    input_params = coder.layer_in.out_features * (coder.layer_in.in_features+1)
    hidden_params = list()
    for layer in coder.hidden:
        hidden_params.append(layer.out_features * (layer.in_features+1))
    output_params = coder.layer_out.out_features * (coder.layer_out.in_features+1)

    return (input_nodes, hidden_nodes, output_nodes), (input_params,hidden_params,output_params)

def print_overview(coder):
    '''
    Print the overview of the given coder.
    '''
    (input_nodes, hidden_nodes, output_nodes), (input_params,hidden_params,output_params) = get_overview(coder)
    total_params = input_params + output_params

    print('{:>8} | {:>5} | {:>10}'.format('#'  , 'nodes'    ,  'parameters'))
    print('-----------------------------------')
    print('{:>8} | {:>5} | {:>10}'.format('input'  ,   input_nodes,  input_params))
    for i in range(len(hidden_params)):
        print('{:>8} | {:>5} | {:>10}'.format('hidden'  ,   hidden_nodes[i],  hidden_params[i]))
        total_params += hidden_params[i]
    print('{:>8} | {:>5} | {:>10}'.format('hidden'  ,   hidden_nodes[i+1],  output_params))
    print('{:>8} | {:>5} | {:>10}'.format('output'  ,   output_nodes, '/'))
    print('-----------------------------------')
    print('{:>8} | {:>5} | {:>10}'.format(' ', ' ', total_params))

    return

def overview(ae):
    '''
    Print the overview of the given autoencoder (ae).
    '''
    print('___________________________________\n')
    print('Encoder')
    print_overview(ae.Encoder)
    print('')
    print('Decoder')
    print_overview(ae.Decoder)
    print('___________________________________\n')
    return
    


## ------------------- OLD VERSION ------------------- ##
## The architecture of these encoder and decoder differs from the version used in Maes et al. (2024)
class Encoder_old(nn.Module):
    """
    Encoder neural network.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, nb_hidden = 1, type = 'straight'):
        super(Encoder_old, self).__init__()

        self.layer_in = nn.Linear( input_dim, hidden_dim)
        self.hidden = nn.ModuleList()

        # print(self.layer_in.type)

        ## encoder with decreasing number of nodes in hidden layers (n/2)
        if type == 'decr':
            i=0
            while ((i < nb_hidden) and (hidden_dim/2 > latent_dim)):
                layer = nn.Linear(hidden_dim, int(round(hidden_dim/2)))
                self.hidden.append(layer)
                hidden_dim = int(round(hidden_dim/2))
                i += 1

        ## encoder with the same number of nodes in hidden layer
        if type == 'straight':
            for i in range(nb_hidden):
                layer = nn.Linear(hidden_dim, hidden_dim)
                self.hidden.append(layer)

        self.layer_out = nn.Linear(hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.Tanh = nn.Tanh()
        
    def forward(self, x):

        h = self.LeakyReLU(self.layer_in(x))
        for layer in self.hidden:
            h = self.LeakyReLU(layer(h))
        h = self.Tanh(self.layer_out(h))
        return h


    def set_name(self, name):
        self.name = name
        return



class Decoder_old(nn.Module):
    """
    Decoder nearal network.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, nb_hidden = 1, type = 'straight'):
        super(Decoder_old, self).__init__()

        self.hidden = nn.ModuleList()

        # self.hidden = list()

        ## encoder with decreasing number of nodes in hidden layers (n/2)
        if type == 'decr':
            i = nb_hidden
            n = hidden_dim/(2**(nb_hidden))
            self.layer_in = nn.Linear( latent_dim,int(round(n)))
            while ((i > 0) and (n <= hidden_dim)):
                layer = nn.Linear(int(round(n)), int(round(n*2)))
                self.hidden.append(layer)
                i -= 1
                n = n*2        

        ## encoder with the same number of nodes in hidden layer
        if type == 'straight':
            self.layer_in = nn.Linear( latent_dim,hidden_dim)
            for i in range(nb_hidden):
                layer = nn.Linear(hidden_dim, hidden_dim)
                self.hidden.append(layer)

        self.layer_out = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, z):
        h = self.LeakyReLU(self.layer_in(z))
        for layer in self.hidden:
            h = self.LeakyReLU(layer(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h

    def set_name(self, name):
        self.name = name
        return


