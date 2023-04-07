import torch.nn          as nn


class Encoder(nn.Module):
    """
    Encoder network.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, nb_hidden = 1, type = 'straight'):
        super(Encoder, self).__init__()

        self.layer_in = nn.Linear( input_dim, hidden_dim)
        self.hidden = list()

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
        
    def forward(self, x):
        h = self.LeakyReLU(self.layer_in(x))
        for layer in self.hidden:
            h = self.LeakyReLU(layer(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h

    def set_name(self, name):
        self.name = name
        return


def overview(coder):
    ## Retrieve the number of nodes in each layer
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
    (input_nodes, hidden_nodes, output_nodes), (input_params,hidden_params,output_params) = overview(coder)
    total_params = input_params + output_params
    print(str(coder.name)+':')
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
    
class Decoder(nn.Module):
    """
    Decoder network.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, nb_hidden = 1, type = 'straight'):
        super(Decoder, self).__init__()

        self.hidden = list()
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
        
    def forward(self, x):
        h = self.LeakyReLU(self.layer_in(x))
        for layer in self.hidden:
            h = self.LeakyReLU(layer(h))
        h = self.LeakyReLU(self.layer_out(h))
        return h

    def set_name(self, name):
        self.name = name
        return
    
    
class Autoencoder(nn.Module):
    """
    Autoencoder.
    """
    def __init__(self, Encoder, Decoder):
        super(Autoencoder, self).__init__()
        
        self.Encoder = Encoder
        self.Decoder = Decoder
                
    def forward(self, x):
        h = self.Encoder(x)
        h = self.Decoder(h)
        return h

    def set_name(self, name):
        self.name = name
        return