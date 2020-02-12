from torch.nn import Module, Sequential, Linear, BatchNorm1d, LeakyReLU, ReLU
from torch.nn.modules import Dropout
from torch import randn_like, cat, exp



def linear_norm_layer(inp_size: int, out_size: int,
                      dropout_rate: float = 0.2,
                      slope: float = 0.2):
    '''

    '''
    return Sequential(
        Linear(inp_size, out_size),
        LeakyReLU(slope),
        Dropout(dropout_rate),
        BatchNorm1d(out_size),
    )


class trVAE(Module):
    '''
    Autoencoder class to be traied with mmd loss
    Unoficcial realization from "Conditional out-of sample generation for
    unpaired data usig trVAE paper

    '''
    def __init__(self, input_dim: int,
                 latent_dim: int = 50):
        '''
        :param input_dim: int, input features dimension
        :param latent_dim
        '''
        super(trVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Sequential(
            linear_norm_layer(input_dim + 1, 800), #+1 for conditionv
            linear_norm_layer(800, 800),
            linear_norm_layer(800, 128)
        )
        self.mu_layer = Linear(128, latent_dim)
        self.var_layer = Linear(128, latent_dim)
        self.mmd_layer = linear_norm_layer(latent_dim + 1, 128)
        self.decoder = Sequential(
            linear_norm_layer(128, 800),
            linear_norm_layer(800, 800),
            Linear(800, input_dim),
            ReLU(inplace=True)
        )

    def encode(self, x, condition):
        '''
        Encodes input features to the parameters of normal distribution

        :param x: Tensor of input features
        :param condition: Tensor with class index
        :return mu, var: Expected values and sigma of latent's distribution
        '''
        features = cat([x, condition], dim=1)
        encoded = self.encoder(features)

        mu = self.mu_layer(encoded)
        sigma = self.var_layer(encoded)

        return mu, sigma

    def reparameterize(self, mu, sigma):
        '''
        Sample latents from normal distribution with mu and sigma params
        '''
        std = exp(sigma / 2)
        eps = randn_like(std)

        return eps*std + mu

    def decode(self, latents, condition):
        '''
        Returns tuple contains reconstructed values and output of MMD layer
        '''
        features = cat([latents, condition], dim=1)
        mmd_latents = self.mmd_layer(features)
        reconstructed = self.decoder(mmd_latents)

        return reconstructed, mmd_latents

    def forward(self, x, condition):
        '''
        Provides input features trough trVAE

        :Returns: reconstructed values as AE and values for MMD criterion
        '''
        if len(condition.shape) == 1:
            condision = condition.view(-1, 1)
        elif condition.size(1) != 1:
            condition = condition.argmax(dim=1).float().view(-1, 1)
        mu, sigma = self.encode(x, condition)
        latents = self.reparameterize(mu, sigma)

        return self.decode(latents, condition), mu, sigma

