from torch.nn import Module, Sequential, Linear, LeakyReLU
from torch.nn import BatchNorm1d
from torch import exp, randn_like, cat

from .modules import MishLayer

import numpy as np

def get_hidden_list(n, h, n_layers, alpha):
    # print(n)
    # print(h)
    # print(n_layers)
    # print(alpha)
    return [int(np.round(n - (n-h)*(x/n_layers)**alpha)) for x in np.arange(n_layers)]


class VAE(Module):
    '''
    Variational autoencoder

    @param bottleneck: int, dimension of latents
    @param input_size: int, dimension of input feature vector
    @param n_layers: int, number of encoder layers
    @param scale_alpha:float, scaling factor #TODO add decent description
    @param count_classes: int
    @ohe_latent_dim
    '''
    def __init__(self, bottleneck: int, input_size: int, count_classes: int, n_layers: int, scale_alpha:float, ohe_latent_dim: int = 10):
        assert bottleneck > 0
        assert input_size > 0
        assert count_classes > 0
        assert ohe_latent_dim > 0

        self.bottleneck = bottleneck
        self.latent_dim = bottleneck
        self.input_size = input_size
        self.count_classes = count_classes
        self.encoder_sizes = get_hidden_list(input_size, bottleneck, n_layers, scale_alpha)

        self.decoder_sizes = [bottleneck+ohe_latent_dim]+get_hidden_list(input_size, bottleneck+ohe_latent_dim, n_layers, scale_alpha)[::-1]


        super(VAE, self).__init__()
        self.ohe_latent_size = ohe_latent_dim
        encoder_module_list = []
        for insize_, outsize_ in zip(self.encoder_sizes, self.encoder_sizes[1:]):
            encoder_module_list.extend([
              Linear(insize_, outsize_, bias=False),
              BatchNorm1d(outsize_),
              MishLayer(), #LeakyReLU(0.2),  
            ])

#         self.pre_encoder = Sequential(
#             #BatchSwapNoise(0.15),
#             Linear(input_size, input_size//2),
#             BatchNorm1d(input_size//2),
#             MishLayer(), #LeakyReLU(0.2),
#             Linear(input_size//2,input_size//3, bias=False),
#             MishLayer()
#             #BatchNorm1d(400),
#             #MishLayer(),
#             #Linear(400, 600),
#             #BatchNorm1d(600),
#             #MishLayer(),
#         )
        self.pre_encoder = Sequential(*encoder_module_list)

        self.fc_mu = Linear(self.encoder_sizes[-1], bottleneck, bias=False)
        self.fc_sigma = Linear(self.encoder_sizes[-1], bottleneck, bias=False)
        self.fc_class = Linear(self.encoder_sizes[-1], self.count_classes, bias=False)


        decoder_module_list = []
        for insize_, outsize_ in zip(self.decoder_sizes, self.decoder_sizes[1:]):
            decoder_module_list.extend([
              MishLayer(),
              Linear(insize_, outsize_, bias=False),
              BatchNorm1d(outsize_),
            ])
        self.decoder = Sequential(*decoder_module_list[:-1])

#         self.decoder = Sequential(
#             MishLayer(),
# #             Linear(bottleneck+self.ohe_latent_size,
# #                    input_size
# #                    )
#             Linear(bottleneck+self.ohe_latent_size,
#                     64
#                     ),
#             BatchNorm1d(64),
#             MishLayer(),
#             Linear(64,
#                     128
#                     ),
#             BatchNorm1d(128),
#             MishLayer(),
#             Linear(128, input_size)
#         )

        self.ohe_pipeline = Sequential(
            Linear(count_classes, self.ohe_latent_size),
#            BatchNorm1d(self.ohe_latent_size),
#            MishLayer(), #LeakyReLU(0.2),
#           Linear(self.ohe_latent_size, self.ohe_latent_size),
#           BatchNorm1d(self.ohe_latent_size),
#           LeakyReLU(0.2),
#           Linear(self.ohe_latent_size, self.ohe_latent_size),
#           BatchNorm1d(self.ohe_latent_size),
#           LeakyReLU(0.2),
#           Linear(self.ohe_latent_size, self.ohe_latent_size),
#           BatchNorm1d(self.ohe_latent_size),
#           LeakyReLU(0.2),
        )

    def encode(self, x, y):
        h = self.pre_encoder(x)
        return self.fc_mu(h), self.fc_sigma(h), self.fc_class(h)

    def reparameterize(self, mu, logvar):
        std = exp(0.5*logvar)
        eps = randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        '''
            :return res, y: return decoded vectors and activations after mmd layer
        '''
        y = self.ohe_pipeline(y)
        w = cat([z,y], dim=1)
        res = self.decoder(w)
        return res, y

    def forward(self, x, y):
        mu, logvar, form = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def latents(self, x, y):
        mu, logvar, form = self.encode(x, y)
        return self.reparameterize(mu, logvar)

