from torch.nn import Module, Sequential, Linear, LeakyReLU
from torch.nn import BatchNorm1d
from torch import exp, randn_like, cat

from .modules import MishLayer


class VAE(Module):
    '''
    Variational autoencoder

    @param bottleneck: int, dimension of latents
    @param input_size: int, dimension of input feature vector
    @param count_classes: int
    @ohe_latent_dim
    '''
    def __init__(self, bottleneck: int, input_size: int, count_classes: int, ohe_latent_dim: int = 10):
        assert bottleneck > 0
        assert input_size > 0
        assert count_classes > 0
        assert ohe_latent_dim > 0

        self.bottleneck = bottleneck
        self.input_size = input_size
        self.count_classes = count_classes

        super(VAE, self).__init__()
        self.ohe_latent_size = ohe_latent_dim

        self.pre_encoder = Sequential(
            #BatchSwapNoise(0.15),
            Linear(input_size, 800),
            BatchNorm1d(800),
            MishLayer(), #LeakyReLU(0.2),
            Linear(800, 600, bias=False),
            MishLayer()
            #BatchNorm1d(400),
            #MishLayer(),
            #Linear(400, 600),
            #BatchNorm1d(600),
            #MishLayer(),
        )

        self.fc_mu = Linear(600, bottleneck, bias=False)
        self.fc_sigma = Linear(600, bottleneck, bias=False)

        self.decoder = Sequential(
            MishLayer(),
            Linear(bottleneck+self.ohe_latent_size,
                   input_size
                   )
        )

        self.ohe_pipeline = Sequential(
            Linear(count_classes, self.ohe_latent_size),
            BatchNorm1d(self.ohe_latent_size),
            MishLayer(), #LeakyReLU(0.2),
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
        return self.fc_mu(h), self.fc_sigma(h)

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
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    def latents(self, x, y):
        mu, logvar = self.encode(x, y)
        return self.reparameterize(mu, logvar)

