import torch
import trvaep
from typing import List


class trVAE(trvaep.CVAE):
    def __init__(self, input_dim: float, num_classes: int,
                 latent_dim: int = 10,
                 encoder_layer_sizes: List[int] = [64, 32],
                 decoder_layer_sizes: List[int] = [32, 64],
                 alpha: float = 1e-3):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        super(trVAE, self).__init__(input_dim,
                                          num_classes=num_classes,
                                          encoder_layer_sizes=encoder_layer_sizes,
                                          decoder_layer_sizes=decoder_layer_sizes,
                                          latent_dim=latent_dim,
                                          alpha=alpha,
                                          use_mmd=True)


    def forward(self, x, y):
        if len(y.shape) > 1 and y.shape[-1] == self.num_classes:
            y = y.argmax(dim=1)
        return super(trVAE, self).forward(x, y)

    def latents(self, x, y):
        if len(y.shape) > 1 and y.shape[-1] == self.num_classes:
            y = y.argmax(dim=1)
        return self.get_latent(x, y, mean=False)

