from torch.nn import Module, Sequential, Linear, BatchNorm1d
from torch.nn import LeakyReLU, LogSoftmax
from torch import rand, add, floor
from torch import LongTensor

class BatchSwapNoise(Module):
    '''
    Swap Noise module
    
    @param p: float, probability
    '''

    def __init__(self, p):
        super().__init__()
        assert p >= 0. and p <= 1., ValueError("The probability value should be in between 0 and 1.")
        self.p = p

    def forward(self, x):
        if self.training:
            mask = rand(x.size()) > (1 - self.p)
            idx = add(torch.arange(x.nelement()),
                            (floor(rand(x.size()) * x.size(0)).type(LongTensor) *
                             (mask.type(LongTensor) * x.size(1))).view(-1))
            idx[idx>=x.nelement()] = idx[idx>=x.nelement()]-x.nelement()
            return x.view(-1)[idx].view(x.size())
        else:
            return x


class Latent_discriminator(Module):
     def __init__(self, bottleneck: int, ohe_latent_dim: int):
        assert bottleneck > 0, ValueError(f"A bottleneck dimension can't be negative: {bottleneck}")
        assert isinstance(bottleneck, int), TypeError("INT!")
        assert ohe_latent_dim > 0, ValueError(f"A dimension can't be negative: {ohe_latent_dim}")
        assert isinstance(ohe_latent_dim, int), TypeError("INT!")
        
        super(Latent_discriminator,self).__init__()
        
        self.model = Sequential(
            Linear(bottleneck, 1024, bias=False),
            BatchNorm1d(1024),
            LeakyReLU(0.2, inplace=True),
            Linear(1024, 1024, bias=False),
            BatchNorm1d(1024),
            LeakyReLU(0.2, inplace=True),
            Linear(1024, ohe_latent_dim, bias=False),
            LogSoftmax(dim=-1))
    
    
     def forward(self,x):
        return self.model(x)