from torch.nn import Module
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