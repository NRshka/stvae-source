from torch import(
                  mean, randn, cuda, 
                  transpose, matmul, exp
                )
from torch import sum as tsum
from torch.nn import MSELoss
from functools import partial
from .mmd import mix_rbf_mmd2


def pairwise_distance(x, y):
    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = transpose(y, 0, 1)
    output = tsum((x - y) ** 2, 1)
    output = transpose(output, 0, 1)

    return output


def gaussian_kernel_matrix(x, y, gamma):
    '''
        Do the gaussian kernel-based estimation of distance
        between distributions over x and y variables
        :param sigmas: sigmas is the hyperparameter sets prior to the kernel computation
    '''
    #sigmas = sigmas.view(sigmas.shape[0], 1)
    #gamma = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = matmul(gamma, dist_)

    return tsum(exp(-s), 0).view_as(dist)


def maximum_mean_discrepancy(x, y, kernel):
    value = kernel(x, x) - 2 * kernel(x, y) + kernel(y, y)
    return value


def mmd_criterion(x, y, use_cuda=True, mu=1, sigma=1e-6):
    '''sigmas = randn(x.shape[0], 1)*sigma + mu
    if use_cuda and cuda.is_available:
        sigmas = sigmas.cuda()

    kernel = partial(gaussian_kernel_matrix, gamma=sigmas)
    loss = maximum_mean_discrepancy(x, y, kernel)
    '''
    loss = mix_rbf_mmd2(x, y, [1e3, 2e3, 4e3, 8e3, 16e3])

    return loss


class VariationalLoss:
    def __init__(self, vae_beta: float, rec_coef: float = 1.):
        self.reconstruction_loss = MSELoss(reduction='mean')
        self.vae_beta = vae_beta
        self.rec_coef = rec_coef

    def __call__(self, recon_x, x, mu, logvar):
        '''
        Reconstruction + KL divergence losses summed over all elements and batch.
        '''
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        rec_loss_ = self.reconstruction_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * mean(1 + logvar - mu.pow(2) - logvar.exp())

        return self.rec_coef*rec_loss_ + self.vae_beta*KLD

