from torch.nn import MSELoss


def get_variational_loss():
    def loss_function(recon_x, x, mu, logvar):
        '''
        Reconstruction + KL divergence losses summed over all elements and batch.
        '''
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        rec_loss_ = loss_function.reconstruction_loss(recon_x, x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return rec_loss_ + vae_beta*KLD
    
    loss_function.reconstruction_loss = nn.MSELoss(reduction='mean')
    
    return loss_function
