from ignite.engine import _prepare_batch
from torch import mean, exp
from torch import save as torch_save
from torch.nn import NLLLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR

import os

from .model.autoencoders import VAE
from .model.modules import RAdam
from .model.losses import get_variational_loss
from .data.utils import create_random_classes, add_noise


class VAEUpdater:
    '''

    '''
    def __init__(self, cfg, vae_model, disc, device=None):
        self.cfg = cfg
        self.noise_beta = self.cfg.noise_beta

        self.model = vae_model
        self.latent_discrim = disc

        self.optimizer = RAdam(self.model.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_decay)
        self.optimizer_discrim = RAdam(self.latent_discrim.parameters(), lr=cfg.disc_lr, weight_decay=cfg.disc_decay)
        self.model_scheduler =  ExponentialLR(self.optimizer, gamma=0.992)
        self.latent_discrim_scheduler = ExponentialLR(self.optimizer_discrim, gamma=0.992)

        self.loss_function = get_variational_loss(vae_beta)
        self.discrim_loss = NLLLoss()

        self.device = device


    def step(self, engine, batch):
        self.noise_beta = self.noise_beta * self.cfg.decay_beta

        data, ohe = _prepare_batch(batch, device=self.device)
        #Autoencoder training
        if self.batch_idx % 2 == 0:
            self.model.train()
            self.latent_discrim.eval()

            self.optimizer.zero_grad()

            #encoding expression batch with initial classes and evaluating the reconstruction loss
            a2b_mu, a2b_logvar = self.model.encode(data, ohe)
            a2b_latents = self.model.reparameterize(a2b_mu, a2b_logvar)
            a2a_expression = self.model.decode(a2b_latents, ohe)
            reconstruction_loss_on_batch = self.loss_function(a2a_expression, data, a2b_mu, a2b_logvar)

            #first style transfer
            transfer_classes = create_random_classes(data.size()[0], ohe.shape[1]).cuda()
            a2b_expression = self.model.decode(a2b_latents, transfer_classes)
            second style transfer
            b2a_mu, b2a_logvar = self.model.encode(a2b_expression, transfer_classes)
            b2a_latents = self.model.reparameterize(b2a_mu, b2a_logvar)
            b2a_expression = self.model.decode(b2a_latents, ohe)
            cyclic_loss_on_batch = self.loss_function.reconstruction_loss(b2a_expression, data)

            #adversarial part
            mu, logvar = self.model.encode(data, ohe)
            latents = self.model.reparameterize(mu, logvar)
            #adv_loss_g = discrim_loss(latent_discrim(latents), ohe.argmax(1))
            #loss-=20.*adv_loss_g
            noisy_latents = add_noise(latents, noise_beta)
            discrim_preds = self.latent_discrim(noisy_latents)
            #shannon entropy
            adv_loss_g = mean(discrim_preds * exp(discrim_preds))

            loss = reconstruction_loss_on_batch + cyclic_weight*cyclic_loss_on_batch + adv_weight*adv_loss_g

            loss.backward()
            train_loss += loss.item()
            #Clip gradient
            clip_grad_norm_(self.model.parameters(), clip_value)
            self.optimizer.step()
            self.model_scheduler.step()
        
        elif self.batch_idx % 2 == 1:
            #discriminator learning part
            self.model.eval()
            self.latent_discrim.train()              

            self.optimizer_discrim.zero_grad()
            mu, logvar = self.model.encode(data, ohe)
            latents = self.model.reparameterize(mu, logvar)
            noisy_latents = add_noise(latents, noise_beta)
            adv_loss_d = self.discrim_loss(self.latent_discrim(noisy_latents.detach()), ohe.argmax(1))

            adv_loss_d.backward()
            clip_grad_norm_(self.latent_discrim.parameters(), clip_value)
            self.optimizer_discrim.step()
            self.latent_discrim_scheduler.step()
