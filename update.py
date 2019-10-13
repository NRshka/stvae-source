from ignite.engine import _prepare_batch
from torch import mean, exp
#from torch import save as torch_save
from torch.nn import NLLLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR

from model import RAdam
from model.losses import get_variational_loss
from data.utils import create_random_classes, add_noise


class VAEUpdater:
    '''

    '''
    def __init__(self, vae_model, disc, cfg, device=None, train=True, cuda=False):
        self.cfg = cfg
        self.noise_beta = self.cfg.noise_beta

        self.model = vae_model
        self.latent_discrim = disc

        self.optimizer = RAdam(self.model.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_decay)
        self.optimizer_discrim = RAdam(self.latent_discrim.parameters(), lr=cfg.disc_lr, weight_decay=cfg.disc_decay)
        self.model_scheduler =  ExponentialLR(self.optimizer, gamma=0.992)
        self.latent_discrim_scheduler = ExponentialLR(self.optimizer_discrim, gamma=0.992)

        self.loss_function = get_variational_loss(self.cfg.vae_beta)
        self.discrim_loss = NLLLoss()

        self.device = device
        self.cuda = cuda


    def __call__(self, engine, batch):
        engine.iteration += 1
        self.noise_beta = self.noise_beta * self.cfg.decay_beta

        data, ohe = _prepare_batch(batch, device=self.device)
        #Autoencoder training
        if engine.iteration % 2 == 0:
            self.model.train()
            self.latent_discrim.eval()

            self.optimizer.zero_grad()

            #encoding expression batch with initial classes and evaluating the reconstruction loss
            a2b_mu, a2b_logvar = self.model.encode(data, ohe)
            a2b_latents = self.model.reparameterize(a2b_mu, a2b_logvar)
            a2a_expression = self.model.decode(a2b_latents, ohe)
            reconstruction_loss_on_batch = self.loss_function(a2a_expression, data, a2b_mu, a2b_logvar)

            #first style transfer
            transfer_classes = create_random_classes(data.size()[0], ohe.shape[1])
            if self.cuda:
                transfer_classes = transfer_classes.cuda()
            a2b_expression = self.model.decode(a2b_latents, transfer_classes)
            #second style transfer
            b2a_mu, b2a_logvar = self.model.encode(a2b_expression, transfer_classes)
            b2a_latents = self.model.reparameterize(b2a_mu, b2a_logvar)
            b2a_expression = self.model.decode(b2a_latents, ohe)
            cyclic_loss_on_batch = self.loss_function.reconstruction_loss(b2a_expression, data)

            #adversarial part
            mu, logvar = self.model.encode(data, ohe)
            latents = self.model.reparameterize(mu, logvar)
            #adv_loss_g = discrim_loss(latent_discrim(latents), ohe.argmax(1))
            #loss-=20.*adv_loss_g
            noisy_latents = add_noise(latents, self.noise_beta)
            discrim_preds = self.latent_discrim(noisy_latents)
            #shannon entropy
            adv_loss_g = mean(discrim_preds * exp(discrim_preds))

            loss = reconstruction_loss_on_batch + self.cfg.cyclic_weight*cyclic_loss_on_batch + self.cfg.adv_weight*adv_loss_g

            loss.backward()
            engine.state.adversarial_loss = loss.item()
            #Clip gradient
            clip_grad_norm_(self.model.parameters(), self.cfg.clip_value)
            self.optimizer.step()
            self.model_scheduler.step()
        
        elif engine.iteration % 2 == 1:
            #discriminator learning part
            self.model.eval()
            self.latent_discrim.train()              

            self.optimizer_discrim.zero_grad()
            mu, logvar = self.model.encode(data, ohe)
            latents = self.model.reparameterize(mu, logvar)
            noisy_latents = add_noise(latents, self.noise_beta)
            adv_loss_d = self.discrim_loss(self.latent_discrim(noisy_latents.detach()), ohe.argmax(1))

            adv_loss_d.backward()
            clip_grad_norm_(self.latent_discrim.parameters(), self.cfg.clip_value)
            self.optimizer_discrim.step()
            self.latent_discrim_scheduler.step()


class Validator:
    def __init__(self, model, discriminator, vae_beta, cuda=False, device=None):
        self.loss_function = get_variational_loss(vae_beta)
        self.model = model
        self.latent_discrim = discriminator
        self.cuda = cuda
        self.device = device
        self.discrim_loss = NLLLoss()


    def __call__(self, engine, batch):
        self.model.eval()
        self.latent_discrim.eval()

        val_expression_tensor, val_class_ohe_tensor = _prepare_batch(batch, device=self.device)

        recon_val, mu_val, logvar_val = self.model(val_expression_tensor, val_class_ohe_tensor)
        latents_val = self.model.reparameterize(mu_val, logvar_val)
        reconstruction_loss_val = self.loss_function.reconstruction_loss(recon_val, val_expression_tensor)

        discrim_preds_val = self.latent_discrim(latents_val)
        discrim_loss_val = self.discrim_loss(discrim_preds_val.detach(), val_class_ohe_tensor.argmax(1))
        discrim_val_acc = (val_class_ohe_tensor.argmax(1) == discrim_preds_val.argmax(1)).float().mean()

        #first style transfer
        transfer_classes_val = create_random_classes(val_expression_tensor.shape[0],
                                                                 val_class_ohe_tensor.shape[1]).cuda()
        a2b_expression = self.model.decode(latents_val, transfer_classes_val)
        #second style transfer
        b2a_mu, b2a_logvar = self.model.encode(a2b_expression, transfer_classes_val)
        b2a_latents = self.model.reparameterize(b2a_mu, b2a_logvar)
        b2a_expression = self.model.decode(b2a_latents, val_class_ohe_tensor)
        cyclic_loss_val = self.loss_function.reconstruction_loss(b2a_expression, val_expression_tensor)

        engine.state.losses = {}
        engine.state.losses['reconstruction'] = reconstruction_loss_val.item()
        engine.state.losses['cyclic'] = cyclic_loss_val.item()
        engine.state.losses['discrim'] = discrim_loss_val.item()
