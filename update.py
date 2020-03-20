import pdb
from itertools import combinations
from ignite.engine import _prepare_batch
from torch import mean, exp, unique, cat, isnan
from torch import norm as torch_norm
#from torch import save as torch_save
from torch.nn import NLLLoss, Linear, CrossEntropyLoss, MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.optim import Adam

from model import RAdam
from ranger import Ranger
from model.losses import get_variational_loss, mmd_criterion
from data.utils import create_random_classes, add_noise




class VAEUpdater:
    '''

    '''
    def __init__(self, vae_model, disc, cfg, device=None, cuda=False):
        self.cfg = cfg
        self.noise_beta = self.cfg.noise_beta

        self.model = vae_model
        self.model_linear_params = cat([x.view(-1) for x in self.model.parameters()])#for l1/l2 reg
        self.latent_discrim = disc

        self.optimizer = Ranger(self.model.parameters(), lr=cfg.vae_lr, weight_decay=cfg.vae_decay)
        self.optimizer_discrim = Ranger(self.latent_discrim.parameters(), lr=cfg.disc_lr, weight_decay=cfg.disc_decay)
        self.model_scheduler =  CosineAnnealingLR(self.optimizer, 5) #ExponentialLR(self.optimizer, gamma=cfg.model_scheduler_gamma)
        self.latent_discrim_scheduler = ExponentialLR(self.optimizer_discrim, gamma=cfg.discr_scheduler_gamma)

        self.loss_function = get_variational_loss(self.cfg.vae_beta, 1.)
        self.discrim_loss = NLLLoss()

        self.form_consistency_loss = CrossEntropyLoss()

        self.device = device
        self.cuda = cuda

    def __call__(self, engine, batch):
        self.noise_beta = self.noise_beta * self.cfg.decay_beta

        data, ohe = _prepare_batch(batch, device=self.device)
        #Autoencoder training
        if engine.iteration % 2 == 0:
            #pdb.set_trace()
            self.model.train()
            self.latent_discrim.eval()

            self.optimizer.zero_grad()

            #encoding expression batch with initial classes and evaluating the reconstruction loss
            a2b_mu, a2b_logvar, a2b_form = self.model.encode(data, ohe)
            a2b_latents = self.model.reparameterize(a2b_mu, a2b_logvar)
            a2a_expression, mmd_values = self.model.decode(a2b_latents, ohe)
            reconstruction_loss_on_batch = self.loss_function(a2a_expression, data, a2b_mu, a2b_logvar)

            #mmd loss
            av_mmd_loss = 0.
            batch_indices = ohe.argmax(1)
            unique_classes = unique(batch_indices)
            count_comb = 0
            for feature1, feature2 in combinations(unique_classes, 2):
                count_comb += 1
                x1 = mmd_values[batch_indices == feature1]
                x2 = mmd_values[batch_indices == feature2]
                min_len = min(x1.shape[0], x2.shape[0])
                #pdb.set_trace()
                av_mmd_loss += mmd_criterion(x1[:min_len], x2[:min_len], self.cuda, mu=self.cfg.kernel_mu, sigma=1e-6).mean()
            av_mmd_loss /= count_comb

            #first style transfer
            transfer_classes = create_random_classes(data.size(0), ohe.shape[1])
            if self.cuda:
                transfer_classes = transfer_classes.cuda()
            a2b_expression = self.model.decode(a2b_latents, transfer_classes)[0]
            #second style transfer
            b2a_mu, b2a_logvar, b2a_form = self.model.encode(a2b_expression, ohe)
            b2a_latents = self.model.reparameterize(b2a_mu, b2a_logvar)
            b2a_expression = self.model.decode(b2a_latents, ohe)[0]
            cyclic_loss_on_batch = self.loss_function.reconstruction_loss(b2a_expression, data)
            
            #form consistency loss
            #form_consistency_loss_on_batch = self.form_consistency_loss(b2a_form, ohe.argmax(1))
            #form_consistency_loss_on_batch = self.form_consistency_loss(b2a_form, transfer_classes.argmax(1))
            #form_consistency_loss_on_batch = self.form_consistency_loss(a2b_form, ohe.argmax(1))
            a2a_mu, a2a_logvar, a2a_form = self.model.encode(a2a_expression, ohe)
            a2a_latents = self.model.reparameterize(a2a_mu, a2a_logvar)
            
            form_consistency_loss_on_batch = self.form_consistency_loss(a2b_form, ohe.argmax(1)) + self.form_consistency_loss(b2a_form, transfer_classes.argmax(1)) + MSELoss()(a2a_latents, b2a_latents)

            #adversarial part
            mu, logvar, form_ = self.model.encode(data, ohe)
            latents = self.model.reparameterize(mu, logvar)
            latents = self.model.reparameterize(mu, logvar)
            #next_layers_out = self.model.decode(latents, ohe)[1]

            #adv_loss_g = discrim_loss(latent_discrim(latents), ohe.argmax(1))
            #loss-=20.*adv_loss_g

            noisy_latents = add_noise(latents, self.noise_beta)
            #noisy_latents = add_noise(next_layers_out)
            discrim_preds = self.latent_discrim(noisy_latents)
            #shannon entropy
            adv_loss_g = mean(discrim_preds * exp(discrim_preds))

            #weight regularization
            l1_regularization = torch_norm(self.model_linear_params, 1)

            loss = reconstruction_loss_on_batch + self.cfg.cyclic_weight*cyclic_loss_on_batch + \
                    self.cfg.adv_weight*adv_loss_g - self.cfg.mmd_weight*av_mmd_loss + \
                    self.cfg.l1_weight*l1_regularization +self.cfg.form_consistency_weight*form_consistency_loss_on_batch
            #print('Cyclic:', cyclic_loss_on_batch.item())
            #print('Sum:', loss.item())
            loss.backward()
            engine.state.adversarial_loss = loss.item()
            #Clip gradient
            clip_grad_norm_(self.model.parameters(), self.cfg.clip_value)
            self.optimizer.step()

        elif engine.iteration % 2 == 1:
            #discriminator learning part
            self.model.eval()
            self.latent_discrim.train()

            self.optimizer_discrim.zero_grad()
            mu, logvar, form_ = self.model.encode(data, ohe)
            latents = self.model.reparameterize(mu, logvar)
            noisy_latents = add_noise(latents, self.noise_beta)
            adv_loss_d = self.discrim_loss(self.latent_discrim(noisy_latents.detach()), ohe.argmax(1))

            adv_loss_d.backward()
            clip_grad_norm_(self.latent_discrim.parameters(), self.cfg.clip_value)
            self.optimizer_discrim.step()
#            self.latent_discrim_scheduler.step(adv_loss_d.item())
#        if engine.iteration > 4500:
#            self.model_scheduler.step()
#        self.model_scheduler.step()
#        self.latent_discrim_scheduler.step()
        engine.iteration += 1


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
        if self.latent_discrim:
            self.latent_discrim.eval()

        val_expression_tensor, val_class_ohe_tensor = _prepare_batch(batch, device=self.device)

        recon_val, mu_val, logvar_val = self.model(val_expression_tensor, val_class_ohe_tensor)
        mmd_values = None
        if isinstance(recon_val, tuple):
            mmd_values = recon_val[1]
            recon_val = recon_val[0]#only reconstr values, without mmd values
        latents_val = self.model.reparameterize(mu_val, logvar_val)
        reconstruction_loss_val = self.loss_function.reconstruction_loss(recon_val, val_expression_tensor)

        av_mmd_loss = 0.
        batch_indices = val_class_ohe_tensor.argmax(1)
        unique_classes = unique(batch_indices)
        count_comb = 0
        for feature1, feature2 in combinations(unique_classes, 2):
            x1 = mmd_values[batch_indices == feature1]
            x2 = mmd_values[batch_indices == feature2]
            min_len = min(x1.shape[0], x2.shape[0])
            av_mmd_loss += mmd_criterion(x1[:min_len], x2[:min_len], self.cuda).mean()
            count_comb += 1
        av_mmd_loss /= count_comb

        engine.state.losses = {}
        if self.latent_discrim:
            discrim_preds_val = self.latent_discrim(latents_val)
            discrim_loss_val = self.discrim_loss(discrim_preds_val.detach(), val_class_ohe_tensor.argmax(1))
            discrim_val_acc = (val_class_ohe_tensor.argmax(1) == discrim_preds_val.argmax(1)).float().mean()

            #first style transfer
            transfer_classes_val = create_random_classes(val_expression_tensor.shape[0],
                                                                    val_class_ohe_tensor.shape[1])
            if self.cuda:
                transfer_classes_val = transfer_classes_val.cuda()
            a2b_expression = self.model.decode(latents_val, transfer_classes_val)[0]
            #second style transfer
            b2a_mu, b2a_logvar, form_ = self.model.encode(a2b_expression, transfer_classes_val)
            b2a_latents = self.model.reparameterize(b2a_mu, b2a_logvar)
            b2a_expression = self.model.decode(b2a_latents, val_class_ohe_tensor)[0]
            cyclic_loss_val = self.loss_function.reconstruction_loss(b2a_expression, val_expression_tensor)
            engine.state.losses['discrim loss'] = discrim_loss_val.item()
            engine.state.losses['cyclic'] = cyclic_loss_val.item()
            engine.state.losses['discrim accuracy'] = discrim_val_acc if isinstance(discrim_val_acc, float) else discrim_val_acc.item()

        engine.state.losses['reconstruction'] = reconstruction_loss_val.item()
        engine.state.losses['mmd'] = av_mmd_loss.item() if not isinstance(av_mmd_loss, float) else av_mmd_loss


class trVAEUpdater:
    def __init__(self, model, cfg, device=None, cuda=False):
        self.model = model
        self.cfg = cfg
        self.cuda = cuda
        self.optimizer = Adam(self.model.parameters(), lr = self.cfg.lr)
        self.device = device
        self.vae_loss = get_variational_loss(cfg.vae_beta, cfg.rec_coef)

    def __call__(self, engine, batch):
        expression, ohe = _prepare_batch(batch, device=self.device)
        batch_indices = ohe.argmax(1).float().view(-1, 1)

        mu, logvar = self.model.encode(expression, batch_indices)
        latents = self.model.reparameterize(mu, logvar)
        reconstructed, mmd_values = self.model.decode(latents, batch_indices)

        vae_loss = self.vae_loss(reconstructed, expression, mu, logvar)

        av_mmd_loss = 0.
        unique_classes = unique(batch_indices)
        count_comb = 0
        #pdb.set_trace()
        for feature1, feature2 in combinations(unique_classes, 2):
           count_comb += 1
           x1 = mmd_values[(batch_indices == feature1).reshape((-1,))]
           x2 = mmd_values[(batch_indices == feature2).reshape((-1,))]
           min_len = min(x1.shape[0], x2.shape[0])
           av_mmd_loss += mmd_criterion(x1[:min_len], x2[:min_len], self.cuda,
                                        mu=self.cfg.kernel_mu, sigma=1e-6).mean()
        av_mmd_loss /= count_comb
        #pdb.set_trace()

        loss_on_batch = vae_loss + self.cfg.mmd_weight * av_mmd_loss

        self.optimizer.zero_grad()
        loss_on_batch.backward()
        self.optimizer.step()

