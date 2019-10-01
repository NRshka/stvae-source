'''
Процесс обучения включает в себя как алгоритм, так и гиперпараметры, контролирующие его.
Поэтому уместно описать обучение с помощью класса.
'''
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


class VAELearn:
    def __init__(self, vae_model, disc_model, noise_beta: float, 
                 decay_beta: float = 1.0, ae_lr=1e-4, disc_lr=1e-4,
                 vae_beta: float = 1e-4, save_dir: str = None):
        assert noise_beta >= 0, ValueError("Noise beta decay can't be negative")
        assert decay_beta >= 0, ValueError("Beta decay can't be negative")
        
        self.current_best_loss = 1e18
        self.noise_beta = noise_beta
        self.decay_beta = decay_beta
        
        assert isinstance(vae_model, VAE)
        
        self.model = vae_model
        self.latent_discrim = disc_model
        
        self.optimizer = RAdam(self.model.parameters(), lr=ae_lr, weight_decay=1e-3)
        self.optimizer_discrim = RAdam(self.latent_discrim.parameters(), lr=disc_lr)
        self.model_scheduler =  ExponentialLR(self.optimizer, gamma=0.992)
        self.latent_discrim_scheduler = ExponentialLR(self.optimizer_discrim, gamma=0.992)
        
        assert isinstance(vae_beta, (float, int)), TypeError("vae_beta param must be int or float")
        self.loss_function = get_variational_loss(vae_beta)
        self.discrim_loss = NLLLoss()
        
        if save_dir:
            assert os.path.isdir(save_dir), ValueError(f"{save_dir} directory does't exists.")
        self.save_dir = save_dir
    
    
    def train_with_reconstruction_and_cycle_loss(self, epoch, dataloader, 
                                                 val_expression_tensor,
                                                 val_class_ohe_tensor,
                                                 cyclic_weight: float = 0.2, 
                                                 adv_weight: float = 0.07, 
                                                 clip_value: float = 1.,
                                                 log_interval: int = 1,
                                                 verbose: str = "silent"):
        train_loss = 0
        noise_beta = self.noise_beta * self.decay_beta
        val_class_ohe_np = val_class_ohe_tensor.cpu().detach().numpy()
        for batch_idx, (data, ohe) in enumerate(dataloader):
            data = data.cuda()
            ohe = ohe.cuda()
            #Autoencoder training
            if batch_idx % 2 == 0:
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
                
                if batch_idx % log_interval == 0:
                    self.model.eval()
                    self.latent_discrim.eval()

                    recon_val, mu_val, logvar_val = self.model(val_expression_tensor, val_class_ohe_tensor)
                    latents_val = self.model.reparameterize(mu_val, logvar_val)
                    reconstruction_loss_val = self.loss_function.reconstruction_loss(recon_val, val_expression_tensor)

                    discrim_preds_val = self.latent_discrim(latents_val)
                    discrim_loss_val = self.discrim_loss(discrim_preds_val.detach(), val_class_ohe_tensor.argmax(1))
                    discrim_preds_val_np = discrim_preds_val.cpu().detach().numpy()
                    discrim_val_acc = (val_class_ohe_np.argmax(1) == discrim_preds_val_np.argmax(1)).mean()

                    #first style transfer
                    transfer_classes_val = create_random_classes(val_expression_tensor.shape[0],
                                                                 val_class_ohe_tensor.shape[1]).cuda()
                    a2b_expression = self.model.decode(latents_val, transfer_classes_val)
                    #second style transfer
                    b2a_mu, b2a_logvar = self.model.encode(a2b_expression, transfer_classes_val)
                    b2a_latents = self.model.reparameterize(b2a_mu, b2a_logvar)
                    b2a_expression = self.model.decode(b2a_latents, val_class_ohe_tensor)
                    cyclic_loss_val = self.loss_function.reconstruction_loss(b2a_expression, val_expression_tensor)


                    if verbose == "stdout" or verbose == "all":
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Reconstruction loss: {:.6f} \tCyclic loss: {:.6f} \tDiscrimLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(dataloader.dataset),
                            100. * batch_idx / len(dataloader),
                            reconstruction_loss_val.item(),
                            cyclic_loss_val.item(),
                            discrim_loss_val.item()))

                    if self.save_dir:
                        if cyclic_loss_val.item() < current_best_loss:
                            current_best_loss = cyclic_loss_val.item()
                            print('Saving model with loss %s' % current_best_loss)
                            torch_save(self.model.state_dict(), os.path.join(self.save_dir, 'checkpoint_VAE.pkl'))
                            torch_save(self.latent_discrim.state_dict(), os.path.join(self.save_dir, 'checkpoint_VAE_discrim.pkl'))



            #discriminator training
            elif batch_idx % 2 == 1:
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


        #self.model_scheduler.step()
        #self.latent_discrim_scheduler.step()