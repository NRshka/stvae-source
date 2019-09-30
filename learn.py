'''
Процесс обучения включает в себя как алгоритм, так и гиперпараметры, контролирующие его.
Поэтому уместно описать обучение с помощью класса.
'''

class VAELearn:
    def __init__(self, vae_model, disc_model, noise_beta: float, 
                 decay_beta: float = 1.0, ae_lr=1e-4, disc_lr=1e-4):
        assert noise_beta >= 0, ValueError("Noise beta decay can't be negative")
        assert decay_beta >= 0, ValueError("Beta decay can't be negative")
        self.current_best_loss = 1e18
        self.noise_beta = noise_beta
        self.decay_beta = decay_beta
        self.model = vae_model
        self.latent_discrim = disc_model
        self.optimizer = RAdam(model.parameters(), lr=ae_lr, weight_decay=1e-3)
        self.optimizer_discrim = RAdam(latent_discrim.parameters(), lr=disc_lr)
    
    
    def train_with_reconstruction_and_cycle_loss(epoch, dataloader):
        train_loss = 0
        noise_beta = self.noise_beta * self.decay_beta
        for batch_idx, (data, ohe) in enumerate(dataloader):
            data = data.cuda()
            ohe=ohe.cuda()
            #Autoencoder training
            if batch_idx%2==0:
                model.train()
                latent_discrim.eval()

                optimizer.zero_grad()


                #encoding expression batch with initial classes and evaluating the reconstruction loss
                a2b_mu, a2b_logvar = model.encode(data, ohe)
                a2b_latents = model.reparameterize(a2b_mu, a2b_logvar)
                a2a_expression = model.decode(a2b_latents, ohe)
                reconstruction_loss_on_batch = loss_function(a2a_expression, data, a2b_mu, a2b_logvar)

                #first style transfer
                transfer_classes = create_random_classes(batch_size, ohe.shape[1]).cuda()
                a2b_expression = model.decode(a2b_latents, transfer_classes)
                #second style transfer
                b2a_mu, b2a_logvar = model.encode(a2b_expression, transfer_classes)
                b2a_latents = model.reparameterize(b2a_mu, b2a_logvar)
                b2a_expression = model.decode(b2a_latents, ohe)
                cyclic_loss_on_batch = reconstruction_loss(b2a_expression, data)


                #adversarial part
                mu, logvar = model.encode(data, ohe)
                latents = model.reparameterize(mu, logvar)
                #adv_loss_g = discrim_loss(latent_discrim(latents), ohe.argmax(1))
                #loss-=20.*adv_loss_g
                noisy_latents = add_noise(latents, noise_beta)
                discrim_preds = latent_discrim(noisy_latents)
                #shannon entropy
                adv_loss_g = torch.mean(discrim_preds*torch.exp(discrim_preds))

                loss = reconstruction_loss_on_batch + cyclic_weight*cyclic_loss_on_batch + adv_weight*adv_loss_g

                loss.backward()
                train_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()
                if batch_idx % log_interval == 0:
                    model.eval()
                    latent_discrim.eval()


                    recon_val, mu_val, logvar_val = model(val_expression_tensor, val_class_ohe_tensor)
                    latents_val = model.reparameterize(mu_val, logvar_val)
                    reconstruction_loss_val = reconstruction_loss(recon_val,val_expression_tensor)


                    discrim_preds_val = latent_discrim(latents_val)
                    discrim_loss_val = discrim_loss(discrim_preds_val.detach(), val_class_ohe_tensor.argmax(1))
                    discrim_preds_val_np = discrim_preds_val.cpu().detach().numpy()
                    discrim_val_acc = (val_class_ohe.argmax(1)==discrim_preds_val_np.argmax(1)).mean()


                    #first style transfer
                    transfer_classes_val = create_random_classes(val_expression_tensor.shape[0],
                                                                 val_class_ohe_tensor.shape[1]).cuda()
                    a2b_expression = model.decode(latents_val, transfer_classes_val)
                    #second style transfer
                    b2a_mu, b2a_logvar = model.encode(a2b_expression, transfer_classes_val)
                    b2a_latents = model.reparameterize(b2a_mu, b2a_logvar)
                    b2a_expression = model.decode(b2a_latents, val_class_ohe_tensor)
                    cyclic_loss_val = reconstruction_loss(b2a_expression, val_expression_tensor)


                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Reconstruction loss: {:.6f} \tCyclic loss: {:.6f} \tDiscrimLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader),
                        reconstruction_loss_val.item(),
                        cyclic_loss_val.item(),
                        discrim_loss_val.item()))

                    #if reconstruction_loss_val.item() < current_best_loss:
                    if cyclic_loss_val.item() < current_best_loss:
                        current_best_loss = cyclic_loss_val.item()
                        print('Saving model with loss %s'%current_best_loss)
                        torch.save(model.state_dict(), 'checkpoint_VAE.pkl')
                        torch.save(latent_discrim.state_dict(), 'checkpoint_VAE_discrim.pkl')



            #discriminator training
            elif batch_idx%2==1:
                model.eval()
                latent_discrim.train()              

                optimizer_discrim.zero_grad()
                mu, logvar = model.encode(data, ohe)
                latents = model.reparameterize(mu, logvar)
                noisy_latents = add_noise(latents, noise_beta)
                adv_loss_d = discrim_loss(latent_discrim(noisy_latents.detach()), ohe.argmax(1))

                adv_loss_d.backward()
                torch.nn.utils.clip_grad_norm_(latent_discrim.parameters(), clip_value)
                optimizer_discrim.step()
            if batch_idx > 2:
                if watch_flag:
                    watcher.observe(
                        #reconstruction_loss=reconstruction_loss,
                            loss=loss,
                            d_loss=adv_loss_d,
                            #aae_g_loss=aae_g_loss,
                            #aae_d_loss=aae_d_loss
                           )


            #print('====> Epoch: {} Average loss: {:.4f}'.format(
            #      epoch, train_loss / len(dataloader.dataset)))
        model_scheduler.step()
        latent_discrim_scheduler.step()