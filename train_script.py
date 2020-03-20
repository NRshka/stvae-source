from pathlib import Path
import sys
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from numpy import inf
from tensorboardX import SummaryWriter
import torch
from collections import defaultdict
import pdb

from utils import save_weights, load_datasets
from experiment import Experiment
from config import Config
from update import VAEUpdater, Validator
from model.autoencoders import VAE
from model.modules import Latent_discriminator


THIS_DIR = Path('./').absolute()
EXPERIMENTS_DIR = THIS_DIR.joinpath('experiment')


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight, 1e-2)
        m.bias.data.fill_(0.01)


def create_update_class(model, discriminator, config):
    is_cuda = torch.cuda.is_available()
    device = None

    if is_cuda:
        device = torch.cuda.device(torch.cuda.current_device())
        device = None

    training_upd = VAEUpdater(model, discriminator, config, device=device,
                              cuda=torch.cuda.is_available())
    validator_upd = Validator(model, discriminator, config.vae_beta,
                              device=device,
                              cuda=torch.cuda.is_available())

    return training_upd, validator_upd


def create_model(cfg):
    vae_model = VAE(cfg.bottleneck, cfg.input_dim, cfg.count_classes, cfg.n_layers, cfg.scale_alpha)
    init_weights(vae_model)
    disc_model = Latent_discriminator(cfg.bottleneck, cfg.count_classes)
    init_weights(disc_model)

    if torch.cuda.is_available():
        vae_model = vae_model.cuda()
        disc_model = disc_model.cuda()

    return vae_model, disc_model


def log_progress(epoch, iteration, losses, mode='train', tensorboard_writer=None, use_iteration=False):
    if not use_iteration:
        losses_str = [
            f'{name}: {val:.3f}'
            for name, val in losses.items()
        ]
        losses_str = ' | '.join(losses_str)

        epoch_str = f'Epoch [{epoch}|{iteration}] {mode}'

        print(f'{epoch_str:<25}{losses_str}')

    for name, val in losses.items():
        tensorboard_writer.add_scalar(f'{name}', val, epoch if not use_iteration else iteration)


def train(dataloader_train, dataloader_val, cfg,
          model=None, disc=None,
          updaters=None, random_seed=None):
    with Experiment(EXPERIMENTS_DIR, cfg, random_seed=random_seed) as exp:
        print(f'Experiment started: {exp.experiment_id}')

        if not model and not disc:
            model, disc = create_model(cfg)

        if not updaters:
            update_class_train, update_class_val = create_update_class(model, disc, cfg)
        elif isinstance(updaters, (tuple, list)):
            update_class_train = updaters[0]
            update_class_val = updaters[1]
        elif isinstance(updaters, dict):
            update_class_train = updaters['train']
            update_class_val = updaters['validation']
        else:
            update_class_train = updaters
            update_class_val = Validator(model, disc, cfg, cuda=cfg.use_cuda)


        trainer = Engine(update_class_train)
        validator = Engine(update_class_val)

        trainer.iteration = 0

        best_loss = inf
        last_cyclic_loss = inf
        count_increas_losses = 0

        if cfg.verbose != 'silent':
            @trainer.on(Events.EPOCH_COMPLETED)
            def log_training_results(engine):
                nonlocal best_loss
                nonlocal last_cyclic_loss
                nonlocal count_increas_losses

                #trainer._process_function.model_scheduler.step()
                #trainer._process_function.latent_discrim_scheduler.step()

                validator_state = validator.run(dataloader_val)

                log_progress(trainer.state.epoch, trainer.state.iteration, validator.state.losses, 'val', tensorboard_writer)

                try:
                    #early stopping
                    if last_cyclic_loss - validator_state.losses['cyclic'] < 0.0018:
                        count_increas_losses += 1
                    else:
                        count_increas_losses = 0
                    last_cyclic_loss = validator_state.losses['cyclic']

                    if count_increas_losses >= cfg.early_stop_epochs:
                        print("=========Early stopping: loss doesn't decreases",
                            f"{cfg.early_stop_epochs} epochs=========")
                        #engine.terminate()


                    if validator_state.losses['cyclic'] < best_loss:
                        best_loss = validator_state.losses['cyclic']
                        try:
                            pass
                            #save_weights(model, exp.experiment_dir.joinpath('best_vae.pth'))
                            #save_weights(disc, exp.experiment_dir.joinpath('best_disc.pth'))
                        except Exception as e:
                            print(e)
                except:
                    # if no cyclic loss or smth
                    pass

        _TENSORBOARD_DIR = exp.experiment_dir.joinpath('log')
        tensorboard_writer = SummaryWriter(str(_TENSORBOARD_DIR))

        trainer.run(dataloader_train, max_epochs=cfg.epochs)

        print(f'Experiment {exp.experiment_id} has finished')

    return model, disc


if __name__ == '__main__':
    cfg = Config()
    dataloader_train, dataloader_val = load_datasets(cfg)
    train(dataloader_train, dataloader_val, cfg)
