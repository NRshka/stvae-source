from pathlib import Path
import sys
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from numpy import inf
from tensorboardX import SummaryWriter
import torch
from collections import defaultdict

from utils import save_weights, load_datasets
from experiment import Experiment
from config import Config
from update import VAEUpdater, Validator
from model.autoencoders import VAE
from model.modules import Latent_discriminator


THIS_DIR = Path('./').absolute()
EXPERIMENTS_DIR = THIS_DIR.joinpath('experiment')


def create_update_class(model, discriminator, config):
    is_cuda = torch.cuda.is_available()
    device = None

    if is_cuda:
        device = torch.cuda.device(torch.cuda.current_device())
        device = None

    training_upd = VAEUpdater(model, discriminator, config, device=device,
                              train=True, cuda=torch.cuda.is_available())
    validator_upd = Validator(model, discriminator, config.vae_beta,
                              device=device,
                              cuda=torch.cuda.is_available())

    return training_upd, validator_upd


def create_model(cfg):
    vae_model = VAE(cfg.bottleneck, cfg.input_dim, cfg.count_classes)
    disc_model = Latent_discriminator(cfg.bottleneck, cfg.count_classes)
    
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


def train(dataloader_train, dataloader_val, cfg):
    with Experiment(EXPERIMENTS_DIR, cfg) as exp:
        print(f'Experiment started: {exp.experiment_id}')

        model, disc = create_model(cfg)

        update_class_train, update_class_val = create_update_class(model, disc, cfg)

        trainer = Engine(update_class_train)
        validator = Engine(update_class_val)
        
        trainer.iteration = 0

        best_loss = inf
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            nonlocal best_loss

            validator_state = validator.run(dataloader_val)

            log_progress(trainer.state.epoch, trainer.state.iteration, validator.state.losses, 'val', tensorboard_writer)

            if validator_state.losses['cyclic'] < best_loss:
                best_loss = validator_state.losses['cyclic']
                save_weights(model, exp.experiment_dir.joinpath('best_vae.pth'))
                save_weights(disc, exp.experiment_dir.joinpath('best_disc.pth'))
        

        _TENSORBOARD_DIR = cfg.experiment_dir.joinpath('log')
        tensorboard_writer = SummaryWriter(str(_TENSORBOARD_DIR))

        trainer.run(dataloader_train, max_epochs=cfg.epochs)

        print(f'Experiment {exp.experiment_id} has finished')

    return model, disc


if __name__ == '__main__':
    cfg = Config()
    dataloader_train, dataloader_val = load_datasets(cfg)
    train(dataloader_train, dataloader_val, cfg)
