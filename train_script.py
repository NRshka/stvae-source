from pathlib import Path
import sys
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from numpy import inf
from tensorboardX import SummaryWriter
import torch
from sklearn.model_selection import train_test_split
from collections import defaultdict

from data import get_raw_data
from experiment import Experiment
from config import Config
from update import VAEUpdater
from model.autoencoders import VAE
from model.modules import Latent_discriminator


THIS_DIR = Path('./').absolute()
EXPERIMENTS_DIR = THIS_DIR.joinpath('test_experiment')

cfg = Config()


def load_datasets(cfg):
    expr, class_ohe = get_raw_data(cfg.data_dir)
    train_expression, val_expression, train_class_ohe, val_class_ohe = train_test_split(
        expr, class_ohe, random_state=512, stratify = class_ohe.argmax(1), test_size=0.15
    )
    val_expression_tensor = torch.Tensor(val_expression)
    val_class_ohe_tensor = torch.Tensor(val_class_ohe)
    train_expression_tensor = torch.Tensor(train_expression)
    train_class_ohe = torch.Tensor(train_class_ohe)
    
    if torch.cuda.is_available():
        val_expression_tensor = val_expression_tensor.cuda()
        val_class_ohe_tensor = val_class_ohe_tensor.cuda()
        train_expression_tensor = train_expression_tensor.cuda()
        train_class_ohe = train_class_ohe.cuda()
    
    trainset = torch.utils.data.TensorDataset(train_expression_tensor,
                                              train_class_ohe)
    dataloader_train = torch.utils.data.DataLoader(trainset,
                                            batch_size=cfg.batch_size,
                                            shuffle=True,
                                            #num_workers=cfg.num_workers,
                                            drop_last=True)
    valset = torch.utils.data.TensorDataset(val_expression_tensor,
                                            val_class_ohe_tensor)
    dataloader_val = torch.utils.data.DataLoader(valset,
                                                 batch_size=val_expression_tensor.size()[0],
                                                 shuffle=False,
                                                 drop_last=True)

    return dataloader_train, dataloader_val


def create_update_class(model, discriminator, config):
    training_upd = VAEUpdater(model, discriminator, config,
                              train=True, cuda=torch.cuda.is_available())
    validator_upd = Validator(model, discriminator, cuda=torch.cuda.is_available())

    return training_upd, validator_upd


def create_model(cfg):
    vae_model = VAE(cfg.bottleneck, cfg.input_dim, cfg.count_classes)
    disc_model = Latent_discriminator(cfg.bottleneck, cfg.count_classes)
    
    if torch.cuda.is_available():
        vae_model = vae_model.cuda()
        disc_model = disc_model.cuda()

    return vae_model, disc_model


class LossAggregatorMetric(Metric):
    def __init__(self, *args, **kwargs):
        self.total_losses = defaultdict(float)
        self.num_updates = defaultdict(int)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.total_losses = defaultdict(float)
        self.num_updates = defaultdict(int)

    def update(self, output):
        for name, val in output.items():
            self.total_losses[name] += float(val)
            self.num_updates[name] += 1

    def compute(self):
        losses = {name: val / self.num_updates[name] for name, val in self.total_losses.items()}

        return losses


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


with Experiment(EXPERIMENTS_DIR, cfg) as exp:
    print(f'Experiment started: {exp.experiment_id}')
    dataloader_train, dataloader_val = load_datasets(cfg)

    model, disc = create_model(cfg)

    update_class_train, update_class_val = create_update_class(model, disc, cfg)

    trainer = Engine(update_class_train)
    validator = Engine(update_class_val)
    
    trainer.iteration = 0

    metrics = {'loss': LossAggregatorMetric(), }
    for metric_name, metric in metrics.items():
        metric.attach(evaluator, metric_name)
    
    best_loss = inf
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        global best_loss

        validator.run(dataloader_val)
        losses_val = engine.state.loss

        log_progress(trainer.state.epoch, trainer.state.iteration, losses_val, 'val', tensorboard_writer)

        # if losses_val[exp.config.best_loss] < best_loss:
        #     best_loss = losses_val[exp.config.best_loss]
        #     save_weights(model, exp.experiment_dir.joinpath('best.th'))
    

    _TENSORBOARD_DIR = cfg.experiment_dir.joinpath('log')
    tensorboard_writer = SummaryWriter(str(_TENSORBOARD_DIR))

    trainer.run(dataloader_train, max_epochs=cfg.epochs)

    print(f'Experiment {exp.experiment_id} has finished')
