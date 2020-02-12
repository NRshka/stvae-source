'''

'''
from dataclasses import dataclass
from pathlib import Path
from numpy import inf
from typing import Optional


@dataclass
class Config:
    '''
    Training configuration
    '''
    noise_beta: float = 0.1
    decay_beta: float = 1.
    clip_value: float = 1.
    vae_lr: float = 1e-4
    vae_decay: float = 1e-3
    disc_lr: float = 1e-4
    disc_decay: float = 0.
    vae_beta: float = 1e-4
    adv_weight: float = 0.07
    cyclic_weight: float = 0.2
    mmd_weight = 1e1#1e1
    l1_weight = 0#.5
    kernel_mu = 0.4
    model_scheduler_gamma = 0.992
    discr_scheduler_gamma = 0.992

    bottleneck: int = 1800
    input_dim: int = 15987
    count_labels: int = 12
    count_classes: int = 3
    epochs: int = 400
    batch_size: int = 256
    num_workers: Optional[int] = None
    activation: str = 'mish'

    classifier_hidden_size: int = 512
    classifier_epochs: int = 250
    celltype_clf_lr: float = 1e-5
    form_clf_lr: float = 3e-4
    celltype_clf_wdecay: float = 0#weight decay
    form_clf_wdecay: float = 0#weight decay

    experiment_dir: Path = Path('./data/experiment')
    data_dir: Path = Path('./genes')
    best_loss = inf
    early_stop_epochs: int = 30#Number of epochs to wait if no loss improvement and then stop the training.
    use_cuda: bool = True
    random_state: int = 512
    cuda: bool = True

    def __str__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])
