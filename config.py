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
    
    bottleneck: int = 200
    input_dim: int = 15987
    count_classes: int = 3
    epochs: int = 10
    batch_size: int = 256
    num_workers: Optional[int] = None

    classifier_hidden_size: int = 512
    classifier_epochs: int = 250
    celltype_clf_lr: float = 1e-5
    form_clf_lr: float = 3e-4
    celltype_clf_wdecay: float = 0#weight decay
    form_clf_wdecay: float = 0#weight decay

    experiment_dir: Path = Path('./data/experiment')
    data_dir: Path = Path('./genes')
    best_loss = inf
    use_cuda: bool = True
    random_state: int = 512
    cuda: bool = True
