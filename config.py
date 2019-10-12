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
    num_workers: Optional[int] = None

    experiment_dir: Path = Path('./data/experiment')
    data_dir: Path = Path('./genes')
    best_loss = inf

    batch_size: int = 256
