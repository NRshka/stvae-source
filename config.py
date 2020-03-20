'''

'''
from dataclasses import dataclass
from pathlib import Path
from numpy import inf
from typing import Optional, Tuple, Union


@dataclass
class Config:
    '''
    Training configuration
    '''
    noise_beta: float = 0.1
    decay_beta: float = 1.
    clip_value: float = 1.
    vae_lr: float = 0.001
    vae_decay: float = 1e-3
    disc_lr: float = 1e-4
    disc_decay: float = 0.
    vae_beta: float = 2e-5
    adv_weight: float = 0.0000001
    cyclic_weight: float = 0.2
    mmd_weight: float = 0
    l1_weight: float = 0#.5
    kernel_mu: float = 0.4
    model_scheduler_gamma: float = 0.992
    discr_scheduler_gamma: float = 0.992
    n_layers: int = 2
    scale_alpha: float = 1.3
    form_consistency_weight: float = 0.2

    bottleneck: int = 30
    input_dim: int = 15987
    count_labels: int = 12
    count_classes: int = 3
    epochs: int = 600
    batch_size: int = 128
    num_workers: Optional[int] = None
    activation: str = 'mish'
    n_genes: int = 700

    classifier_hidden_size: int = 512
    classifier_epochs: int = 450
    celltype_clf_lr: float = 1e-5
    form_clf_lr: float = 3e-4
    celltype_clf_wdecay: float = 0#weight decay
    form_clf_wdecay: float = 0#weight decay

    experiment_dir: Union[Path, str] = Path('./data/experiment')
    data_dir: Union[Path, str] = Path('./genes')
    metrics_dir: Union[Path, str] = Path('./metrics')
    best_loss = inf
    early_stop_epochs: int = 30#Number of epochs to wait if no loss improvement and then stop the training.
    use_cuda: bool = True
    random_state: int = 512
    cuda: bool = True
    reproducibility: bool = True
    verbose: str = 'all'
    device_ids: Tuple[int] = (2,)

    def __str__(self):
        return '\n'.join([f"{key}: {value}" for key, value in self.__dict__.items()])
