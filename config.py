from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    noise_beta: float = 0.1
    decay_beta: float = 1.
    clip_value: float = 1.
    ae_lr: float = 1e-4
    disc_lr: float = 1e-4
    vae_beta: float = 1e-4

    experiment_dir: str = 'experiment'
