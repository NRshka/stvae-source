import os
from collections import Iterable
import torch
import numpy as np
import tempfile
from pathlib import Path

from utils import save_pickle, load_pickle


class Experiment:
    _CONFIG_FILENAME = 'config.pkl'

    def __init__(self, experiments_dir, config, prefix=None, random_seed=0):
        self.config = config
        self.experiments_dir = Path(experiments_dir)
        self.prefix = prefix
        self.random_seed = random_seed if isinstance(random_seed, int) else 0

        # create dir for the experiment
        if self.prefix is not None:
            self.prefix = f'{self.prefix}.'

        self.experiment_dir = None
        self.experiment_id = None

    def __enter__(self):
        # check if dir exists
        if not self.experiments_dir.is_dir():
            self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_dir = Path(tempfile.mkdtemp(dir=self.experiments_dir, prefix=self.prefix))
        self.experiment_id = self.experiment_dir.name

        # save the config file
        Experiment._save_config(self.config, self.experiment_dir)

        #make it reproducibility
        if self.config.reproducibility:
            np.random.seed(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(self.random_seed)

        # use cuda
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        try:
            if isinstance(self.config.device_ids, Iterable):
                device_ids = list(map(str, self.config.device_ids))
            else:
                device_ids = [str(self.config.device_ids)]
        except TypeError as err:
            raise TypeError("device_ids must contains values that can be \
                            casted into a string")
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(device_ids)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @classmethod
    def load(cls, experiments_dir, experiment_id):
        experiment_dir = experiments_dir.joinpath(experiment_id)

        config = Experiment._load_config(experiment_dir)

        exp = Experiment(experiments_dir, config)
        exp.experiment_dir = experiment_dir
        exp.experiment_id = exp.experiment_dir.name

        return exp

    @classmethod
    def _save_config(cls, config, experiment_dir):
        filename = experiment_dir.joinpath(Experiment._CONFIG_FILENAME)
        save_pickle(config, filename)

    @classmethod
    def _load_config(cls, experiment_dir):
        filename = experiment_dir.joinpath(Experiment._CONFIG_FILENAME)
        config = load_pickle(filename)

        return config
