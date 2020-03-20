import argparse
from functools import partial
from pathlib import Path
import numpy as np

from utils import(
    load_datasets,
    get_high_variance_genes,
    scvi_anndata,
    predefined_preprocessing
)
from data import get_raw_data
from train_script import train
from test_script import test
from config import Config
from experiment import Experiment
from model.autoencoders import VAE

from benchmark_scvi import benchmark_scvi
from benchmark_stvae import benchmark_stvae
from benchmark_scgen import benchmark_scgen

from customDatasets import MouseDataset, PbmcDataset
from scvi.dataset import (
    PreFrontalCortexStarmapDataset,
    RetinaDataset
)


datasets = {
    'retina': RetinaDataset,
    'starmap': PreFrontalCortexStarmapDataset,
    #'mouse': MouseDataset(
    #    './mouse_genes/ST1 - original_expression.csv',
    #    './mouse_genes/batches.csv',
    #    './mouse_genes/labels.csv'
    #)
}

parser = argparse.ArgumentParser(description="A way to define variables for \
                                 training, tests and models")
parser.add_argument('--metrics_dir', type=str, help="Path to save metrics file \
                    \nDisabled if save_metrics flag is False.\n\nDefault value \
                    is './'")
parser.add_argument('--save_metrics', type=bool, help='Boolean flag determines \
                    whether metrics should be saved.\n\n Default value is True')
parser.add_argument('--custom_config', type=str, help='The path to the \
                    configuration file to be used. If equal to None, then \
                    the config located in the same folder will be used.',
                    default=None)
args = parser.parse_args()

cfg = Config()
with Experiment('', cfg) as exp:
    if args.metrics_dir:
        cfg.metrics_dir = args.metrics_dir

    for log_name, dataset in datasets.items():
        data = dataset()
        data = predefined_preprocessing(data, framework='scvi')
        import pdb
        pdb.set_trace()
        benchmark_scgen(data, log_name, cfg)
