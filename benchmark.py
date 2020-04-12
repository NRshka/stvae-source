import argparse
import importlib

from scvi.dataset import (
    PreFrontalCortexStarmapDataset,
    RetinaDataset
)

from customDatasets import (
    MouseDataset,
    BermudaDataset
)
from utils import (
    predefined_preprocessing
)
from config import Config
from experiment import Experiment

from benchmarkst import (
    benchmark_scvi,
    benchmark_scgen,
    benchmark_trvae,
    benchmark_stvae
)
import pdb
pdb.set_trace()


BENCHMARK_FUNCTIONS = (
    benchmark_stvae,
    benchmark_scvi,
    benchmark_scgen,
    benchmark_trvae
)
EPOCHS = (600, 100, 100, 300)
datasets = {
    'pbmc': BermudaDataset('./pbmc/pbmc8k_seurat.csv'),
    'pancreas': BermudaDataset('./pancreas/muraro_seurat.csv'),
    'retina': RetinaDataset,
    'starmap': PreFrontalCortexStarmapDataset,
    'mouse': MouseDataset(
        './mouse_genes',
        'batches.csv',
        'labels.csv'
    ),
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
    if args.custom_config:
        try:
            spec = importlib.util.spec_from_file_location(args.custom_config)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            cfg = config_module.Config()
        except:
            raise ValueError(f"Cannot import config module from {args.custom_config}")

    for log_name, dataset in datasets.items():
        for bench_func, epoch in zip(BENCHMARK_FUNCTIONS, EPOCHS):
            cfg.epochs = epoch
            data = dataset()
            data = predefined_preprocessing(data, framework='scvi')
            bench_func(data, log_name, cfg)
