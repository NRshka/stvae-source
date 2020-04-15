import argparse
import importlib
from pathlib import Path

from scvi.dataset import (
    PreFrontalCortexStarmapDataset,
    RetinaDataset,
    PbmcDataset,
    CsvDataset
)

from customDatasets import (
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


DIRPATH = Path('./').absolute()

BENCHMARK_FUNCTIONS = (
    benchmark_stvae,
    benchmark_scvi,
    benchmark_scgen,
    benchmark_trvae
)
FRAMEWORKS = (
    'stvae',
    'scvi',
    'scgen',
    'trvae'
)
#EPOCHS = (600, 100, 100, 300)
EPOCHS = (1, 1, 1, 1)

datasets = {
    'scvi_pbmc': PbmcDataset(),
    'bermuda_pbmc': CsvDataset(
        str(DIRPATH / './pbmc/expression.csv'),
        labels_file = str(DIRPATH / './pbmc/labels.csv'),
        batch_ids_file = str(DIRPATH / './pbmc/batches.csv'),
        gene_by_cell = False
    ),
    'mouse': CsvDataset(
        str(DIRPATH / './mouse_genes/ST1 - original_expression.csv'),
        labels_file = str(DIRPATH / './mouse_genes/labels.csv'),
        batch_ids_file = str(DIRPATH / './mouse_genes/batches.csv'),
        gene_by_cell = False
    ),
    #'pancreas': BermudaDataset('./pancreas/muraro_seurat.csv'),
    'retina': RetinaDataset(),
    'starmap': PreFrontalCortexStarmapDataset(),
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
        for bench_func, epoch, framework in zip(BENCHMARK_FUNCTIONS,
                                                EPOCHS,
                                                FRAMEWORKS):
            cfg.epochs = epoch
            data = predefined_preprocessing(dataset, framework=framework)
            #data = dataset
            bench_func(data, log_name, cfg)
