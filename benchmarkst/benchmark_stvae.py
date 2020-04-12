from utils import load_datasets, get_high_variance_genes, scvi_anndata
from train_script import train
from test_script import test
from config import Config
from experiment import Experiment
from model.autoencoders import VAE
from data import get_raw_data

from scvi.dataset import (
                    PreFrontalCortexStarmapDataset,
                    RetinaDataset,
                  )
import argparse
import os
import json
import numpy as np
from torch import zeros, Tensor, LongTensor, log, cuda, save, load
import scanpy as sc
from functools import partial
from pathlib import Path
import gc


def benchmark_stvae(dataset, log_name, cfg, **kwargs):
    ds = dataset
    n_genes = min(ds.X.shape[1], cfg.n_genes)
    expression = np.log(ds.X + 1.)
    scvai_genes, scvai_batches_ind, scvai_labels_ind = get_high_variance_genes(
        expression,
        ds.batch_indices,
        ds.labels,
        n_genes=n_genes,
        argmax=False
    )

    cfg.count_classes = np.unique(ds.batch_indices).shape[0]
    cfg.count_labels = np.unique(ds.labels).shape[0]
    cfg.input_dim = int(scvai_genes.shape[1])

    data = load_datasets(cfg, True, True, (scvai_genes, scvai_batches_ind, scvai_labels_ind))
    dataloader_train = data[0]
    dataloader_val = data[1]
    dataloader_test = data[2]
    annot_train = data[3]
    annot_test = data[4]

    styletransfer_test_expr = annot_test.dataset.tensors[0].cpu().numpy()
    styletransfer_test_class = annot_test.dataset.tensors[1].cpu().numpy()
    styletransfer_test_celltype = annot_test.dataset.tensors[2].cpu().numpy()


    model = None
    disc = None

    print('Training...')
    model, disc = train(dataloader_train, dataloader_val, cfg, model, disc)

    print('Tests...')
    print('Dataset:', log_name)
    cfg.classifier_epochs=1
    res = test(cfg,
        model, disc,
        annot_train,
        styletransfer_test_expr,
        styletransfer_test_class,
        styletransfer_test_celltype,
        dataset_name=log_name
    )

    (Path(cfg.metrics_dir) / 'stVAE').mkdir(parents=True, exist_ok=True)
    with open(Path(cfg.metrics_dir) / 'stVAE/' / (log_name + '.json'), 'w') as file:
        json.dump(res, file, indent=4)

    del ds
    del model, disc
    del styletransfer_test_expr
    del styletransfer_test_class
    del styletransfer_test_celltype
    del data
    del dataloader_train, dataloader_val, dataloader_test
    del annot_train, annot_test
    del scvai_genes, scvai_batches_ind, scvai_labels_ind
    gc.collect()
    cuda.empty_cache()
