from data import make_anndata
from utils import load_datasets, scvi_anndata, get_high_variance_genes
from train_script import train
from test_script import test
from config import Config
from model.trvae import trVAE
from experiment import Experiment

from scvi.dataset import (
                    PreFrontalCortexStarmapDataset,
                    FrontalCortexDropseqDataset,
                    RetinaDataset,
                    HematoDataset,
                    AnnDatasetFromAnnData
                  )
import os
import json
import numpy as np
from torch import zeros, Tensor, LongTensor, log, cuda, save, load
import scanpy as sc
import trvaep
from functools import partial
from pathlib import Path


def benchmark_trvae(dataset, log_name, cfg, **kwargs):
    ds = dataset
    n_genes = min(ds.X.shape[1], cfg.n_genes)

    scvai_genes, scvai_batches_ind, scvai_labels_ind = get_high_variance_genes(
        ds.X,
        ds.batch_indices,
        ds.labels,
        n_genes = n_genes,
        argmax=False
    )
    cfg.count_classes = int(np.max(ds.batch_indices) + 1)
    cfg.count_labels = int(np.max(ds.labels) + 1)
    cfg.input_dim = int(scvai_genes.shape[1])


    data = load_datasets(cfg, True, True,
                         (scvai_genes, scvai_batches_ind, scvai_labels_ind),
                         0.9)
    dataloader_train = data[0]
    dataloader_val = data[1]
    dataloader_test = data[2]
    annot_train = data[3]
    annot_test = data[4]
    x, batch_ind, celltype = annot_train.dataset.tensors
    batch_ind = batch_ind.argmax(dim=1)
    celltype = celltype.argmax(dim=1)

    anndata_train = make_anndata(x.cpu().numpy(),
                                 batch_ind.cpu().numpy(),
                                 'condition',
                                 celltype.cpu().numpy(),
                                 'cell_type')
    x_test, batch_ind_test, celltype_test = annot_test.dataset.tensors
    batch_ind_test = batch_ind_test.argmax(dim=1)
    celltype_test = celltype_test.argmax(dim=1)
    anndata_test = make_anndata(x_test.cpu().numpy(), batch_ind_test.cpu().numpy(),
                                'condition', celltype_test.cpu().numpy(), 'cell_type')
    sc.pp.normalize_per_cell(anndata_train)
    sc.pp.normalize_per_cell(anndata_test)
    sc.pp.log1p(anndata_train)
    sc.pp.log1p(anndata_test)

    n_conditions = anndata_train.obs["condition"].unique().shape[0]
    x_test = anndata_test.X
    batch_ind_test_tmp = anndata_test.obs['condition']
    batch_ind_test = zeros(batch_ind_test_tmp.shape[0], cfg.count_classes)
    batch_ind_test = batch_ind_test.scatter(1, LongTensor(batch_ind_test_tmp.astype('uint16')).view(-1, 1), 1).numpy()
    celltype_test_tmp = anndata_test.obs['cell_type']
    celltype_test = zeros(celltype_test_tmp.shape[0], cfg.count_labels)
    celltype_test = celltype_test.scatter(1, LongTensor(celltype_test_tmp.astype('uint16')).view(-1, 1), 1).numpy()

    model = trVAE(x.shape[1],
                  num_classes=n_conditions,
                  encoder_layer_sizes=[128, 32],
                  decoder_layer_sizes=[32, 128],
                  latent_dim=cfg.bottleneck,
                  alpha=0.0001,
                 )
    trainer = trvaep.Trainer(model, anndata_train)

    print('Training...')
    trainer.train_trvae(cfg.epochs, 512, 50)#n_epochs, batch_size, early_patience

    print('Tests...')
    print('Dataset:', log_name)
    res = test(cfg,
                model, None,
                annot_train,
                x_test,
                batch_ind_test,
                celltype_test
    )
    res['n_genes'] = n_genes

    metrics_path = Path(cfg.metrics_dir) / 'trVAE'
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / (log_name + '.json'), 'w') as file:
        json.dump(res, file, indent=4)

    del ds
    del model
    del data
    del dataloader_train, dataloader_val, dataloader_test
    del annot_train, annot_test
    del scvai_genes, scvai_batches_ind, scvai_labels_ind
    cuda.empty_cache()

