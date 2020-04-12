from utils import load_datasets, scvi_anndata, get_high_variance_genes
from model.ml import entropy_batch_mixing
from model.ml import get_knn_purity
from train_script import train
from test_script import test, train_classifiers
from config import Config
from experiment import Experiment

from scvi.dataset import (
    PreFrontalCortexStarmapDataset,
    FrontalCortexDropseqDataset,
    RetinaDataset,
    HematoDataset,
    CsvDataset,
    AnnDatasetFromAnnData,
)
from scvi.models import SCANVI, VAE
from scvi.inference import (
    UnsupervisedTrainer,
    JointSemiSupervisedTrainer,
    SemiSupervisedTrainer,
)
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np
import torch
from torch import zeros, Tensor, LongTensor, log, cuda, save, load
from tsnecuda import TSNE
from matplotlib import pyplot as plt
from tqdm import tqdm
from pathlib import Path
import scanpy as sc


def benchmark_scvi(dataset, dataset_name, cfg, **kwargs):
    log_name = dataset_name
    n_genes = min(dataset.X.shape[1], cfg.n_genes)

    vae = VAE(
        dataset.nb_genes,
        n_batch=dataset.n_batches,
        n_labels=dataset.n_labels,
        n_hidden=128,
        n_latent=30,
        n_layers=2,
        dispersion="gene",
    )

    trainer = UnsupervisedTrainer(vae, dataset, train_size=0.75)
    n_epochs = cfg.epochs if not "epochs" in kwargs else kwargs["epochs"]
    trainer.train(n_epochs=n_epochs)

    full = trainer.create_posterior(
        trainer.model, dataset, indices=np.arange(len(dataset))
    )
    latents, batch_indices, labels = full.sequential().get_latent()

    res = {}
    res["knn purity"] = []
    res["entropy batch mixing"] = []
    res["knn purity"].append(get_knn_purity(latents, labels.reshape((-1, 1))))
    ebm = entropy_batch_mixing(latents, batch_indices)
    res["entropy batch mixing"].append(ebm[1] if isinstance(ebm, tuple) else ebm)

    cfg.input_dim = latents.shape[1]
    cfg.count_classes = np.unique(dataset.batch_indices).shape[0]
    cfg.count_labels = np.unique(dataset.labels).shape[0]

    (
        latents_train,
        latents_test,
        batches_train,
        batches_test,
        labels_train,
        labels_test,
    ) = train_test_split(
        latents,
        batch_indices,
        labels,
        test_size=0.25,
        stratify=batch_indices.reshape(-1),
    )

    latents_train = torch.Tensor(latents_train).cuda()
    latents_test = torch.Tensor(latents_test).cuda()

    batches_train_tensor = torch.zeros(latents_train.shape[0], cfg.count_classes)
    batches_train_tensor = batches_train_tensor.scatter(
        1, LongTensor(batches_train.astype("int16")).view(-1, 1), 1
    )
    batches_train_tensor = batches_train_tensor.cuda()

    labels_train_tensor = torch.zeros(latents_train.shape[0], cfg.count_labels)
    labels_train_tensor = labels_train_tensor.scatter(
        1, LongTensor(labels_train.astype("int16")).view(-1, 1), 1
    )
    labels_train_tensor = labels_train_tensor.cuda()

    train_dataset = torch.utils.data.TensorDataset(
        latents_train, batches_train_tensor, labels_train_tensor
    )
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size)

    cfg.classifier_input_dim = cfg.bottleneck
    ohe_classifier, form_classifier = train_classifiers(
        cfg, dataloader, cfg.count_labels, cfg.count_classes
    )
    preds_batches = ohe_classifier(latents_test)
    preds_labels = form_classifier(latents_test)

    res["batch classifing accuracy"] = (
        preds_batches.argmax(1).cpu().detach().numpy() == batches_test
    ).mean()
    res["labels classifing accuracy"] = (
        preds_labels.argmax(1).cpu().detach().numpy() == labels_test
    ).mean()

    (Path(cfg.metrics_dir) / 'scVI').mkdir(parents=True, exist_ok=True)
    with open(os.path.join(Path(cfg.metrics_dir) / "scVI", log_name + ".json"), "w") as file:
        for key in res.keys():
            if type(key) is not str:
                try:
                    res[str(key)] = res[key]
                except:
                    try:
                        res[repr(key)] = res[key]
                    except:
                        raise TypeError("Unexpected key")
        json.dump(res, file)

    del vae, trainer
    del latents, batch_indices, labels, full
    del preds_batches, preds_labels, train_dataset, dataloader
    cuda.empty_cache()
