from pathlib import Path
import sys
from ignite.engine import Engine, Events
from ignite.metrics import Metric
import numpy as np
from tensorboardX import SummaryWriter
from torch.cuda import is_available as cuda_is_available
from torch import nn
from torch import Tensor
import torch
from sklearn.model_selection import train_test_split
import argparse

from data import get_raw_data
from model.autoencoders import VAE
from model.modules import Latent_discriminator
from model import Classifier
from config import Config
from utils import load_datasets
from model.classifier import Classifier

cfg = Config()
THIS_DIR = Path('./').absolute()
EXPERIMENTS_DIR = THIS_DIR.joinpath('experiment')

def train_classifiers(annot_dataloader, count_celltypes, count_classes):
    celltype_clf = Classifier(inp_size=cfg.input_dim, out_size=count_celltypes)
    form_clf = Classifier(inp_size=cfg.input_dim, out_size=count_classes)
    if cfg.use_cuda and cuda_is_available():
        celltype_clf = celltype_clf.cuda()
        form_clf = form_clf.cuda()
    
    celltype_clf_opt = torch.optim.Adam(celltype_clf.parameters(), 
                                        weight_decay=cfg.celltype_clf_wdecay,
                                        lr=cfg.celltype_clf_lr)
    form_clf_opt = torch.optim.Adam(form_clf.parameters(),
                                        weight_decay=cfg.form_clf_wdecay,
                                        lr=cfg.form_clf_lr)

    celltype_criterion = nn.CrossEntropyLoss()
    form_criterion = nn.CrossEntropyLoss()

    print('\n')
    for epoch in range(cfg.classifier_epochs):
        print(f'\rTraining classifier [{epoch+1}/{cfg.classifier_epochs}]', end='')
        celltype_clf.train()
        form_clf.train()
        for exp_, form_, cell_type_ in annot_dataloader:
            exp_ = exp_.cuda()
            form_ = form_.argmax(-1).cuda()
            cell_type_ = cell_type_.argmax(-1).cuda()
            
            predicted_celltype_ = celltype_clf(exp_)
            celltype_loss_on_batch = celltype_criterion(predicted_celltype_, cell_type_)
            
            celltype_clf_opt.zero_grad()
            celltype_loss_on_batch.backward()
            celltype_clf_opt.step()
            
            predicted_form_ = form_clf(exp_)
            form_loss_on_batch = form_criterion(predicted_form_, form_)
            
            form_clf_opt.zero_grad()
            form_loss_on_batch.backward()
            form_clf_opt.step()

        celltype_clf.eval()
        form_clf.eval()

        return celltype_clf, form_clf

def test(vae_model, discrim, annot_dataloader,
            test_expression = None, 
            class_ohe_test = None, celltype_test = None):
    #STYLE TRANSFER
    ge_transfer_raw = np.repeat(test_expression, class_ohe_test.shape[1], axis=0)
    ge_transfer_raw = Tensor(ge_transfer_raw)
    init_classes_transfer = np.repeat(class_ohe_test, class_ohe_test.shape[1], axis=0)
    init_classes_transfer = Tensor(init_classes_transfer)
    init_celltypes_transfer = np.repeat(celltype_test, class_ohe_test.shape[1], axis=0)
    init_celltypes_transfer = Tensor(init_celltypes_transfer)

    if cfg.use_cuda and cuda_is_available():
        ge_transfer_raw = ge_transfer_raw.cuda()
        init_classes_transfer = init_classes_transfer.cuda()
        init_celltypes_transfer = init_celltypes_transfer.cuda()

    target_classes_transfer = np.zeros((class_ohe_test.shape[0] * class_ohe_test.shape[1],
                                        class_ohe_test.shape[1]))
    target_classes_transfer[np.arange(target_classes_transfer.shape[0]),
                        np.arange(target_classes_transfer.shape[0])%target_classes_transfer.shape[1]] = 1

    target_classes_transfer = Tensor(target_classes_transfer).cuda()
    transfer_expression_tensor, _, __ = vae_model(ge_transfer_raw, target_classes_transfer)
    transfer_expression_np = transfer_expression_tensor.cpu().detach().numpy()
    
    test_expression_tensor = Tensor(test_expression)
    class_ohe_test_tensor = Tensor(class_ohe_test)
    if cfg.use_cuda and cuda_is_available():
        test_expression_tensor = test_expression_tensor.cuda()
        class_ohe_test_tensor = class_ohe_test_tensor.cuda()
    reconstruction = vae_model(test_expression_tensor, class_ohe_test_tensor)[0].cpu().detach().numpy()

    mse = (reconstruction - test_expression) ** 2
    print('Mean square error reconstructed expression:', np.mean(mse))
    print('Mean values of test expression:', test_expression.mean())
    print('Mean values of reconstructed expression:', reconstruction.mean())
    #TODO label
    print('test expression <0.5 mean:', (test_expression < 0.5).mean())
    print('reconstructed exrression <0.5 mean:', (reconstruction < 0.5).mean())

    residual_transfer = ge_transfer_raw.cpu().numpy() - transfer_expression_np
    res_norm = (residual_transfer**2).mean(1).reshape((-1, cfg.count_classes))#l2 norm,
    print('Calibration accuracy', (res_norm.argmin(1) == class_ohe_test.argmax(1)).mean())

    #train
    celltype_clf, form_clf = train_classifiers(annot_dataloader, 
                                                celltype_test.shape[1], 
                                                cfg.count_classes)
    
    print('\n')

    celltype_train_labelenc = annot_dataloader.dataset.tensors[2].argmax(1).cpu().numpy()
    celltype_test_raw_labelenc = celltype_test.argmax(1)
    celltype_test_labelenc = init_celltypes_transfer.argmax(1).cpu().numpy()
    form_test_labelenc = init_celltypes_transfer.argmax(1).cpu().numpy()
    
    predicted_celltype_test_raw = celltype_clf(test_expression_tensor)
    predicted_celltype_test = celltype_clf(transfer_expression_tensor)
    predicted_celltype_train = celltype_clf(annot_dataloader.dataset.tensors[0])
    predicted_form_test = form_clf(transfer_expression_tensor)
    
    predicted_celltype_test_labelenc = predicted_celltype_test.argmax(1).cpu().detach().numpy()
    predicted_celltype_test_raw_labelenc = predicted_celltype_test_raw.argmax(1).cpu().detach().numpy()
    predicted_celltype_train_labelenc = predicted_celltype_train.argmax(1).cpu().detach().numpy()
    sameclass_mask = (init_classes_transfer.argmax(1) == target_classes_transfer.argmax(1)).cpu().numpy()
    
    celltype_train_accuracy = (predicted_celltype_train_labelenc == celltype_train_labelenc).mean()
    celltype_test_raw_accuracy = (predicted_celltype_test_raw_labelenc == celltype_test_raw_labelenc).mean()
    celltype_test_accuracy = (predicted_celltype_test_labelenc == celltype_test_labelenc).mean()
    
    celltype_notransfer_accuracy = (predicted_celltype_test_labelenc[sameclass_mask] == celltype_test_labelenc[sameclass_mask]).mean(0)
    celltype_transfer_accuracy = (predicted_celltype_test_labelenc[~sameclass_mask] == celltype_test_labelenc[~sameclass_mask]).mean(0)
    form_test_accuracy = (predicted_form_test.argmax(1).cpu().detach().numpy() == form_test_labelenc).mean(0)
    
    print('Cell type prediction accuracy [train]:', celltype_train_accuracy)
    print('Cell type prediction accuracy [test hold out data]', celltype_test_raw_accuracy)
    print('Cell type prediction accuracy [vae transfered data]', celltype_test_accuracy,
            '< with the same class transfer')
    #reconstructed means transfered to the same class
    print('Cell type prediction accuracy [reconstructed data]', celltype_notransfer_accuracy)
    print('Cell type prediction accuracy [transfered to another classes]', celltype_transfer_accuracy)
    print('Class prediction accuracy:', form_test_accuracy)

    return {
        'Cell type prediction accuracy [train]:': celltype_train_accuracy,
        'Cell type prediction accuracy [test hold out data]': celltype_test_raw_accuracy,
        'Cell type prediction accuracy [vae transfered data]': celltype_test_accuracy,
        'Cell type prediction accuracy [reconstructed data]': celltype_notransfer_accuracy,
        'Cell type prediction accuracy [transfered to another classes]': celltype_transfer_accuracy,
        'Class prediction accuracy:': form_test_accuracy
    }


if __name__ == '__main__':
    pass
