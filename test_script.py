from typing import Optional, Union, Dict
from pathlib import Path
import sys
from ignite.engine import Engine, Events
from ignite.metrics import Metric
from itertools import combinations
import numpy as np
from tensorboardX import SummaryWriter
from torch.cuda import is_available as cuda_is_available
from torch import nn
from torch import Tensor
import torch
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pandas as pd

from data import get_raw_data
from model.autoencoders import VAE
from model.modules import Latent_discriminator
from model import Classifier
from config import Config
from utils import load_datasets
from model.classifier import Classifier
from model.ml import get_knn_purity, get_batch_entropy, entropy_batch_mixing


THIS_DIR = Path('./').absolute()
EXPERIMENTS_DIR = THIS_DIR.joinpath('experiment')

def train_classifiers(cfg, annot_dataloader, count_celltypes, count_classes,
                      track = False, test_data = None):
    device = torch.device('cuda' if cuda_is_available() and cfg.use_cuda else 'cpu')
    celltype_clf = Classifier(inp_size=cfg.classifier_input_dim, out_size=count_celltypes)
    form_clf = Classifier(inp_size=cfg.classifier_input_dim, out_size=count_classes)
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
    metrics = {
        'celltype_losses':   [],
        'form_losses':       [],
        'celltype_accuracy': [],
        'form_accuracy':     [],
        'epoch':             []
    }
    for epoch in range(cfg.classifier_epochs):
        print(f'\rTraining classifier [{epoch+1}/{cfg.classifier_epochs}]', end='')
        celltype_clf.train()
        form_clf.train()
        metrics['epoch'].append(epoch + 1)
        celltype_av_loss = 0.
        form_av_loss = 0.
        counter = 0
        for exp_, form_, cell_type_ in annot_dataloader:
            exp_ = exp_.to(device)
            form_ = form_.argmax(-1).to(device)
            cell_type_ = cell_type_.argmax(-1).to(device)

            predicted_celltype_ = celltype_clf(exp_)
            celltype_loss_on_batch = celltype_criterion(predicted_celltype_, cell_type_)
            celltype_av_loss += celltype_loss_on_batch.item()

            celltype_clf_opt.zero_grad()
            celltype_loss_on_batch.backward(retain_graph=True)
            celltype_clf_opt.step()

            predicted_form_ = form_clf(exp_)
            form_loss_on_batch = form_criterion(predicted_form_, form_)
            form_av_loss += form_loss_on_batch.item()

            form_clf_opt.zero_grad()
            form_loss_on_batch.backward(retain_graph=True)
            form_clf_opt.step()

            counter += 1

        metrics['celltype_losses'].append(celltype_av_loss / counter)
        metrics['form_losses'].append(form_av_loss / counter)
    if track:
        celltype_clf.eval()
        form_clf.eval()
        counter = 0
        celltype_accuracy = 0.
        form_accuracy = 0.
        for exp_, form_, cell_type_ in test_data:
            exp_ = exp_.to(device)
            form_ = form_.argmax(-1)
            cell_type_ = cell_type_.argmax(-1)

            predicted_celltype_ = celltype_clf(exp_).cpu().detach().numpy().argmax(-1)
            cell_type_ = cell_type_.cpu().detach().numpy()
            celltype_accuracy += (predicted_celltype_ == cell_type_).mean()

            predicted_form_ = form_clf(exp_).cpu().detach().numpy().argmax(-1)
            form_ = form_.cpu().detach().numpy()
            form_accuracy += (predicted_form_ == form_).mean()

            counter += 1
        metrics['celltype_accuracy'].append(celltype_accuracy / counter)
        metrics['form_accuracy'].append(form_accuracy / counter)

    if track:
        return celltype_clf, form_clf, metrics
    return celltype_clf, form_clf

def test(cfg, vae_model, discrim, annot_dataloader,
         test_expression: np.ndarray = None,
         class_ohe_test = None, celltype_test = None,
         pretrained_classifiers=None, dataset_name: str = '') -> Dict[str, Union[float, str, None]]:
    '''Calculate metrics and return dict
        :Param cfg: Config dataclass
        :Param vae_model: autoecoder with declarated signature
        :Param discrim: deprecated, unused
        :Param annot_dataloader: dataloader(expression, batch_indices, labels)
            for classifier training
        :Param test_expression:
    '''
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

    target_classes_transfer = Tensor(target_classes_transfer)
    if cfg.use_cuda and cuda_is_available():
        target_classes_transfer = target_classes_transfer.cuda()

    transfer_expression_tensor = vae_model(ge_transfer_raw, target_classes_transfer)[0]
    if isinstance(transfer_expression_tensor, tuple):
        transfer_expression_tensor = transfer_expression_tensor[0]
    transfer_expression_np = transfer_expression_tensor.cpu().detach().numpy()

    test_expression_tensor = Tensor(test_expression)
    class_ohe_test_tensor = Tensor(class_ohe_test)
    if cfg.use_cuda and cuda_is_available():
        target_classes_transfer = target_classes_transfer.cuda()
        test_expression_tensor = test_expression_tensor.cuda()
        class_ohe_test_tensor = class_ohe_test_tensor.cuda()
    reconstruction = vae_model(test_expression_tensor, class_ohe_test_tensor)[0]
    if isinstance(reconstruction, tuple):
        reconstruction = reconstruction[0]
    reconstruction = reconstruction.cpu().detach().numpy()

    mse = (reconstruction - test_expression) ** 2
    print('Mean square error reconstructed expression:', np.mean(mse))
    print('Mean values of test expression:', test_expression.mean())
    print('Mean values of reconstructed expression:', reconstruction.mean())

    print('part of test expression <0.5 mean:', (test_expression < 0.5).mean())
    print('part reconstructed exrression <0.5 mean:', (reconstruction < 0.5).mean())

    residual_transfer = ge_transfer_raw.cpu().numpy() - transfer_expression_np
    res_norm = (residual_transfer**2).mean(1).reshape((-1, cfg.count_classes))#l2 norm,
    res_equal = class_ohe_test.argmax(1)
    res_equal = res_equal if isinstance(res_equal, np.ndarray) else res_equal.cpu().detach().numpy()
    print('Calibration accuracy', (res_norm.argmin(1) == res_equal).mean())


    test_latents = vae_model.latents(test_expression_tensor, class_ohe_test_tensor)
    transfered_latents = vae_model.latents(ge_transfer_raw, target_classes_transfer)

    #train
    celltype_clf_expr = None
    celltype_clf_latents = None
    form_clf_expr = None
    form_clf_latents = None
    if not isinstance(test_latents, torch.Tensor):
        test_latents = torch.Tensor(test_latents)
    if not isinstance(transfered_latents, torch.Tensor):
        transfered_latents = torch.Tensor(transfered_latents)
    if cfg.use_cuda and cuda_is_available():
        test_latents = test_latents.cuda()
        transfered_latents = transfered_latents.cuda()

    if pretrained_classifiers is None:
        cfg.classifier_input_dim = cfg.input_dim
        celltype_clf_expr, form_clf_expr = train_classifiers(
                                                    cfg, annot_dataloader,
                                                    celltype_test.shape[1],
                                                    cfg.count_classes
                                                )
        cfg.classifier_input_dim = vae_model.latent_dim
        latents_dataset = torch.utils.data.TensorDataset(
            test_latents,
            class_ohe_test_tensor,
            torch.randn(test_latents.shape[0], cfg.count_labels)
        )
        latents_dataloader = torch.utils.data.DataLoader(latents_dataset,
                                                         batch_size=cfg.batch_size,
                                                         shuffle=True,
                                                         drop_last=True)
        celltype_clf_latents, form_clf_latents = train_classifiers(
                                                    cfg, latents_dataloader,
                                                    celltype_test.shape[1],
                                                    cfg.count_classes
                                                )
    else:
        celltype_clf = pretrained_classifiers[0]
        form_clf = pretrained_classifiers[1]


    if isinstance(test_latents, Tensor):
        test_latents_np = test_latents.cpu().detach().numpy()
    if isinstance(transfered_latents, Tensor):
        transfered_latents_np = transfered_latents.cpu().detach().numpy()

    # entropy batch mixing
    '''
    ebm_score = {}
    ebm_score['test'] = [0. for i in range(1)]
    ebm_score['transfered'] = [0. for i in range(1)]
    for n in range(1):
        for lat, bind, key in zip(
                       (test_latents_np, transfered_latents_np),
                       (class_ohe_test_tensor, target_classes_transfer),
                       ebm_score.keys()):

            batch_ind = bind.argmax(1).cpu().numpy()
            ind = list(combinations(np.unique(batch_ind), 2))
            for i in ind:
                a = (batch_ind == i[0])
                b = (batch_ind == i[1])
                condition = np.logical_or(a, b)

                #Important breakpoint
                ebm_score['test'][n] += entropy_batch_mixing(
                                  lat[condition], batch_ind[condition]
                              )
                ebm_score['transfered'][n] += entropy_batch_mixing(
                                         lat[condition], batch_ind[condition]
                                       )

            if len(ind) > 0:
                ebm_score[key][n] /= len(ind)


    # KNN Purity
    test_purity = []
    transfered_purity = []
    #for i in tqdm(range(2, 501)):
    test_purity.append(get_knn_purity(test_latents_np, celltype_test.argmax(1)))
    transfered_purity.append(get_knn_purity(transfered_latents_np, target_classes_transfer.argmax(dim=1).cpu().numpy()))
    '''

    print('\n')

    celltype_train_labelenc = annot_dataloader.dataset.tensors[2].argmax(1).cpu().numpy()
    celltype_test_raw_labelenc = celltype_test.argmax(1)
    celltype_test_labelenc = init_celltypes_transfer.argmax(1).cpu().numpy()
    #form_test_labelenc = init_classes_transfer.argmax(1).cpu().numpy()
    form_test_labelenc = target_classes_transfer.argmax(1).cpu().numpy()

    predicted_celltype_test_raw = celltype_clf_expr(test_expression_tensor)
    predicted_celltype_test = celltype_clf_expr(transfer_expression_tensor)
    predicted_celltype_train = celltype_clf_expr(annot_dataloader.dataset.tensors[0])
    predicted_form_test = form_clf_expr(transfer_expression_tensor)
    #predict celltypes and batch indices by model's latents
    predicted_celltype_test_latents = celltype_clf_latents(test_latents)
    predicted_celltype_transfer_latents = celltype_clf_latents(transfered_latents)
    predicted_form_latents = form_clf_latents(transfered_latents)

    # classifier on expression
    predicted_celltype_test_labelenc = predicted_celltype_test.argmax(1).cpu().detach().numpy()
    predicted_celltype_test_raw_labelenc = predicted_celltype_test_raw.argmax(1).cpu().detach().numpy()
    predicted_celltype_train_labelenc = predicted_celltype_train.argmax(1).cpu().detach().numpy()

    sameclass_mask = (init_classes_transfer.argmax(1) == target_classes_transfer.argmax(1)).cpu().numpy().astype('bool')

    celltype_train_accuracy = (predicted_celltype_train_labelenc == celltype_train_labelenc).mean()
    celltype_test_raw_accuracy = (predicted_celltype_test_raw_labelenc == np.array(celltype_test_raw_labelenc)).mean()
    celltype_test_accuracy = (predicted_celltype_test_labelenc == celltype_test_labelenc).mean()

    celltype_notransfer_accuracy = (predicted_celltype_test_labelenc == celltype_test_labelenc).mean(0)
    recon_confusion = confusion_matrix(celltype_test_labelenc[sameclass_mask],
                                       predicted_celltype_test_labelenc[sameclass_mask])
    print('Confusion matrix | CELLTYPE RECONSTRUCTION')
    print(recon_confusion)

    from sklearn.metrics import classification_report
    rep = classification_report(
        celltype_test_labelenc[sameclass_mask],
        predicted_celltype_test_labelenc[sameclass_mask],
        output_dict=True
    )
    df = pd.DataFrame(rep).transpose()
    print(df)
    print('Celltype reconstruction report')

    if (~sameclass_mask).any():
        celltype_transfer_accuracy = (predicted_celltype_test_labelenc[~sameclass_mask] == celltype_test_labelenc[~sameclass_mask]).mean(0)
        transfer_confusion = confusion_matrix(celltype_test_labelenc[~sameclass_mask],
                                              predicted_celltype_test_labelenc[~sameclass_mask])
        print('Confusion matrix | CELLTYPE TRANSFER')
        print(transfer_confusion)

        print('Celltype transfer report')
        rep = classification_report(
            celltype_test_labelenc[~sameclass_mask],
            predicted_celltype_test_labelenc[~sameclass_mask],
            output_dict=True
        )
        df = pd.DataFrame(rep).transpose()
        print(df)

        print('Form transfer report')
        pred_f_t = predicted_form_test if isinstance(predicted_form_test, np.ndarray) else predicted_form_test.argmax(1).cpu().detach().numpy()
        rep = classification_report(
            form_test_labelenc[~sameclass_mask],
            pred_f_t[~sameclass_mask],
            output_dict=True
        )
        df = pd.DataFrame(rep).transpose()
        print(df)
    else:
        celltype_transfer_accuracy = "Not enoug classes for transfering"
    predicted_form_test = predicted_form_test.argmax(1).cpu().detach().numpy()
    form_test_accuracy = (predicted_form_test == form_test_labelenc).mean(0)
    recon_confusion = confusion_matrix(form_test_labelenc[sameclass_mask],
                                       predicted_form_test[sameclass_mask],
                                       normalize='all')
    print('Confusion matrix | FORM RECONSTRUCTED')
    print(recon_confusion)

    transfer_confusion = confusion_matrix(form_test_labelenc[~sameclass_mask],
                                          pred_f_t[~sameclass_mask],
                                          normalize='all')
    print('Confusion matrix | FORM TRANSFER')
    print(transfer_confusion)

    rep = classification_report(
        form_test_labelenc[~sameclass_mask],
        pred_f_t[~sameclass_mask],
        output_dict=True
    )
    df = pd.DataFrame(rep).transpose()
    print(df)
    print('Form reconstructed report')


    # classifier on latents
    predicted_celltype_latents_labelenc = predicted_celltype_test_latents.argmax(1).cpu().detach().numpy()
    predicted_celltype_transfer_latents_labelenc = predicted_celltype_transfer_latents.argmax(1).cpu().detach().numpy()

    #celltype_test_latents_accuracy = (predicted_celltype_latents_labelenc == celltype_test_raw_labelenc).mean()
    celltype_notransfer_latents_accuracy = (np.array(predicted_celltype_latents_labelenc) == np.array(celltype_test_raw_labelenc)).mean(0)
    if (~sameclass_mask).any():
        celltype_transfer_latents_accuracy = (predicted_celltype_transfer_latents_labelenc[~sameclass_mask] == celltype_test_labelenc[~sameclass_mask]).mean()
    else:
        celltype_transfer_latents_accuracy = "Not enoug classes for transfering"

    #reconstructed means transfered to the same class

    print('On expression:')
    print('\tCell type prediction accuracy [vae transfered data]', celltype_test_accuracy,
            '< with the same class transfer')
    print('\tCell type prediction accuracy [reconstructed data]', celltype_notransfer_accuracy)
    print('\tCell type prediction accuracy [transfered to another classes]', celltype_transfer_accuracy)
    print('\tClass prediction accuracy:', form_test_accuracy)
    print('On latents:')
    #print('\tCell type prediction accuracy [vae transfered data]', celltype_test_latents_accuracy,
    #        '< with the same class transfer')
    print('\tCell type prediction accuracy [reconstructed data]', celltype_notransfer_latents_accuracy)
    print('\tCell type prediction accuracy [transfered to another classes]', celltype_transfer_latents_accuracy)

    #print('K neighbors purity [vae test data]:', test_purity)
    #print('K neighbors purity [transfer]:', transfered_purity)
    #print('Entropy batch mixing [vae test data]:', ebm_score['test'])
    #print('Entropy batch mixing [transfered data]:', ebm_score['transfered'])


    return {
        'MSE': float(np.mean(mse)),
        'Calibration accuracy': float((res_norm.argmin(1) == res_equal).mean()),
        'Cell type prediction accuracy [train]:': celltype_train_accuracy,
        'Cell type prediction accuracy expression [test hold out data]': celltype_test_raw_accuracy,
        #'Cell type prediction accuracy latents [test hold out data]': celltype_test_latents_accuracy,
        #'Cell type prediction accuracy expression [vae transfered data]': celltype_test_accuracy,
        #'Cell type prediction accuracy latents [vae transfered data]': celltype_test_accuracy,
        'Cell type prediction accuracy expression [reconstructed data]': celltype_notransfer_accuracy,
        'Cell type prediction accuracy latents [reconstructed data]': celltype_notransfer_latents_accuracy,
        'Cell type prediction accuracy expression [transfered to another classes]': celltype_transfer_accuracy,
        'Cell type prediction accuracy latents [transfered to another classes]': celltype_transfer_latents_accuracy,
        'Class prediction accuracy': form_test_accuracy,
        #'K neighbors purity [vae test data]': test_purity,
        #'K neighbors purity [transfer]': transfered_purity,
        #'Entropy batch mixing [test data]': ebm_score['test'],
        #'Entropy batch mixing [transfer data]': ebm_score['transfered']
    }


if __name__ == '__main__':
    pass
