from collections import Iterable
from typing import Union, Optional
import logging
import pickle
import torch
import numpy as np
from numpy import max
from scvi.dataset import GeneExpressionDataset
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from data import get_raw_data, make_anndata


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)

    return obj

def load_weights(model, filename):
    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        def map_location(storage, loc): return storage
    else:
        map_location = None

    state_dict = torch.load(str(filename), map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    model.load_state_dict(state_dict)


def save_weights(model, filename):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    torch.save(model.state_dict(), str(filename))


def init_weights(modules):
    if isinstance(modules, torch.nn.Module):
        modules = modules.modules()

    for m in modules:
        if isinstance(m, torch.nn.Sequential):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.ModuleList):
            init_weights(m_inner for m_inner in m)

        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.xavier_normal_(m.weight.data)
            # m.bias.data.zero_()
            if m.bias is not None:
                m.bias.data.normal_(0, 0.01)

        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()

        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight.data)
            m.bias.data.zero_()


def to_device(obj, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(obj, (list, tuple)):
        return [to_device(o, device) for o in obj]

    if isinstance(obj, dict):
        return {k: to_device(o, device) for k, o in obj.items()}

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = obj.to(device)
    return obj

def load_datasets(cfg, test: bool = False, annot: bool = False,
                  custom_data=None, preprocessing=None, test_size: float = 0.25,
                  indices=None):
    expr, class_ohe, cell_type = None, None, None
    if custom_data is None:
        expr, class_ohe, cell_type = get_raw_data(cfg.data_dir, preprocessing)
    else:
        expr, class_ohe, cell_type = custom_data

    assert not (expr is None or class_ohe is None or cell_type is None), "Any data have not to be None"
    if indices is None:
        indices = np.array(expr.shape[0] * [0])

    train_expression, val_expression, train_class_ohe, val_class_ohe, train_annot, test_annot, train_ind, test_ind = train_test_split(
        expr, class_ohe, cell_type,
        indices,
        random_state=cfg.random_state,
        stratify = class_ohe.argmax(1), test_size=test_size
    )

    full_test_expr = torch.Tensor(val_expression)
    full_test_form = torch.Tensor(val_class_ohe)

    if test:
        val_expression, test_expression, val_class_ohe, test_class_ohe, val_ind, test_ind = train_test_split(
            val_expression, val_class_ohe, test_ind, random_state=cfg.random_state, test_size=0.15
        )
        test_expression_tensor = torch.Tensor(test_expression)
        test_class_ohe_tensor = torch.Tensor(test_class_ohe)

    val_expression_tensor = torch.Tensor(val_expression)
    val_class_ohe_tensor = torch.Tensor(val_class_ohe)
    train_expression_tensor = torch.Tensor(train_expression)
    train_class_ohe_tensor = torch.Tensor(train_class_ohe)
    train_annot_tensor = torch.Tensor(train_annot)
    test_annot_tensor = torch.Tensor(test_annot)


    if cfg.cuda and torch.cuda.is_available():
        val_expression_tensor = val_expression_tensor.cuda()
        val_class_ohe_tensor = val_class_ohe_tensor.cuda()
        train_expression_tensor = train_expression_tensor.cuda()
        train_class_ohe_tensor = train_class_ohe_tensor.cuda()
        if test:
            test_expression_tensor = test_expression_tensor.cuda()
            test_class_ohe_tensor = test_class_ohe_tensor.cuda()
        full_test_expr = full_test_expr.cuda()
        train_annot_tensor = train_annot_tensor.cuda()
        test_annot_tensor = test_annot_tensor.cuda()
        full_test_form = full_test_form.cuda()

    trainset = torch.utils.data.TensorDataset(train_expression_tensor,
                                            train_class_ohe_tensor)
    dataloader_train = torch.utils.data.DataLoader(trainset,
                                            batch_size=cfg.batch_size,
                                            shuffle=True,
                                            #num_workers=cfg.num_workers,
                                            drop_last=True)
    valset = torch.utils.data.TensorDataset(val_expression_tensor,
                                            val_class_ohe_tensor)
    dataloader_val = torch.utils.data.DataLoader(valset,
                                                batch_size=val_expression_tensor.size(0),
                                                shuffle=False,
                                                drop_last=True)

    result = tuple((dataloader_train, dataloader_val))

    if test:
        testset = torch.utils.data.TensorDataset(test_expression_tensor,
                                                test_class_ohe_tensor)
        dataloader_test = torch.utils.data.DataLoader(testset,
                                                batch_size=test_expression_tensor.size(0),
                                                shuffle=False,
                                                drop_last=True)

        result += tuple((dataloader_test,))

    if annot:
        annot_dataset_train = torch.utils.data.TensorDataset(train_expression_tensor,
                                                            train_class_ohe_tensor,
                                                            train_annot_tensor)
        annot_dataloader_train = torch.utils.data.DataLoader(annot_dataset_train,
                                                            batch_size=cfg.batch_size,
                                                            shuffle=True,
                                                            drop_last=True)

        annot_dataset_test = torch.utils.data.TensorDataset(full_test_expr,
                                                            full_test_form,
                                                            test_annot_tensor)
        annot_dataloader_test = torch.utils.data.DataLoader(annot_dataset_test,
                                                            batch_size=full_test_expr.size(0),
                                                            shuffle=False,
                                                            drop_last=True)

        result += tuple((annot_dataloader_train, annot_dataloader_test))

    if indices:
        result += tuple((train_ind, test_ind))

    return result

def get_high_variance_genes(expression: np.ndarray,
                            class_ohe: np.ndarray,
                            cell_type: np.ndarray,
                            genes: Optional[Union[list, tuple, np.ndarray]] = None,
                            n_genes: Optional[int] = None,
                            min_disp: Optional[float] = None,
                            max_disp: Optional[float] = None,
                            argmax: bool = True) -> tuple:
    assert n_genes or min_disp or max_disp, "Choose smth"

    genes_data = None
    if not genes is None:
        genes_data = [("genes", genes)]
        if not isinstance(genes, np.ndarray):
            genes = np.array(genes)
    if argmax:
        class_ohe = class_ohe.argmax(1)
        cell_type = cell_type.argmax(1)
    anndata = make_anndata(expression, class_ohe, "condition",
                           cell_type, "cell_type", genes_data)
    sc.pp.filter_genes_dispersion(anndata,
                                  n_top_genes=n_genes,
                                  min_disp=min_disp,
                                  max_disp=max_disp)

    expr = anndata.X
    expr = expr if isinstance(expr, np.ndarray) else np.array(expr)
    ohe = np.array(anndata.obs["condition"].tolist())
    ohe = OneHotEncoder(sparse=False).fit_transform(
        ohe.reshape(-1, 1)
    )
    celltype = np.array(anndata.obs["cell_type"])
    celltype = OneHotEncoder(sparse=False).fit_transform(
        celltype.reshape(-1, 1)
    )
    if not genes is None:
        return expr, ohe, celltype, anndata.var["genes"]

    return expr, ohe, celltype



def scvi_anndata(dataset_class):
    '''Make scanpy.Anndata from scvi dataset class'''
    ds = dataset_class
    return make_anndata(ds.X, ds.batch_indices, "condition",
                        ds.labels, "cell_type")


def predefined_preprocessing(data, framework: str,
                             data_format: Optional[str] = 'adaptive'):
    '''Prepare for any supported frameworks
    :Param data: data of supoorted type [scanpy.Anndata,
        tuple(at least one numpy.ndarray), scVI dataset like class]
    :Param framework: flag defining one of supported frameworks
    :Param data_format: define what kind of data is passed
    Note: Unrecognized framework param occurs NotImplementedError
        Unrecognized data_format param occurs ValueError
        If type of data unrecognized it occurs TypeError
    '''

    _supported_formats = ['adaptive', 'raw', 'scvi', 'anndata']
    _supported_frameworks = ['stvae', 'scvi', 'scgen']
    framework = framework.lower()

    if not framework in _supported_frameworks:
        raise NotImplementedError(f"{framework} isn't supported value \
                                    {' '.join(_supported_frameworks)}")
    if not data_format in _supported_formats:
        raise ValueError(f"{data_format} is not supported. \
                            Supported values are {''.join(_supported_formats)}")

    if data_format == 'scvi':
        data = scvi_anndata(data)
    elif data_format == 'raw':
        pass
    elif data_format == 'adaptive':
        if isinstance(data, Iterable):
            if len(data) < 1:
                raise ValueError("Data must not be empty")
            _keys = ('batch_index', 'cell_info', 'variables_info')
            anndata_arguments = {_keys[i]: data[i + 1] for i in range(len(data))}
            if len(data) > 1:
                _keys['batch_name'] = 'condition'
            if len(data) > 2:
                _keys['cell_info_name'] = 'cell_type'

            data = make_anndata(data[0], **_keys)
        elif isinstance(data, GeneExpressionDataset):
            data = scvi_anndata(data)
        elif isinstance(data, sc.AnnData):
            pass
        else:
            raise TypeError(f"Unrecognized type of data: {type(data)}")
    if framework == 'scgen':
        sc.pp.normalize_per_cell(data)
        sc.pp.log1p(data)

    return data
