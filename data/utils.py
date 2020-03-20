from numpy import zeros, arange, ndarray
from numpy.random import randint
from torch import Tensor, randn, cuda
import pandas as pd

from pathlib import Path
from os import listdir


def create_random_classes(batch_size: int, num_classes: int):
    assert isinstance(batch_size, int), TypeError("batch size must be numeric, integer")
    assert batch_size > 0, ValueError("It can't be negative")

    assert isinstance(num_classes, int), TypeError("num_classes muust be numeric, integer")
    assert num_classes > 0, ValueError("It can't be negative")

    res_np = zeros((batch_size, num_classes))
    res_np[arange(batch_size), randint(0, num_classes, batch_size)] = 1.

    return Tensor(res_np)


def add_noise(expr: ndarray, beta: float = 1.0):
    #TODO doc-string
    rand_values = randn(expr.shape)
    if cuda.is_available():
        rand_values = rand_values.cuda()
    res = expr + rand_values * beta

    return res


def get_raw_data(directory: Path, preprocessing=None, indices=False):
    '''preprocess - callable'''
    assert callable(preprocessing) or preprocessing is None
    original_expr_df = pd.read_csv(directory.joinpath('ST1 - original_expression.csv'), index_col=0)
    transfer_annot_df = pd.read_csv(directory.joinpath('ST2 - transfer_annotation.csv'), index_col=0)
    transfer_annot_df = transfer_annot_df.drop_duplicates(subset='sample_ID')

    expression = original_expr_df.to_numpy()
    class_ohe = pd.get_dummies(transfer_annot_df['Init state']).to_numpy()
    cell_type = pd.get_dummies(transfer_annot_df['Cell type']).to_numpy()
    cell_indices = list(original_expr_df.index)

    if preprocessing:
        expression = preprocessing(expression)
        class_ohe = preprocessing(class_ohe)
        cell_type = preprocessing(cell_type)

    if indices:
        return expression, class_ohe, cell_type, cell_indices
    return expression, class_ohe, cell_type
