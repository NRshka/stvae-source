from numpy import zeros, arange, ndarray
from numpy.random import randint
from torch import Tensor, randn
import pandas as pd

from pathlib import Path
from os import listdir


def create_random_classes(batch_size: int, num_classes: int):
    #TODO адекватные сообщения об ошибке
    assert isinstance(batch_size, int), TypeError("INT")
    assert batch_size > 0, ValueError("It can't be negative")
    
    assert isinstance(num_classes, int), TypeError("INT")
    assert num_classes > 0, ValueError("It can't be negative")
    
    res_np = zeros((batch_size, num_classes))
    res_np[arange(batch_size), randint(0, num_classes, batch_size)] = 1.
    
    return Tensor(res_np)


def add_noise(expr: ndarray, beta: float = 1.0):
    #TODO doc-string
    res = expr + randn(expr.shape).cuda() * beta
    
    return res


def get_raw_data(directory: Path):
    original_expr_df = pd.read_csv(directory.joinpath('ST1 - original_expression.csv'), index_col=0)
    transfer_annot_df = pd.read_csv(directory.joinpath('ST2 - transfer_annotation.csv'), index_col=0)
    transfer_annot_df = transfer_annot_df.drop_duplicates(subset='sample_ID')

    expression = original_expr_df.to_numpy()
    class_ohe = pd.get_dummies(transfer_annot_df['Init state']).to_numpy()
    print('Expression shape:', expression.shape)
    print('Class ohe shape:', class_ohe.shape)
    
    return expression, class_ohe
