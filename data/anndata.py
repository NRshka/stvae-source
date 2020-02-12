'''
Module with functions to work with scanpy AnnData objects
'''

from typing import Optional
from scanpy import AnnData
from pandas import DataFrame
import numpy as np



def _make_anndata(X: np.ndarray, observation: DataFrame) -> AnnData:
    '''Make a scanpy AnnData object out of pieces

        :Param X: numpy array with biological data, e.g. expression
        :Param observation: annotation for biological data
        :Return: AnnData object
    '''
    return AnnData(X, observation)


def make_anndata(expression: np.ndarray,
                 batch_index: Optional[np.ndarray] = None,
                 batch_name: Optional[str] = None,
                 cell_info: Optional[np.ndarray] = None,
                 cell_info_name: Optional[str] = None) -> AnnData:
    '''Convert data to scanpy AnnData format

        :Param expression: gene expression or same thing, X at the dataset,
            shape [count_observation X cont genes]
        :Param batch_index: indeces of batches for each observation, shape
            [count_observation]
        :Param cell_info any biological information what must be saved after
            autoencoder, shape [cout_observation]
        :Param batch_name: name of the column with batch_index
        :Param cell_info_name: name of the column with cell_info

        NOTE: batch_index and cell_info will be flatten

        TODO: custom annotation data
    '''

    annotation = {}
    if batch_index or batch_name:
        if batch_index and batch_name:
            batch_index = batch_index.reshape(-1) #make observation flatten
            if batch_index.shape[0] != expression.shape[0]:
                raise ValueError(f"Count observation in expression and \
                                 batch_index must be same, got \
                                 {expression.shape[0]} != \
                                 {batch_index.shape[0]}")
            annotation[batch_name] = batch_index
        else:
            raise ValueError(f"Expected both batch_index and batch_name or \
                            noone, but got {type(batch_index)} and \
                             {type(batch_name)}")
    if cell_info or cell_info_name:
        if cell_info and cell_info_name:
            cell_info = cell_info.reshape(-1) #make it flatten for table
            if cell_info.shape[0] != expression.shape[0]:
                raise ValueError(f"Count observation in expression and \
                                    batch_index must be same, got \
                                    {expression.shape[0]} != \
                                    {batch_index.shape[0]}")

            annotation[cell_info_name] = cell_info
        else:
            raise ValueError(f"Expected both cell_info and cell_info_name or \
                            noone, but got {type(cell_info)} and \
                             {type(batch_name)}")

    observation = DataFrame(annotation)

    return _make_anndata(expression, observation)
