from pathlib import Path
from typing import Optional, Union, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scvi.dataset import GeneExpressionDataset


class CsvDataset:
    def __init__(self, names: List[Union[Path, str]]):
        dfs = [pd.read_csv(filename) for filename in names]
        self.assign(*dfs)
        self.filter(*dfs)

    def assign(self, expression_df, batches_df, labels_df):
        self.labels = LabelEncoder().fit_transform(labels_df.to_numpy())
        self.n_labels = np.unique(self.labels).shape[0]
        self.batch_indices = LabelEncoder().fit_transform(batches_df.to_numpy())
        self.n_batches = np.unique(self.batch_indices).shape[0]
        self.X = expression_df.to_numpy()
        self.nb_genes = self.X.shape[1]

    def filter(self, expression_df, labels_df, batches_df):
        for checking in (self.labels, self.batch_indices):
            labels, counts = np.unique(checking, return_counts=True)
            for idx, count in enumerate(counts):
                if count < 2:
                    cell_name = labels_df[labels_df.label == labels[idx]].index[0]
                    expression_df = expression_df[expression_df.index != cell_name]
                    labels_df = labels_df[labels_df.index != cell_name]
                    batches_df = batches_df[batches_df.index != cell_name]
        self.assign(expression_df, labels_df, batches_df)

    def __len__(self):
        return self.X.shape[0]

class ScratchDataset(GeneExpressionDataset):
    def __init__(self,
                 expression   : np.ndarray,
                 batch_indices: np.ndarray,
                 labels       : np.ndarray):
        self.X = expression
        self.nb_genes = self.X.shape[1]
        self.labels = labels
        self.n_labels = np.unique(labels).shape[0]
        self.batch_indices = batch_indices
        self.n_batches = np.unique(batch_indices).shape[0]

    def __len__(self):
        return self.X.shape[0]

class MouseDataset(CsvDataset):
    def __init__(self, directory: Union[Path, str],
                 batch_filename: Optional[Union[Path, str]] = None,
                 labels_filename: Optional[Union[Path, str]] = None):
        self.cell_attribute_names = {'labels',
                                     #'local_vars',
                                     #'local_means',
                                     'batch_indices'}
        if isinstance(directory, str):
            directory = Path(directory)
        if batch_filename is None:
            batch_filename = directory / 'batches.csv'
        else:
            if not isinstance(batch_filename, Path):
                batch_filename = Path(batch_filename)
            if not directory in batch_filename.parents:
                batch_filename = directory / batch_filename

        if labels_filename is None:
            labels_filename = directory / 'labels.csv'
        else:
            if not isinstance(labels_filename, Path):
                labels_filename = Path(labels_filename)
            if not directory in labels_filename.parents:
                labels_filename = directory / labels_filename

        expression_df = pd.read_csv(directory / 'ST1 - original_expression.csv',
                                 index_col=0)

        labels_df = pd.read_csv(labels_filename, index_col=0)
        assert all(expression_df.index == labels_df.index)

        batches_df = pd.read_csv(batch_filename, index_col=0)
        assert all(expression_df.index == batches_df.index)

        dfs = (expression_df, labels_df, batches_df)
        self.assign(*dfs)
        self.filter(*dfs)
