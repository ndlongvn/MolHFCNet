import torch
import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    GroupKFold, 
    KFold, 
    StratifiedKFold,
)
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

class Splitter(object):
    """
    The Splitter class is responsible for splitting a dataset into train and test sets 
    based on the specified method.
    """
    def __init__(self, split_method='5fold_random', seed=42):
        """
        Initializes the Splitter with a specified split method and random seed.

        :param split_method: (str) The method for splitting the dataset, in the format 'Nfold_method'. 
                             Defaults to '5fold_random'.
        :param seed: (int) Random seed for reproducibility in random splitting. Defaults to 42.
        """
        self.n_splits, self.method = int(split_method.split('fold')[0]), split_method.split('_')[-1]    # Nfold_xxxx
        self.seed = seed
        self.splitter = self._init_split()

    def _init_split(self):
        """
        Initializes the actual splitter object based on the specified method.

        :return: The initialized splitter object.
        :raises ValueError: If an unknown splitting method is specified.
        """
        if self.method == 'random':
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        elif self.method == 'scaffold' or self.method == 'group':
            splitter = GroupKFold(n_splits=self.n_splits)
        elif self.method == 'stratified':
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        else:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))

        return splitter

    def split(self, data, target=None, group=None):
        """
        Splits the dataset into train and test sets based on the initialized method.

        :param data: The dataset to be split.
        :param target: (optional) Target labels for stratified splitting. Defaults to None.
        :param group: (optional) Group labels for group-based splitting. Defaults to None.

        :return: An iterator yielding train and test set indices for each fold.
        :raises ValueError: If the splitter method does not support the provided parameters.
        """
        try:
            return self.splitter.split(data, target, group)
        except:
            raise ValueError('Unknown splitter method: {}fold - {}'.format(self.n_splits, self.method))



# splitter function

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

# dataset, smiles_list, task_idx=None, null_value=0,
#                    frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0
def random_scaffold_split(dataset, smiles_list, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):
    print('Random scaffold split ...........')
    rng = np.random.RandomState(random_seed)
    scaffolds = defaultdict(list)
    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)
    idxs= list(scaffolds.keys())
    idxs = rng.permutation(idxs)
    scaffold_sets = [scaffolds[idx] for idx in idxs]

    n_total_valid = int(ration_valid * len(dataset) * (1-ratio_test))
    n_total_test = int(ratio_test * len(dataset))
    print('Num train: {}, Num val {}, Num test {}'.format(len(smiles_list)-n_total_test-n_total_valid, n_total_valid, n_total_test))
    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        elif len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0
    assert len(set(train_idx)) + len(set(test_idx))+ len(set(valid_idx)) == len(smiles_list), 'total not match'

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    return train_dataset, valid_dataset, test_dataset

def random_split(dataset, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):
    print('Random split ...........')
    data = list(range(len(dataset)))
    X_, X_test = train_test_split(data, test_size=ratio_test, random_state=random_seed)
    X_train, X_val = train_test_split(X_, test_size=ration_valid, random_state=random_seed)
    assert len(X_train) + len(X_val) + len(X_test) == len(data)
    # print('train: {}, valid: {}, test: {}'.format(len(X_train), len(X_val), len(X_test)))
    print('Num train: {}, Num val {}, Num test {}'.format(len(X_train), len(X_val), len(X_test)))
    train_dataset = dataset[torch.tensor(X_train)]
    valid_dataset = dataset[torch.tensor(X_val)]
    test_dataset = dataset[torch.tensor(X_test)]
    
    return train_dataset, valid_dataset, test_dataset


def stra_split(dataset, task_idx=None, null_value=0, 
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42,
                 smiles_list=None):
    """

    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :param smiles_list: list of smiles corresponding to the dataset obj, or None
    :return: train, valid, test slices of the input dataset obj. If
    smiles_list != None, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if seed is not None:
          np.random.seed(seed)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value  # boolean array that correspond to non null values
        idx_array = np.where(non_null)[0]
        dataset = dataset[torch.tensor(idx_array)]  # examples containing non
        # null labels in the specified task_idx
    else:
        pass

    num_mols = len(dataset)
    # print('num_mols:',num_mols)
    # random.seed(seed)
    all_idx = list(range(num_mols))
    # random.shuffle(all_idx)

    labels = [data.y.item() for data in dataset]
    sortidx = np.argsort(labels)
    split_cd = 10
    train_cutoff = int(np.round(frac_train * split_cd))
    valid_cutoff = int(np.round(frac_valid * split_cd)) + train_cutoff

    train_idx = np.array([])
    valid_idx = np.array([])
    test_idx = np.array([])

    while sortidx.shape[0] >= split_cd:
      sortidx_split, sortidx = np.split(sortidx, [split_cd])
      shuffled = np.random.permutation(range(split_cd))
      train_idx = np.hstack([train_idx, sortidx_split[shuffled[:train_cutoff]]])
      valid_idx = np.hstack(
          [valid_idx, sortidx_split[shuffled[train_cutoff:valid_cutoff]]])
      test_idx = np.hstack([test_idx, sortidx_split[shuffled[valid_cutoff:]]])

    # Append remaining examples to train
    if sortidx.shape[0] > 0:
      np.hstack([train_idx, sortidx])

    print('train_idx',train_idx.shape,train_idx.dtype,train_idx)

    train_dataset = dataset[torch.tensor(train_idx.astype(int))]
    valid_dataset = dataset[torch.tensor(valid_idx.astype(int))]
    test_dataset = dataset[torch.tensor(test_idx.astype(int))]

    if not smiles_list:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in valid_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)


def cv_random_split(dataset, fold_idx = 0,
                   frac_train=0.9, frac_valid=0.1, seed=0,
                 smiles_list=None):
    """

    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :param smiles_list: list of smiles corresponding to the dataset obj, or None
    :return: train, valid, test slices of the input dataset obj. If
    smiles_list != None, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """

    np.testing.assert_almost_equal(frac_train + frac_valid, 1.0)

    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    # labels = [data.y.item() for data in dataset]
    labels = [data.y.item()//100 for data in dataset]

    idx_list = []

    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, val_idx = idx_list[fold_idx]

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(val_idx)]

    return train_dataset, valid_dataset


