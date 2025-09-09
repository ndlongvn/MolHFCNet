import random
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse as sp

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [1 if x == s else 0 for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]

# Get node's feature 
def atom_attr(mol, explicit_H=True, use_chirality=True):
    feat = []
    # print("*"*100)
    for i, atom in enumerate(mol.GetAtoms()):
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding(atom.GetDegree(),
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        # print("result 1", results)
        results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        # print("result 2", results)
        results += [atom.HasProp('_ChiralityPossible')]
        
        try:
            results = results + onehot_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) #+ [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [0, 0]
        # print("result 3", results)
        results= np.array(results, dtype=bool).astype(int)
        feat.append(results)


    return np.array(feat)

# Get edge's index 

def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if mol.GetAtomWithIdx(i).GetSymbol() == 'H' or mol.GetAtomWithIdx(j).GetSymbol() == 'H':
                    continue
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)

# Encode mol to graph
def mol2graph(mol):
    if mol is None: return None
    node_attr = atom_attr(mol)
    edge_index, edge_attr = bond_attr(mol)
    
    FP_list = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)
    FP_list = np.array(FP_list).reshape(1, -1)
    
    generator = RDKit2DNormalized()
    features_map = generator.process(mol)
    arr = np.array(list(features_map))
    arr_ = arr[1:].reshape(1, -1)

    # print(node_attr)
    data = Data(
        x=torch.FloatTensor(node_attr),
        edge_index=torch.LongTensor(edge_index).t(),
        edge_attr=torch.FloatTensor(edge_attr),
        fingerprints=torch.FloatTensor(FP_list),
        descriptors=torch.FloatTensor(arr_),
        smiles = None,
        y=None  # None as a placeholder
    )
    return data

def get_encode(smilesList, target):
    data_list = []
    for i, smi in enumerate(tqdm(smilesList)):
        try:
            mol = MolFromSmiles(smi)
            data = mol2graph(mol)
            if data is not None:
                data.y = torch.FloatTensor([target[i]]) # FoatTensor for regression
                data.smiles = smi
                data_list.append(data)
        except:
            continue
    return data_list


def get_encode_multi(smilesList, target):
    data_list = []
    for i, smi in enumerate(tqdm(smilesList)):

        mol = MolFromSmiles(smi)
        data = mol2graph(mol)
        if data is not None:
            label = []
            for idx, value in enumerate(target[i]):
                if np.isnan(value):
                    label.append(6)
                else:
                    label.append(value)
            data.y = torch.LongTensor([label])
            data.smiles = smi
            data_list.append(data)
    return data_list

def get_encode_covid(smilesList, target):
    data_list = []
    for i, smi in enumerate(tqdm(smilesList)):

        mol = MolFromSmiles(smi)
        data = mol2graph(mol)
        if data is not None:
            label = []
            for idx, value in enumerate(target[i]):
                if np.isnan(value):
                    label.append(6)
                else:
                    label.append(value)
            # print(len(label))
            data.y = torch.FloatTensor([label])
            data.smiles = smi
            data_list.append(data)
    return data_list

import torch
import numpy as np
from rdkit.Chem import MolFromSmiles
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# copy from xiong et al. attentivefp
class ScaffoldGenerator(object):
    """
    Generate molecular scaffolds.
    Parameters
    ----------
    include_chirality : : bool, optional (default False)
        Include chirality in scaffolds.
    """

    def __init__(self, include_chirality=False):
        self.include_chirality = include_chirality

    def get_scaffold(self, mol):
        """
        Get Murcko scaffolds for molecules.
        Murcko scaffolds are described in DOI: 10.1021/jm9602928. They are
        essentially that part of the molecule consisting of rings and the
        linker atoms between them.
        Parameters
        ----------
        mols : array_like
            Molecules.
        """
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=self.include_chirality)

# copy from xiong et al. attentivefp
def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    engine = ScaffoldGenerator(include_chirality=include_chirality)
    scaffold = engine.get_scaffold(mol)
    return scaffold

# copy from xiong et al. attentivefp
def split(scaffolds_dict, smiles_tasks_df, sample_size, random_seed=0):
    # Chia làm sao đảm bảo tý lệ của cần chia ví dụ 0.9<10%<1.1 
    count = 0
    optimal_count = 0.1 * len(smiles_tasks_df)
    while (count < optimal_count * 0.8 or count > optimal_count * 1.2):
        random_seed += 1
        random.seed(random_seed)
        scaffold = random.sample(list(scaffolds_dict.keys()), sample_size)
        count = sum([len(scaffolds_dict[scaffold]) for scaffold in scaffold])
        index = [index for scaffold in scaffold for index in scaffolds_dict[scaffold]]
    return scaffold, index

# copy from xiong et al. attentivefp
# def scaffold_randomized_spliting(smiles_tasks_df, random_seed=8, ratio_test= 0.1, ration_valid= 0.1):
#     print('generating scaffold......')
#     scaffold_list = []
#     all_scaffolds_dict = {}
#     for index, smiles in enumerate(smiles_tasks_df):
#         scaffold = generate_scaffold(smiles)
#         scaffold_list.append(scaffold)
#         if scaffold not in all_scaffolds_dict:
#             all_scaffolds_dict[scaffold] = [index]
#         else:
#             all_scaffolds_dict[scaffold].append(index)

#     samples_size_test = int(len(all_scaffolds_dict.keys()) * ratio_test)
#     samples_size_val = int(len(all_scaffolds_dict.keys()) * ration_valid * (1-ratio_test))
#     test_scaffold, test_index = split(all_scaffolds_dict, smiles_tasks_df, samples_size_test,
#                                       random_seed=random_seed)
#     training_scaffolds_dict = {x: all_scaffolds_dict[x] for x in all_scaffolds_dict.keys() if x not in test_scaffold}
#     valid_scaffold, valid_index = split(training_scaffolds_dict, smiles_tasks_df, samples_size_val,
#                                         random_seed=random_seed)

#     training_scaffolds_dict = {x: training_scaffolds_dict[x] for x in training_scaffolds_dict.keys() if
#                                x not in valid_scaffold}
#     train_index = []
#     for ele in training_scaffolds_dict.values():
#         train_index += ele
#     assert len(train_index) + len(valid_index) + len(test_index) == len(smiles_tasks_df)

#     return train_index, valid_index, test_index


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

from collections import defaultdict
def scaffold_randomized_spliting(dataset, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1):
    print('generating scaffold......')

    rng = np.random.RandomState(random_seed)
    # if isinstance(dataset, str):
    #     dataset= pd.read_csv(dataset)
    # try:
    #     smiles_list= dataset['smiles'].values
    # except:
    #     smiles_list= dataset['SMILES'].values
    smiles_list = dataset
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

    return train_idx, valid_idx, test_idx