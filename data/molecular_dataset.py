import os
import pickle
from itertools import chain, repeat

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
import json
from tqdm import tqdm
from rdkit import RDLogger
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

RDLogger.DisableLog('rdApp.warning')  # Disable RDKit warnings

from data.utils import bond_attr, onehot_encoding_unk, onehot_encoding



def inner_smi2coords(mol, num_conformers=10, prune_rms_thresh=0.5, seed=42, mode='fast', remove_hs=True, return_mol=False):
    try:
        try:
            mol = AllChem.AddHs(mol)
            conformer_ids = AllChem.EmbedMultipleConfs(
                mol,
                numConfs=num_conformers,
                pruneRmsThresh=prune_rms_thresh,
                randomSeed=seed,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
            )
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            # print("atoms: ", atoms)
            assert len(atoms) > 0, 'No atoms in molecule: {}'.format(mol)
            # energies = []
            # for conf_id in conformer_ids:
            #     # Optimize the conformer using the MMFF94 force field
            #     energy = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
            #     energies.append(energy)
                
            # # Find the conformer with the smallest energy
            # min_energy_index = min(range(len(energies)), key=lambda i: energies[i][1])
            # min_energy_conf_id = conformer_ids[min_energy_index]
            
            # min_con = mol.GetConformer(min_energy_conf_id)
            # coordinates = min_con.GetPositions().astype(np.float32)
            energies = []

            props = AllChem.MMFFGetMoleculeProperties(mol)
            energies = []
            for conf_id in conformer_ids:
                ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                if ff is not None:
                    ff.Minimize()
                    energies.append(ff.CalcEnergy())
                else:
                    energies.append(np.inf)

            min_energy_index = int(np.argmin(energies))
            min_energy_conf_id = conformer_ids[min_energy_index]
            
            min_con = mol.GetConformer(min_energy_conf_id)
            coordinates = min_con.GetPositions().astype(np.float32)
    
            
        except:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d

    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms), 3))

    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(mol)

    # get atom feature
    atoms_feat = atom_attr(mol)
    bond_indices, bond_features = bond_attr(mol, use_chirality=True)
  
    if remove_hs:
        non_hydrogen_idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        idx_map = {i: j for j, i in enumerate(non_hydrogen_idx)}  # Original idx -> new idx
        
        # Filter atoms and coordinates
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[non_hydrogen_idx]
        atoms_feat_no_h = atoms_feat[non_hydrogen_idx]

        # print("atoms_no_h: ", atoms_no_h)
        
        if bond_indices.size > 0:
            # Remap bond indices to non-hydrogen subset
            valid_bond_mask = np.all(np.isin(bond_indices, non_hydrogen_idx), axis=1)
            bond_indices_no_h = bond_indices[valid_bond_mask]
            bond_features_no_h = bond_features[valid_bond_mask]
            
            # Adjust bond indices using the mapping
            remapped_indices = np.array([[idx_map[i], idx_map[j]] for i, j in bond_indices_no_h])
            # print("remapped_indices: ", remapped_indices)
        else:
            # print("bond_indices ", bond_indices)
            remapped_indices = np.empty((0, 2), dtype=np.int64)
            bond_features_no_h = np.empty((0, bond_features.shape[1] if bond_features.ndim == 2 else 10), dtype=np.int64)
        
        assert len(atoms_feat_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(mol)
        if return_mol:
            return atoms_feat_no_h, coordinates_no_h, remapped_indices, bond_features_no_h, mol
        return atoms_feat_no_h, coordinates_no_h, remapped_indices, bond_features_no_h
    else:
        if return_mol:
            return atoms_feat, coordinates, bond_indices, bond_features, mol
        return atoms_feat, coordinates, bond_indices, bond_features

    
def atom_attr(mol, explicit_H=True, use_chirality=True):
 
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        # if atom.GetSymbol()=="H":
        #     continue
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other']
        ) + onehot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              onehot_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other'
              ]) + [atom.GetIsAromatic()]
        results = results + onehot_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        results += [atom.HasProp('_ChiralityPossible')]
        try:
            results = results + onehot_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S'])
        except:
            results = results + [0, 0]
        results = np.array(results, dtype=bool).astype(int)
        feat.append(results)
    
    feat = np.array(feat)
    
    return feat


def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    x, conf_coords, edge_index, edge_attr = inner_smi2coords(mol)
    
    
    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.FloatTensor(edge_attr)
    positions = torch.FloatTensor(conf_coords)
    

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data


# Function to process a single molecule
def process_molecule(args):
    i, rdkit_mol, smiles, labels = args

    Chem.SanitizeMol(rdkit_mol)
    # data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
    data = mol_to_graph_data_obj_simple_3D(rdkit_mol)

    data.id = torch.tensor([i])  # ID corresponds to the index in the dataset
    if len(labels.shape) == 2:
        data.y = torch.FloatTensor(labels[i, :].reshape(1, -1))
        
    else:
        data.y = torch.FloatTensor([labels[i]])
    
    return data, smiles

# Function to use multiprocessing
def process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels, num_workers=20):
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPU cores

    # Create a list of arguments for each molecule
    args_list = [(i, rdkit_mol_objs[i], smiles_list[i], labels) for i in range(len(smiles_list))]
    
    data_list = []
    data_smiles_list = []
    
    # Use multiprocessing to process molecules in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_molecule, args_list), total=len(smiles_list), desc="Processing molecules"))
    
    # Separate data and smiles
    for data, smiles in results:
        data_list.append(data)
        data_smiles_list.append(smiles)
    
    return data_list, data_smiles_list


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        # self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])


    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

       

        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)

        elif self.dataset == 'hiv':       
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            processed_data_list = []
            for i in range(0, len(smiles_list), 4000):
                batch = smiles_list[i:i + 4000]
                data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs[i:i + 4000], batch, labels[i:i + 4000])
                save_path = os.path.join(self.processed_dir, f'batch_{i}.pt')
                torch.save((data_list, data_smiles_list), save_path)
                processed_data_list.append(save_path)
            data_list = []
            data_smiles_list = []
            for path in processed_data_list:
                data, smiles = torch.load(path)
                data_list.extend(data)
                data_smiles_list.extend(smiles)
            

        elif self.dataset == 'bace' or self.dataset == "jak1" or self.dataset == "jak2" or self.dataset == "mapk14":
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)


        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            data_list, data_smiles_list = process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels)                    

        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain
def merge_dataset_objs(dataset_1, dataset_2):
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]

    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    for d in dataset_1:
        old_y = d.y
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    for d in dataset_2:
        old_y = d.y
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))

    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset

def create_circular_fingerprint(mol, radius, size, chirality):
    """

    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)



def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['SMILES']
    rdkit_mol_objs_list = [Chem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list.values, rdkit_mol_objs_list, labels.values

def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['SMILES']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    folds = np.array(input_df['class'])

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds, labels.values

def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['SMILES']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                                          rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    print('preprocessed_smiles_list:',len(preprocessed_smiles_list))
    labels = input_df['class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values

def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['SMILES']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values

def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['SMILES']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values


def _load_esol_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values
# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured']
    
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values



def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['SMILES']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # print('label',labels)
    # print('',input_df['Injury, poisoning and procedural complications'].values)
    # print('label',labels.value)
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values



def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def create_all_datasets():
    #### create dataset
    downstream_dir = [
            'bace',
            'bbbp',
            'clintox',
            'esol',
            'freesolv',
            'lipophilicity',
            'sider',
            'tox21',
            'hiv',
            ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        root = "data/dataset/" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        dataset = MoleculeDataset(root, dataset=dataset_name)
        print(dataset)


    # dataset = MoleculeDataset(root = "dataset/chembl_filtered", dataset="chembl_filtered")
    # print(dataset)
    # dataset = MoleculeDataset(root = "dataset/zinc_standard_agent", dataset="zinc_standard_agent")
    # print(dataset)


# test MoleculeDataset object
if __name__ == "__main__":

    create_all_datasets()

