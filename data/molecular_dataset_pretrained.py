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
from torch_geometric.utils import subgraph, to_networkx
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')  # Disable RDKit warnings

from rdkit.Chem import Descriptors, QED
from rdkit.Contrib.SA_Score.sascorer import calculateScore as sa_score
from multiprocessing import Pool, cpu_count

from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import pickle
from data.molecular_dataset import inner_smi2coords

import torch
from tqdm import tqdm

from functools import partial
import random
from copy import deepcopy
import math
import threading


def extract_molecular_properties(mol, return_dict=False):
    if mol is None:
        return None

    # Extract molecular properties
    properties = {
        "MolWt": Descriptors.MolWt(mol),                 # Molecular Weight
        "LogP": Descriptors.MolLogP(mol),               # Octanol-water partition coefficient
        "TPSA": Descriptors.TPSA(mol),                  # Topological Polar Surface Area
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),  # Number of Rotatable Bonds
        "QED": QED.qed(mol),                            # QED score (Drug-likeness)
        "SA": sa_score(mol)                             # Synthetic Accessibility score
    }
    if return_dict:
        return properties
    return list(properties.values())

def scaler_proprety(train_smiles_data):
    # Initialize scalers for each property

    scalers = {
        "MolWt": RobustScaler(),
        "LogP": RobustScaler(),
        "TPSA": RobustScaler(),
        "RotatableBonds": RobustScaler(),
        "QED": MinMaxScaler(),
        "SA": MinMaxScaler()
    }

    # Collect property values for each property
    property_values = {key: train_smiles_data[:, idx] for idx, key in enumerate(scalers.keys())}
    # print(list(property_values.values())[0].shape)
    
    # Fit scalers
    for key, scaler in scalers.items():
        scalers[key].fit(np.array(property_values[key]).reshape(-1, 1))
    
    return scalers



def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    x, conf_coords, edge_index, edge_attr = inner_smi2coords(mol)
    
    x = torch.FloatTensor(x)
    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.FloatTensor(edge_attr)
    positions = torch.FloatTensor(conf_coords)
    
    
    FP_list = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512)
    FP_list = np.array(FP_list).reshape(1, -1)
    
    arr_ = extract_molecular_properties(mol, return_dict=False)
    arr_ = np.array(arr_).reshape(1, -1)
    

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions,
                fingerprints=torch.FloatTensor(FP_list),
                descriptors=torch.FloatTensor(arr_))
    return data


# Function to process a single molecule
def process_molecule_v0(args):

    i, rdkit_mol, smiles, labels = args
    # print(i, rdkit_mol, smiles, labels)
    Chem.SanitizeMol(rdkit_mol)
    data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
    # Add mol ID and labels
    data.id = torch.tensor([i])  # ID corresponds to the index in the dataset
    if labels is not None:
        if len(labels.shape) == 2:
            data.y = torch.tensor(labels[i, :])
        else:
            data.y = torch.tensor([labels[i]])

    return data, smiles

def process_molecule_v1(args):
    """
    Processes a single molecule.
    """
    i, rdkit_mol, smiles, labels = args
    try:
        Chem.SanitizeMol(rdkit_mol)
        data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
        data.id = torch.tensor([i])  # ID corresponds to the index in the dataset
        if labels is not None:
            if len(labels.shape) == 2:
                data.y = torch.tensor(labels[i, :])
            else:
                data.y = torch.tensor([labels[i]])
        return data, smiles
    except Exception as e:
        print(f"Error processing molecule {i}: {e}")
        return None, None



def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        return atom_poses
def get_2d_atom_poses(mol):
    """get 2d atom poses"""
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    atom_poses = get_atom_poses(mol, conf)
    return atom_poses

def process_molecule(args):
    """
    Processes a single molecule.
    """
    i, rdkit_mol, smiles, labels = args
    try:
        Chem.SanitizeMol(rdkit_mol)
        data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
        data.id = torch.tensor([i])  # ID corresponds to the index in the dataset
        if labels is not None:
            if len(labels.shape) == 2:
                data.y = torch.tensor(labels[i, :])
            else:
                data.y = torch.tensor([labels[i]])
        return data, smiles
    except Exception as e:
        print(f"Error processing molecule {i}: {e}")
        atom_poses = get_2d_atom_poses(rdkit_mol)
        return atom_poses, smiles

def process_molecule_with_timeout(args, timeout=30):
    """
    Wrapper function to run process_molecule with a timeout using a thread.
    """
    i, rdkit_mol, smiles, labels = args

    # Shared variable to store the result
    result = None

    def worker():
        nonlocal result
        result = process_molecule(args)

    # Create and start the thread
    thread = threading.Thread(target=worker)
    thread.start()

    # Wait for the thread to finish or timeout
    thread.join(timeout=timeout)

    if thread.is_alive():
        # If the thread is still alive after the timeout, it means the function took too long
        print(f"Processing molecule {i} exceeded timeout of {timeout} seconds.")
        # Return the 2D atom poses as a fallback
        atom_poses = get_2d_atom_poses(rdkit_mol)
        return atom_poses, smiles
    else:
        # If the thread finished, return the result
        return result


# Function to use multiprocessing
def process_all_molecules_multiprocessing(rdkit_mol_objs, smiles_list, labels, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()  # Use all available CPU cores

    # Create a list of arguments for each molecule
    if rdkit_mol_objs is None:
        rdkit_mol_objs = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    args_list = [(i, rdkit_mol_objs[i], smiles_list[i], labels) for i in range(len(smiles_list))]
    
    data_list = []
    data_smiles_list = []

    # Use Pool to process molecules in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_molecule_with_timeout, args_list), total=len(smiles_list), desc="Processing molecules"))
        
    
    # Use multiprocessing to process molecules in parallel
    # with Pool(num_workers) as pool:
    #     results = list(tqdm(pool.imap(process_molecule, args_list), total=len(smiles_list), desc="Processing molecules"))
    
    # Separate data and smiles
    for data, smiles in results:
        if smiles is not None:
            data_list.append(data)
            data_smiles_list.append(smiles)
        
    return data_list, data_smiles_list


class MoleculeMaskingDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 mask_ratio,
                 test_set_ratio,
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
        self.mask_ratio = mask_ratio

        super(MoleculeMaskingDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            
        test_set = int(len(self.data.descriptors)*test_set_ratio)
            
        self.scalers = scaler_proprety(self.data.descriptors.numpy()[:len(self.data)-test_set])
        
            
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))
    
    def attr_mask(self, data):

        node_num, _ = data.x.size()
        mask_num = int(node_num * self.mask_ratio)

        token = torch.FloatTensor([0]*len(data.x[0]))
        pos_token = torch.FloatTensor([0]*len(data.positions[0]))
        idx_mask = np.random.choice(
            node_num, mask_num, replace=False)
        
        mask_node = data.x[idx_mask].clone() # save the masked node feature
        mask_pos = data.positions[idx_mask].clone() # save the masked node position
        
        data.positions[idx_mask] = pos_token
        data.x[idx_mask] = token
        data.mask_idx = torch.BoolTensor([0 if i not in idx_mask else 1 for i in range(node_num)])
        data.mask_node = mask_node
        data.mask_pos = mask_pos
        return data
    
    def edge_mask(self, data):

        _, edge_num = data.edge_index.size()
        mask_num = int(edge_num * self.mask_ratio)

        token = torch.FloatTensor([0]*len(data.edge_attr[0]))
        idx_mask = np.random.choice(
            edge_num, mask_num, replace=False)
        
        mask_edge = data.edge_attr[idx_mask].clone() # save the masked edge feature
        data.edge_attr[idx_mask] = token
        data.mask_edge_start = torch.BoolTensor([0 if i not in idx_mask else 1 for i in range(edge_num)])
        # data.mask_edge_end = torch.BoolTensor([0 if i not in idx_mask else 1 for i in range(edge_num)])
        data.mask_edge = mask_edge
        return data
    
    def subgraph(self, data):
        G = to_networkx(data)
        node_num, _ = data.x.size()
        sub_num = int(node_num * (1 - self.mask_ratio))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        # BFS
        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_idx, edge_attr = subgraph(subset=idx_nondrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_attr,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.positions = data.positions[idx_nondrop]
        data.__num_nodes__, _ = data.x.shape
        return data
    
    def remove_subgraph(self, Graph, center, percent=0.1):
        assert percent <= 1, "Percent must be less than or equal to 1"
        G = Graph.copy()  # Create a copy of the graph to avoid modifying the original
        num = int(np.floor(len(G.nodes) * percent))  # Number of nodes to remove
        removed = []  # List to keep track of removed nodes
        temp = [center]  # Start with the center node
        
        while len(removed) < num:
            neighbors = []
            
            # Ensure temp has nodes that still exist in the graph
            temp = [n for n in temp if n in G]

            if not temp:  # Break if no nodes are left in temp
                break

            # Gather neighbors of the current nodes
            for n in temp:
                if n in G:  # Check if the node is still in the graph
                    neighbors.extend([i for i in G.neighbors(n) if i not in temp and i not in removed])

            # Remove nodes from temp
            for n in temp:
                if len(removed) < num:
                    if n in G:  # Ensure the node is still in the graph
                        G.remove_node(n)
                        removed.append(n)
                else:
                    break

            temp = list(set(neighbors))  # Update temp with unique neighbors for the next iteration
        
        return G, removed
    
    
    def augment(self, data):

        ####################
        # Subgraph Masking #
        ####################
        id = data.id
        try:
            edges = data.edge_index.t().tolist() 
            x = data.x.numpy()
            pos = data.positions.numpy()
            
            molGraph = nx.Graph(edges)
            
            N = len(data.x)
            M = len(data.edge_attr)
            
            # Get the graph for i and j after removing subgraphs
            start_i = random.sample(list(range(N)), 1)[0]
            percent_i = random.uniform(0, 0.2)
            G_i, removed_i = self.remove_subgraph(molGraph, start_i, percent=percent_i)
    
    
            atom_remain_indices_i = [i for i in range(N) if i not in removed_i]
            
            # Only consider bond still exist after removing subgraph
            row_i, col_i = [], []
            edge_feat_i= []
            G_i_edges = list(G_i.edges)
    
            for start, end in edges:
                # print(start, end)
                if (start, end) in G_i_edges: 
                    row_i += [start, end]
                    edge_feat_i.append(data.edge_attr[edges.index([start, end])].reshape(1, -1))
                    
                if  (end, start) in G_i_edges:
                    col_i += [end, start]
                    edge_feat_i.append(data.edge_attr[edges.index([end, start])].reshape(1, -1))
            
            edge_index_i = torch.tensor([row_i, col_i], dtype=torch.long)
    
            edge_feat_i = torch.cat(edge_feat_i, dim=0)
    
     
            edge_attr_i = torch.tensor(edge_feat_i, dtype=torch.long)
    
            # atom and edge masking
    
            num_mask_nodes_i = max([0, math.floor(0.25*N)-len(removed_i)])
            
            num_mask_edges_i = max([0, edge_attr_i.size(0)//2 - math.ceil(0.75*M)])
    
            mask_nodes_i = random.sample(atom_remain_indices_i, num_mask_nodes_i)
    
            mask_edges_i_single = random.sample(list(range(edge_attr_i.size(0)//2)), num_mask_edges_i)
    
            mask_edges_i = [2*i for i in mask_edges_i_single] + [2*i+1 for i in mask_edges_i_single]
      
            x_i = deepcopy(x)
            pos_i = deepcopy(pos)
    
            mask_nodes_mask_i = torch.zeros(N, dtype=torch.long)
            mask_nodes_remove_i = torch.zeros(N, dtype=torch.long)
            # mask_nodes_feat_i = torch.zeros(num_mask_nodes_i, len(x[0]), dtype=torch.float)
    
            mask_edges_mask_i = torch.zeros(edge_attr_i.size(0), dtype=torch.long)
            # mask_edges_feat_i = torch.zeros(num_mask_edges_i, 10, dtype=torch.float)
    
            for atom_idx in range(N):
                if (atom_idx in mask_nodes_i) or (atom_idx in removed_i):
                    # x_i[atom_idx,:] = torch.tensor([0]*len(x_i[atom_idx,:]))
                    # pos_i[atom_idx,:] = torch.tensor([0]*len(pos_i[atom_idx,:]))
                    if atom_idx in mask_nodes_i:
                        mask_nodes_mask_i[atom_idx] = 1
                    if atom_idx in removed_i:
                        mask_nodes_remove_i[atom_idx] = 1
    
            for bond_idx in range(edge_attr_i.size(0)):
                if bond_idx in mask_edges_i:
                    mask_edges_mask_i[bond_idx] = 1
                    # edge_attr_i[bond_idx,:] = torch.tensor([0]*len(edge_attr_i[bond_idx,:]))
            
    
            mask_nodes_feat_i = torch.argmax(torch.tensor(x_i[mask_nodes_mask_i.bool(), :16]).clone(), dim=1)
    
            mask_edges_feat_i = torch.argmax(torch.tensor(edge_attr_i[mask_edges_mask_i.bool(), :4]).clone(), dim=1)
    
            # mask atom type
            x_i[mask_nodes_mask_i.bool(), :16]=0
    
            # mask edge type
            edge_attr_i[mask_edges_mask_i.bool(), :4]=0
    
            # remove node feature
            x_i[mask_nodes_remove_i.bool()] = torch.tensor([0]*len(x_i[0]))
    
            
            # mask egde
            mask_pos_i = torch.tensor(pos_i[mask_nodes_mask_i.bool()]).clone()
            pos_i[mask_nodes_mask_i.bool()] = torch.tensor([0]*len(pos_i[0]))
                
            data_i = Data(x=torch.FloatTensor(x_i),
                    edge_index=torch.LongTensor(edge_index_i),
                    edge_attr=edge_attr_i.float(),
                    mask_idx=mask_nodes_mask_i.bool(),
                    mask_edge_start=mask_edges_mask_i.bool(),
                    mask_node=mask_nodes_feat_i.long() ,
                    mask_edge=mask_edges_feat_i.long() ,
                    mask_pos=mask_pos_i.float(),
                    positions=torch.FloatTensor(pos_i))
            
            return data, data_i
        except:
            return data, Data(
                    x=torch.empty(0, 44).float(),  # Empty tensor for node features
                    edge_index=torch.empty(2, 0).long(),  # Empty tensor for edge indices
                    edge_attr=torch.empty(0, 10).float(),  # Empty tensor for edge attributes
                    mask_idx=torch.empty(0).bool(),  # Empty tensor for mask_idx
                    mask_edge_start=torch.empty(0).bool(),  # Empty tensor for mask_edge_start
                    mask_node=torch.empty(0).long(),  # Empty tensor for mask_node
                    mask_edge=torch.empty(0).long(),  # Empty tensor for mask_edge
                    mask_pos=torch.empty(0, 3).float(),  # Empty tensor for mask_pos
                    positions=torch.empty(0, 3).float(),  # Empty tensor for positions (3D coordinates)
                    id=id
                )


    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
            
        
        # scale data
        descriptors = data.descriptors.numpy().reshape(-1)
        normalized_properties = []
        for idx, value in enumerate(descriptors):
            # print(value)
            normalized_value = list(self.scalers.values())[idx].transform([[value]])[0, 0]
            normalized_properties.append(normalized_value)
        properties = np.array(normalized_properties).reshape(1, -1)
        data.descriptors = torch.FloatTensor(properties)
            
        if self.mask_ratio > 0:
            org_data, data = self.augment(data)
        return org_data, data


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
        if self.dataset == 'pretrained':
            smiles_list = read_smiles(self.raw_paths[0])
            processed_data_list = []
            for idx, i in enumerate(range(0, len(smiles_list), 4000)):
                print(f"Batch {i}")
                batch = smiles_list[i:i + 4000]
                save_path = os.path.join(self.processed_dir, f'batch_{i}.pt')
                processed_data_list.append(save_path)
                if os.path.exists(save_path):
                    continue

                data_list, data_smiles_list = process_all_molecules_multiprocessing(None, batch, None)
                
                torch.save((data_list, data_smiles_list), save_path)
                
            data_list = []
            data_smiles_list = []
            for path in processed_data_list:
                data, smiles = torch.load(path)
                data_list.extend(data)
                data_smiles_list.extend(smiles)
                
        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print("data_smiles_list", len(data_smiles_list))

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        for i in processed_data_list:
            os.remove(i)

def read_smiles(data_path):
    smiles_data = []
    with open(data_path, 'r') as f:
        for line in tqdm(f, desc="read"):
            smiles_data.append(line.strip())
    return smiles_data

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

def create_all_datasets():
    #### create dataset
    downstream_dir = [
            'pretrained'
            ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        root = "data/dataset/" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        dataset = MoleculeMaskingDataset(root, dataset=dataset_name, mask_ratio=0.1, test_set_ratio=0.1)
        print(dataset)



# test MoleculeDataset object
if __name__ == "__main__":

    create_all_datasets()

