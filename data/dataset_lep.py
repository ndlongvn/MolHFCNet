
from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data
from data.molecular_dataset_3d import mol_to_graph_data_obj_simple_3D

import os
import pandas as pd
import scipy as sp
from tqdm import tqdm
from itertools import repeat

class TransformLEP(object):
    """
    Transforms atomic data into graph-based features using RDKit and PyTorch Geometric.
    """
    def __init__(self, dist, maxnum, droph):
        self._dist = dist
        self._maxnum = maxnum
        self._droph = droph

    def _drop_hydrogen(self, df):
        """Drop hydrogen atoms from the DataFrame."""
        df_noh = df[df['element'] != 'H']
        return df_noh

    def _replace(self, df, keep=["H", "C", "N", "O", "F", "S", "P", "S", "Cl"], new="Others"):
        return df

    def _select_env_by_dist(self, df, chain):
        # # TODO: debug
        # temp = df['chain'].to_list()
        # print('chain:', set(temp))
        # """
        # chain: {'L', 'A'}
        # chain: {'L', 'D', 'E', 'B', 'G'}
        # chain: {'C', 'A', 'B', 'L'}
        # chain: {'L', 'A', 'B'}
        # chain: {'L', 'D', 'C'}
        # """

        # Separate pocket and ligand
        ligand = df[df['chain']==chain]
        pocket = df[df['chain']!=chain]
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        key_pts = kd_tree.query_ball_point(ligand_coords, r=self._dist, p=2.0)
        key_pts = np.unique([k for l in key_pts for k in l])
        # Construct the new data frame
        new_df = pd.concat([ pocket.iloc[key_pts], ligand ], ignore_index=True)
        # print('Number of atoms after distance selection:', len(new_df))
        return new_df

    def _select_env_by_num(self, df, chain):
        # Separate pocket and ligand
        ligand = df[df['chain']==chain]
        pocket = df[df['chain']!=chain]
        # Max. number of protein atoms
        num = int(max([1, self._maxnum - len(ligand.x)]))
        # Extract coordinates
        ligand_coords = np.array([ligand.x, ligand.y, ligand.z]).T
        pocket_coords = np.array([pocket.x, pocket.y, pocket.z]).T
        # Select the environment around the mutated residue
        kd_tree = sp.spatial.KDTree(pocket_coords)
        dd, ii = kd_tree.query(ligand_coords, k=len(pocket.x), p=2.0)
        # Get minimum distance to any lig atom for each protein atom
        dist = [min(dd[ii==j]) for j in range(len(pocket.x)) ]
        # Sort indices by distance
        indices = np.argsort(dist)
        # Select the num closest atoms
        indices = np.sort(indices[:num])
        # Construct the new data frame
        new_df = pd.concat([pocket.iloc[indices], ligand], ignore_index=True)
        # print('Number of atoms after number selection:', len(new_df))
        return new_df

    def _generate_features(self, df):
        """Generate features using RDKit and the provided feature generation functions."""
        mol = self._df_to_mol(df)
        data = mol_to_graph_data_obj_simple_3D(mol)
        return data

    def __call__(self, x):
        """Apply transformations to the input data."""
        chain = 'L'  # Select the ligand chain
        x['atoms_active'] = self._replace(x['atoms_active'])
        x['atoms_inactive'] = self._replace(x['atoms_inactive'])
        if self._droph:
            x['atoms_active'] = self._drop_hydrogen(x['atoms_active'])
            x['atoms_inactive'] = self._drop_hydrogen(x['atoms_inactive'])
        x['atoms_active'] = self._select_env_by_dist(x['atoms_active'], chain)
        x['atoms_active'] = self._select_env_by_num(x['atoms_active'], chain)
        x['atoms_inactive'] = self._select_env_by_dist(x['atoms_inactive'], chain)
        x['atoms_inactive'] = self._select_env_by_num(x['atoms_inactive'], chain)


        return x


from atom3d.datasets import LMDBDataset, extract_coordinates_as_numpy_arrays
from rdkit import Chem


from rdkit import Chem
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from data.molecular_dataset_3d import atom_attr, bond_attr


from rdkit import Chem
import numpy as np

def dataframe_to_rdkit_mol(df):
    """
    Convert a DataFrame of atoms to an RDKit molecule.
    :param df: DataFrame containing atom information (element, x, y, z)
    :return: RDKit molecule object
    """
    # Create an editable molecule
    mol = Chem.RWMol()

    # Add atoms to the molecule
    for _, row in df.iterrows():
        element = row['element']
        
        # Handle numeric atomic numbers
        if isinstance(element, (int, float, np.number)):
            atom = Chem.Atom(int(element))  # Convert to integer
        # Handle element symbols
        elif isinstance(element, str):
            atom = Chem.Atom(element)  # Pass as string
        else:
            raise ValueError(f"Unsupported element type: {type(element)}")
        
        mol.AddAtom(atom)

    # Add a conformer with the given positions
    conf = Chem.Conformer(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        conf.SetAtomPosition(i, (row['x'], row['y'], row['z']))  # Use 'x', 'y', 'z' for coordinates
    mol.AddConformer(conf)

    # Sanitize the molecule to infer bonds and other properties
    mol.UpdatePropertyCache(strict=False)  # Ensure properties are updated
    Chem.SanitizeMol(mol)  # Infer bonds, aromaticity, etc.

    return mol

class DatasetLEP(InMemoryDataset):
    def __init__(
        self,
        root,
        split_option,
        dataframe_transformer,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        self.split_option = split_option
        self.dataframe_transformer = dataframe_transformer
        self.dataset = "lep"

        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        super(DatasetLEP, self).__init__(self.root, self.transform, self.pre_transform, self.pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])
        return

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return "geometric_data_processed_{}.pt".format(self.split_option)

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.lmdb_data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}
        print("stats: ", self.stats)

    def convert_units(self, units_dict):
        # TODO: no longer used?
        for key in self.lmdb_data.keys():
            if key in units_dict:
                self.lmdb_data[key] *= units_dict[key]
        self.calc_stats()

    def __lmdb_len__(self):
        return self.num_pts
    
    def __lmdb_getitem__(self, idx):
        return {key: val[idx] for key, val in self.lmdb_data.items()}

    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    def load_lmdb(self):
        key_names = ['index', 'num_atoms', 'charges', 'positions']

        folder = os.path.join(self.root, "raw/split-by-protein/data", self.split_option)
        print("Loading from ", folder)
        dataset = LMDBDataset(folder, transform=self.dataframe_transformer)

        # Load original atoms
        act = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_active'])
        for k in key_names:
            act[k+'_active'] = act.pop(k)
        
        # Load mutated atoms
        ina = extract_coordinates_as_numpy_arrays(dataset, atom_frames=['atoms_inactive'])
        for k in key_names:
            ina[k+'_inactive'] = ina.pop(k)

        # Merge datasets with atoms
        dsdict = {**act, **ina}
        ldict = {'A':1, 'I':0}
        labels = [ldict[dataset[i]['label']] for i in range(len(dataset))]
        dsdict['label'] = np.array(labels, dtype=int)

        self.lmdb_data = {key: torch.from_numpy(val) for key, val in dsdict.items()}
        print("Done loading from {}.".format(folder))
        return

    def preprocess_lmdb(self):
        # Get the size of all parts of the dataset
        ds_sizes = [len(self.lmdb_data[key]) for key in self.lmdb_data.keys()]
        # Make sure all parts of the dataset have the same length
        for size in ds_sizes[1:]:
            assert size == ds_sizes[0]

        # Set the dataset size
        self.num_pts = ds_sizes[0]

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()
        return

    def dataframe_to_rdkit_mol(self, df):
        """
        Convert a DataFrame of atoms to an RDKit molecule.
        :param df: DataFrame containing atom information (element, x, y, z)
        :return: RDKit molecule object
        """
        # Create an editable molecule
        mol = Chem.RWMol()

        # Add atoms to the molecule
        for _, row in df.iterrows():
            atom = Chem.Atom(row['element'])  # Use the 'element' column for atom type
            mol.AddAtom(atom)

        # Add a conformer with the given positions
        conf = Chem.Conformer(len(df))
        for i, (_, row) in enumerate(df.iterrows()):
            conf.SetAtomPosition(i, (row['x'], row['y'], row['z']))  # Use 'x', 'y', 'z' for coordinates
        mol.AddConformer(conf)

        # Sanitize the molecule to infer bonds and other properties
        mol.UpdatePropertyCache(strict=False)  # Ensure properties are updated
        Chem.SanitizeMol(mol)  # Infer bonds, aromaticity, etc.

        return mol


    def process(self):
        print("Preprocessing LEP {} ...".format(self.split_option))

        # First, load LMDB format
        self.load_lmdb()

        # Second, preprocess using the LMDB format
        self.preprocess_lmdb()

        # Third, transform it into PyG format
        data_list = []
        for i in tqdm(range(self.__lmdb_len__())):
            lmdb_data = self.__lmdb_getitem__(i)
            
            # Create DataFrames for active and inactive states
            df_active = pd.DataFrame({
                'element': lmdb_data["charges_active"][:lmdb_data["num_atoms_active"]],
                'x': lmdb_data["positions_active"][:lmdb_data["num_atoms_active"], 0],
                'y': lmdb_data["positions_active"][:lmdb_data["num_atoms_active"], 1],
                'z': lmdb_data["positions_active"][:lmdb_data["num_atoms_active"], 2],
            })
            df_inactive = pd.DataFrame({
                'element': lmdb_data["charges_inactive"][:lmdb_data["num_atoms_inactive"]],
                'x': lmdb_data["positions_inactive"][:lmdb_data["num_atoms_inactive"], 0],
                'y': lmdb_data["positions_inactive"][:lmdb_data["num_atoms_inactive"], 1],
                'z': lmdb_data["positions_inactive"][:lmdb_data["num_atoms_inactive"], 2],
            })

            # Convert DataFrames to RDKit molecules
            mol_active = dataframe_to_rdkit_mol(df_active)
            mol_inactive = dataframe_to_rdkit_mol(df_inactive)

            # Generate atom features using RDKit
            x_active, positions_active_ = atom_attr(mol_active)
            x_inactive, positions_inactive_ = atom_attr(mol_inactive)

            edge_index_active, _ = bond_attr(mol_active)
            edge_index_active = torch.LongTensor(edge_index_active).t()
    
            edge_index_inactive, _ = bond_attr(mol_inactive)
            edge_index_inactive = torch.LongTensor(edge_index_inactive).t()
            
            # Convert features to tensors
            x_active = torch.tensor(x_active, dtype=torch.float)
            x_inactive = torch.tensor(x_inactive, dtype=torch.float)

            # Extract positions
            positions_active = lmdb_data["positions_active"][:lmdb_data["num_atoms_active"]].float()
            positions_inactive = lmdb_data["positions_inactive"][:lmdb_data["num_atoms_inactive"]].float()

            # assert np.array_equal(positions_active, positions_active_), "positions_active and positions_active_ is not match"
            # assert np.array_equal(positions_inactive, positions_inactive_), "positions_inactive and positions_inactive_ is not match"
            # Extract label
            label = lmdb_data["label"]
            
            # Create a PyG Data object
            data = Data(
                x_active=x_active,
                positions_active=torch.tensor(positions_active_, dtype=torch.float),
                edge_index_active=edge_index_active,
                x_inactive=x_inactive,
                positions_inactive=torch.tensor(positions_inactive_, dtype=torch.float),
                edge_index_inactive=edge_index_inactive,
                y=torch.tensor(label),
            )
            data_list.append(data)
        
        # Apply pre-filtering if specified
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # Apply pre-transformation if specified
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Collate and save the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return


