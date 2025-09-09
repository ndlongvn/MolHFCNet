import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import NTXentLoss
import argparse
from munch import Munch
import random

from utils import parse_args
from tqdm import tqdm

from data.molecular_dataset_pretrained import MoleculeMaskingDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection._split import train_test_split

import warnings
warnings.filterwarnings("ignore")
apex_support = False

# for egnn

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(0)

class Pretrain(object):
    def __init__(self, root, dataset, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        
        log_dir = os.path.join(args.save_path, args.gflayer)
        os.makedirs(log_dir, exist_ok=True)

        self.log_dir = log_dir

        self.dataset = MoleculeMaskingDataset(root=os.path.join(root, dataset), dataset=dataset, mask_ratio=args.mask_ratio,
                                              test_set_ratio=args.test_set_ratio)
        
        self.alpha = 0.2
        
        self.beta = 0.2

        self.property_criterion = torch.nn.SmoothL1Loss()
        self.fingerprint_criterion = torch.nn.BCEWithLogitsLoss()
        self.position_loss = torch.nn.SmoothL1Loss()

        self.criterion_atom = torch.nn.CrossEntropyLoss()  # For atom feature prediction
        self.criterion_bond = torch.nn.CrossEntropyLoss()  # For bond feature prediction


    def _step(self, model, xis, xjs, n_iter):
        # get the representations and the projections
        # print("xjs", xjs)
        zis, pro_i, fip_i = model(xis)  # [N,C]

        # get the representations and the projections
        zjs, mask_node_j, mask_edge_j, mask_pos_j = model(xjs, mask_task=True)  # [N,C]
        

        loss_ct = model.info_nce(zis, zjs)

        loss_property = self.property_criterion(pro_i, xis.descriptors)  # Molecular properties
        loss_fingerprint = self.fingerprint_criterion(fip_i, xis.fingerprints)  # Fingerprints


        loss_atom = torch.tensor(0.0)
        loss_bond = torch.tensor(0.0)
        loss_pos = torch.tensor(0.0)
        
        # Atom feature prediction
        if mask_node_j is not None:
            loss_atom = self.criterion_atom(mask_node_j, xjs.mask_node)
        else:
            loss_atom = torch.tensor(0.0)
            
        if self.args.model_type == '2d':
            # Bond feature prediction
            if mask_edge_j is not None:
                loss_bond = self.criterion_bond(mask_edge_j, xjs.mask_edge)
            else:
                loss_bond = torch.tensor(0.0)
                
        if self.args.model_type == '3d':
            # Position prediction
            if mask_pos_j is not None:
                loss_pos = self.position_loss(mask_pos_j, xjs.mask_pos)
            else:
                loss_pos = torch.tensor(0.0)

        return loss_ct, loss_property, loss_fingerprint, loss_atom, loss_bond, loss_pos

    def train(self):
        dataset_size = len(self.dataset)
        test_size = int(self.args.test_set_ratio * dataset_size)
        train_dataset, valid_dataset = self.dataset[:dataset_size - test_size], self.dataset[dataset_size - test_size:]
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4)
        # train_loader, valid_loader = self.dataset.get_data_loaders()

        print("Train size:", len(train_loader.dataset))
        print("Valid size:", len(valid_loader.dataset))

        
        print("Data 0", train_loader.dataset[0])
        
        if self.args.gflayer in ['GAT', 'GIN', 'GCN', 'GTN']:
            from models.graph_2d import MolHFCEncoder
            self.args.model_type = '2d'
        elif self.args.gflayer in ['CFC', 'SGCN']:
            from models.graph_3d import MolHFCEncoder
            self.args.model_type = '3d'
        
        model = MolHFCEncoder(node_dim = train_loader.dataset[0][0].x.shape[1], edge_dim=train_loader.dataset[0][0].edge_attr.shape[1], 
                              base_dim=self.args.base_dim, gflayer=self.args.gflayer,
                              fingerprint_dim=512, num_mol_properties=6)
        
        model = model.to(self.device)
        print("Number of parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-5)

        scheduler = CosineAnnealingLR(
                optimizer, T_max=self.args.epochs-self.args.warm_up, 
                eta_min=0, last_epoch=-1
            )

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.args.epochs):
            # print(f"Epoch {epoch_counter}")
            total_loss = 0.0
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch_counter}', unit='batch') as pbar:
                for bn, (xis, xjs) in enumerate(train_loader):
                    optimizer.zero_grad()
    
                    xis = xis.to(self.device)
                    xjs = xjs.to(self.device)
    
                    loss_ct, loss_property, loss_fingerprint, loss_atom, loss_bond, loss_pos = self._step(model, xis, xjs, n_iter)
                    loss_ct = torch.nan_to_num(loss_ct)
                    loss_property = torch.nan_to_num(loss_property)
                    loss_fingerprint = torch.nan_to_num(loss_fingerprint)
                    loss_atom = torch.nan_to_num(loss_atom)
                    loss_bond = torch.nan_to_num(loss_bond)
                    loss_pos = torch.nan_to_num(loss_pos)
                    

                    loss = self.alpha*loss_ct + loss_property + loss_fingerprint + self.beta*(loss_atom + loss_bond + loss_pos)
    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    total_loss += loss.item()
                    avg_loss = total_loss / (bn + 1)
    
                    optimizer.step()
                    n_iter += 1

                    pbar.set_postfix(
                        avg_loss=f"{avg_loss:.2f}",
                        loss_ct=f"{loss_ct:.2f}",
                        loss_prt=f"{loss_property:.2f}",
                        loss_fps=f"{loss_fingerprint:.2f}",
                        loss_mask=f"{(loss_atom + loss_bond + loss_pos):.2f}"
                    )
                    pbar.update(1)

    
                # validate the model if requested
    
            valid_loss = self._validate(model, valid_loader)
            # print(epoch_counter, bn, valid_loss, '(validation)')
            print(f"Epoch {epoch_counter} Validation loss {valid_loss}")

            if valid_loss < best_valid_loss:
                # save the model weights
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(self.log_dir, f'{self.args.base_dim}_model_{epoch_counter}.pth'))
                print(f"Model saved at {self.log_dir}")

            valid_n_iter += 1
            
            # warmup for the first few epochs
            if epoch_counter >= self.args.warm_up and self.args.use_scheduler:
                scheduler.step()

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in tqdm(valid_loader, desc="Validation"):
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss_ct, loss_property, loss_fingerprint, loss_atom, loss_bond, loss_pos = self._step(model, xis, xjs, counter)
                loss_ct = torch.nan_to_num(loss_ct)
                loss_property = torch.nan_to_num(loss_property)
                loss_fingerprint = torch.nan_to_num(loss_fingerprint)
                loss_atom = torch.nan_to_num(loss_atom)
                loss_bond = torch.nan_to_num(loss_bond)
                loss_pos = torch.nan_to_num(loss_pos)
                loss = self.alpha*loss_ct + loss_property + loss_fingerprint + self.beta*(loss_atom + loss_bond + loss_pos)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--no_cuda', default=False, action='store_true', help='Use CPU')
    parser.add_argument('--base_dim', default=32, help='Training method')
    parser.add_argument('--gflayer', default='CFC', choices= ['GAT', 'GIN', 'GCN','GTN', 'CFC', "SGCN"], help='Layer method')
    parser.add_argument('--use_scheduler', default=True, type=bool, help='use scheduler')
    parser.add_argument('--warm_up', default=0, type=int, help='Warm up epochs')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--dataset', default='pretrained', help='Dataset name')
    # parser.add_argument('--model_type', default='2d', choices=['2d', '3d'], help='Model type: 2d or 3d')
    parser.add_argument('--mask_ratio', default=0.15, type=float, help='Masking ratio for pretraining')
    parser.add_argument("--data_path", default='data/dataset', type=str, help='Path to data')
    parser.add_argument('--test_set_ratio', default=0.1, type=float, help='Ratio of test set in the dataset')
    parser.add_argument('--save_path', default='ckpt', type=str, help='Path to save the model checkpoints')

    args = parser.parse_args()
    print(args)
    

    pretrain = Pretrain(root=args.data_path, dataset=args.dataset, args=args)
    pretrain.train()


if __name__ == "__main__":
    main()
