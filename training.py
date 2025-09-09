# Import Python libraries
import os
import argparse
import yaml
import torch
import torch.nn as nn
from munch import Munch
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from data.molecular_dataset import MoleculeDataset
from data.splitters import random_split, random_scaffold_split
import random 
from utils import parse_args
import numpy as np 
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings("ignore")



class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
    
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(args):
    seed_everything(0)
    args.tasks = None
    # Training for binary classification dataset
    if args.name=='BACE' or args.name=='BBBP' or args.name=='HIV' or args.name=='jak1' or args.name=='jak2' or args.name=='mapk14':
        from train_func_binary import train_funct, test
        from utils import get_perform_binary as get_perform
        criterion = nn.BCELoss()
        args.regression = False
        args.output_dim = 1
        
    # Training for regression dataset
    elif args.name=='ESOL' or args.name=='FreeSolv' or args.name=='Lipophilicity' or args.name=='p110Alpha':
        from train_func_regression import train_funct, test
        from utils import get_perform_regression as get_perform
        criterion = RMSELoss()
        args.regression = True
        args.output_dim = 1

    elif args.name=='ClinTox' or args.name=='SIDER' or args.name=='Tox21':
        from train_func_multitask import train_funct, test
        from utils import get_perform_multitask as get_perform
        
        criterion = nn.BCELoss()
        args.regression = False
        if args.name == 'ClinTox':
            args.tasks = ['CT_TOX', 'FDA_APPROVED']
        if args.name == 'SIDER':
            args.tasks = ['Blood and lymphatic system disorders', 'Cardiac disorders', 'Congenital, familial and genetic disorders', 'Ear and labyrinth disorders', 'Endocrine disorders', 'Eye disorders', 'Gastrointestinal disorders', 'General disorders and administration site conditions', 'Hepatobiliary disorders', 'Immune system disorders', 'Infections and infestations', 'Injury, poisoning and procedural complications', 'Investigations', 'Metabolism and nutrition disorders', 'Musculoskeletal and connective tissue disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 'Nervous system disorders', 'Pregnancy, puerperium and perinatal conditions', 'Product issues', 'Psychiatric disorders', 'Renal and urinary disorders', 'Reproductive system and breast disorders', 'Respiratory, thoracic and mediastinal disorders', 'Skin and subcutaneous tissue disorders', 'Social circumstances', 'Surgical and medical procedures', 'Vascular disorders']
        if args.name == 'Tox_21':
            args.tasks = ['NR-AhR', 'NR-AR-LBD', 'NR-AR', 'NR-Aromatase', 'NR-ER-LBD', 'NR-ER', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        
        args.output_dim = len(args.tasks)
    
    else:
        raise ValueError("Customize your dataset and rewrite it using the sample code above.")
    
    #Bunch of classification tasks
    args.dataset = args.name.lower()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
   
    #set up dataset
    dataset = MoleculeDataset("data/dataset/" + args.dataset, dataset=args.dataset)

    print(dataset)
    
    if args.type_split == "scaffolds":
        smiles_list = pd.read_csv('data/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        # train_dataset, valid_dataset, test_dataset, _ = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)

        # dataset, smiles_list, random_seed= 8, ratio_test= 0.1, ration_valid= 0.1
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, ratio_test= 0.1, ration_valid= 0.1, random_seed = args.seed)
        print("scaffold")
    elif args.type_split == "random":
        smiles_list = pd.read_csv('data/dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_split(dataset, ratio_test= 0.1, ration_valid= 0.1, random_seed = args.seed)
        print("random")
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    if args.dataset == 'freesolv':
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)


    # Model
    if args.gflayer in ['GAT', 'GIN', 'GCN', 'GTN']:
            from models.graph_2d import MolHFC
    elif args.gflayer in ['CFC', 'SGCN']:
            from models.graph_3d import MolHFC

    model = MolHFC(node_dim = train_dataset[0].x.shape[1], edge_dim=train_dataset[0].edge_attr.shape[1],
                num_classes_tasks = args.output_dim, base_dim=args.base_dim, regression=args.regression,
                gflayer=args.gflayer)  # geolayer=args.geolayer,
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    model = model.to(args.device)

    if args.ckpt_path != '':
        try:
            ckpt= torch.load(args.ckpt_path)
            ckpt = {k:v for k, v in ckpt.items() if "backbone." in k}
            ckpt = {k.replace("backbone.", ""):v for k, v in ckpt.items()}
            model.backbone.load_state_dict(ckpt, strict=True)
        except:
            new_state_dict = {}
            ckpt= torch.load(args.ckpt_path)
            ckpt = {k:v for k, v in ckpt.items() if "backbone." in k}
            ckpt = {k.replace("backbone.", ""):v for k, v in ckpt.items()}
            for key, value in ckpt.items():
                # Replace "lin_key" with "gnn.lin_key", "lin_query" with "gnn.lin_query", etc.
                new_key = key.replace('.lin_key', '.gnn.lin_key') \
                             .replace('.lin_query', '.gnn.lin_query') \
                             .replace('.lin_value', '.gnn.lin_value') \
                             .replace('.lin_edge', '.gnn.lin_edge') \
                             .replace('.lin_skip', '.gnn.lin_skip')
                new_state_dict[new_key] = value

            # Load the modified state_dict into the model
            model.backbone.load_state_dict(new_state_dict, strict=True)

    args.early_stopping = args.epochs // 5
    
    # Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs-args.warm_up, 
            eta_min=0, last_epoch=-1)
    
    print('size of training set: ', len(train_dataset))
    print('size of validation set: ', len(valid_dataset))
    print('size of test set: ', len(test_dataset))


    print("Training Model")
    best_performance = 0
    auc_check = 0
    reg_check = 100
    early_stop = 0

    # training 1
    print(f"Training model with {args.epochs} epochs")

    for epoch in range(args.epochs):
        train_results         = train_funct(args, epoch, model, optimizer, criterion, train_loader, scheduler, args.tasks )
        # validation_results    = validate(epoch, model, criterion, val_loader, args.tasks)

        val_outputs = test(epoch, model, criterion, val_loader, args.tasks, use_test=False)

        val_perform =  get_perform(val_outputs[2], val_outputs[1], args.tasks)
        
        if not args.regression:
            [val_roc, val_prc, val_acc, val_ba, val_mcc, val_ck, sensitivity, specificity, precision, f1] = val_perform
            if auc_check <= val_roc:
                auc_check= val_roc
                
                test_outputs = test(epoch, model, criterion, test_loader, args.tasks)

                perform =  get_perform(test_outputs[2], test_outputs[1], args.tasks)

                best_performance = perform

                early_stop = 0

                print(f"========> Epoch {epoch} Val - AUC= {val_roc} , PR_AUC={val_prc}")
                print(f"========> Epoch {epoch} Test - AUC= {perform[0]} , PR_AUC={perform[1]}")


            else:
                early_stop+=1
                if early_stop>args.early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            [rmse, mae, r2] = val_perform

            if reg_check>=rmse:
                reg_check= rmse
                
                test_outputs = test(epoch, model, criterion, test_loader, args.tasks)

                perform =  get_perform(test_outputs[2], test_outputs[1], args.tasks)

                best_performance = perform

                early_stop=0

                print(f"========> Epoch {epoch} Val - RMSE= {rmse} , MAE={mae}")
                print(f"========> Epoch {epoch} Test - RMSE= {perform[0]} , MAE={perform[1]}")

            else:
                early_stop+=1
                if early_stop>args.early_stopping:
                    print(f"Early stopping at epoch {epoch}")
                    break

        
    print(f"Best performance: {best_performance[0]} for task {args.name}, seed {args.seed}, split {args.type_split}")
        
    return (best_performance[0], best_performance[1], best_performance[2]) if args.regression \
                else (best_performance[0], best_performance[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    
    parser.add_argument('--no_cuda', default=False, action='store_true', help='Use CPU')
    parser.add_argument('--base_dim', default=32, help='Training method')
    parser.add_argument('--gflayer', default='GIN', choices= ['GAT', 'GIN', 'GCN','GTN', 'CFC', "SGCN"], help='Layer method')
    parser.add_argument('--use_scheduler', default=True, type=bool, help='use scheduler')
    parser.add_argument('--warm_up', default=0, type=int, help='Warm up epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--name', default='BACE', choices= ["BACE", "BBBP", "ClinTox", "ESOL", "FreeSolv", "Lipophilicity", "SIDER", "Tox21" , "HIV"], help='Dataset name')
    parser.add_argument('--model_type', default='2d', choices=['2d', '3d'], help='Model type: 2d or 3d')
    parser.add_argument("--data_path", default='data/dataset', type=str, help='Path to data')
    parser.add_argument('--ckpt_path', default='', type=str, help='Path to save the model checkpoints')
    parser.add_argument('--out_path', default='result', type=str, help='Path to save the results')
    parser.add_argument('--type_split', default='random', choices=['random', 'scaffolds'], help='Type of split for the dataset')
    parser.add_argument('--seed', default=0, type=int, help='Splitter seed')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers for data loading')
  

    args = parser.parse_args()

    result= train(args) # trn_roc, trn_prc
    print(f"Result: {result}")

    
        
        
