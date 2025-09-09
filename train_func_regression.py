# Import Python libraries
import torch
import numpy as np
import time
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##########################################################################################                 
###                  DEFINE TRAINING, VALIDATION, AND TEST FUNCTION                    ###           
##########################################################################################
# Training Function
def train_funct(args, epoch, model, optimizer, criterion, train_loader, scheduler, tasks=None):
    model.train()
    train_loss = 0
    start_time = time.time()
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        data       = batch.to(device)
        labels     = batch.y.to(device).float() # Chứa label của 12 tasks

        if len(labels) < args.batch_size * 0.5:
            continue
        outputs = model(data)[0]

        #------------------- 
        optimizer.zero_grad()
        #------------------- 
        avg_loss = criterion(outputs.view_as(labels), labels)  

        avg_loss.backward()
        train_loss += avg_loss.item() #(loss.item is the average loss of training batch)
        optimizer.step() 
        if batch_idx >= args.warm_up and args.use_scheduler:
            scheduler.step()

    #------------------- 
    print('====> Epoch: {}, training time {},  Average Train Loss: {:.4f}'.format(epoch, time.time() - start_time, train_loss / len(train_loader)))
    train_loss = (train_loss / len(train_loader.dataset) )

    return train_loss

##########################################################################################
# Validation Function
def validate(epoch, model, criterion, val_loader, tasks=None):
    model.eval()
    validation_loss = 0
   
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            data       = batch.to(device)
            labels     = batch.y.to(device).float() # Chứa label của 12 task
            outputs = model(data)[0].view_as(labels)
            avg_loss = criterion(outputs, labels)  
            validation_loss += avg_loss.item() #(loss.item is the average loss of training batch)
        
    print('====> Epoch: {} Average Validation Loss: {:.4f}'.format(epoch, validation_loss / len(val_loader)))
    validation_loss = (validation_loss / len(val_loader.dataset) )
    # perform = get_performace(y_label_list, y_pred_list, tasks)

    return validation_loss

def validate_covid(epoch, model, criterion, val_loader, tasks=None, test=True):
    model.eval()
    validation_loss = 0
   
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            data       = batch.to(device)
            labels     = batch.y.to(device).float() # Chứa label của 12 task
            outputs = model(data).view_as(labels)
            avg_loss = criterion(outputs, labels)  
            validation_loss += avg_loss.item() #(loss.item is the average loss of training batch)
        
    print('====> Epoch: {} Average Validation Loss: {:.4f}'.format(epoch, validation_loss / len(val_loader)))
    validation_loss = (validation_loss / len(val_loader.dataset) )
    # perform = get_performace(y_label_list, y_pred_list, tasks)

    return validation_loss


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# ##########################################################################################
# Test Function
def test(current_iter, model, criterion, test_loader, tasks=None, use_test=True):
    model.eval()
    pred = []
    labels_ = []
    smiles = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):

            data       = batch.to(device)
            labels     = batch.y.to(device).float()
            
            outputs = model(data)[0].view_as(labels)
            # smiles.append(batch.smiles)
            labels_.append(labels.detach().cpu().numpy())
            pred.append(outputs.detach().cpu().numpy())

    pred = np.concatenate(pred, axis=0).tolist()
    labels_ = np.concatenate(labels_, axis=0).tolist()
    # smiles = np.concatenate(smiles, axis=0)
    test_rmse = rmse(np.array(pred), np.array(labels_))
    
    # print("Performance of model at epoch {} on test dataset:  {}".format(current_iter, test_rmse))
    # print('====> Epoch: {} Average Test Loss: {:.4f}'.format(current_iter, test_rmse))
    return test_rmse, pred, labels_,smiles


