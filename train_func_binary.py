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
    for idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        
        data       = batch.to(device)
        labels     = batch.y.to(device).float()
        labels = (labels+1)/2
        if len(labels) < args.batch_size * 0.5:
            continue
        outputs = model(data)[0]
        #------------------- 
        optimizer.zero_grad()
        #------------------- 

        avg_loss = criterion(outputs.view_as(labels), labels) 

        train_loss += avg_loss.item()

        avg_loss.backward()
        optimizer.step() 

    if idx >= args.warm_up and args.use_scheduler:
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
        for _, batch in enumerate(tqdm(val_loader, desc="Validation")):
            data       = batch.to(device)
            labels     = batch.y.to(device).float()
            outputs = model(data)[0].view_as(labels)
            avg_loss = criterion(outputs, labels)  
            validation_loss += avg_loss.item()
        
    print('====> Epoch: {} Average Validation Loss: {:.4f}'.format(epoch, validation_loss / len(val_loader)))
    validation_loss = (validation_loss / len(val_loader.dataset) )

    return validation_loss
    
# ##########################################################################################
# Test Function
def test(current_iter, model, criterion, test_loader, tasks=None, use_test= True):
    model.eval()
    test_loss = 0
    pred = []
    labels_ = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):

            data       = batch.to(device)
            labels     = batch.y.to(device).float()
            labels = (labels+1)/2
           
            outputs = model(data)[0].view_as(labels)

            labels_.append(labels.detach().cpu().numpy())
            pred.append(outputs.detach().cpu().numpy())
            # avg_loss = criterion(outputs, labels) 
            # test_loss += avg_loss.item()
    pred = np.concatenate(pred, axis=0).tolist()
    labels_ = np.concatenate(labels_, axis=0).tolist()

    return test_loss, pred, labels_

