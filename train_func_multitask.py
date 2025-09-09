# Import Python libraries
import torch
import numpy as np
import time
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_loss_task(criterion, num_classes_tasks, probs, labels, device):
    total_loss = 0
    for t_id in range(num_classes_tasks):
        y_pred = probs[t_id] # output of each task
        y_label = labels[:, t_id:t_id+1].squeeze() # label of task
   
        # validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0]
        validId = y_label**2 > 0
        # validId= range(len(y_label))    
        if len(validId) == 0:
            # raise ValueError("No valid data for get loss task {}".format(t_id))
            continue
        
        if y_label.dim() == 0:
            y_label = y_label.unsqueeze(0)

        y_pred = y_pred[torch.tensor(validId).to(device)]
        y_label = y_label[torch.tensor(validId).to(device)]

        loss = criterion(y_pred.view(-1).squeeze(), ((y_label+1)/2).squeeze())
        total_loss += loss
    total_loss = total_loss/num_classes_tasks
    return total_loss



def get_prob_task(num_classes_tasks, probs, labels, device):
    prob_list = []
    label_list = []
    for t_id in range(num_classes_tasks):
        y_pred = probs[t_id] # output of each task
        y_label = labels[:, t_id:t_id+1].squeeze() # label of task
        # validId = np.where((y_label.cpu().numpy() == 0) | (y_label.cpu().numpy() == 1))[0] 
        validId = y_label**2 > 0

        if len(validId) == 0:
            prob_list.append([])
            label_list.append([])
            continue

        if y_label.dim() == 0:
            
            y_label = y_label.unsqueeze(0)

        y_pred = y_pred[torch.tensor(validId).to(device)]
        y_label = y_label[torch.tensor(validId).to(device)]
        
        prob_list.append(y_pred.detach().cpu().view_as(y_label).numpy().tolist())
        label_list.append(((y_label+1)/2).detach().cpu().numpy().tolist())
    return prob_list, label_list


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
        labels     = batch.y.to(device).float()
        
        
        if len(labels) < args.batch_size * 0.5:
            continue
        outputs = model(data)
        #------------------- 
        optimizer.zero_grad()
        #------------------- 

        avg_loss = get_loss_task(criterion, len(tasks), outputs, labels, device) 

        avg_loss.backward()
        train_loss += avg_loss.item()
        optimizer.step() 
        if batch_idx >= args.warm_up and args.use_scheduler:
            scheduler.step()

    #------------------- 
    print('====> Epoch: {}, training time {},  Average Train Loss: {:.4f}'.format(epoch, time.time() - start_time, train_loss / len(train_loader)))
    train_loss = (train_loss / len(train_loader.dataset) )

    return train_loss


##########################################################################################
# Validation Function
def validate(epoch, model, criterion, val_loader, tasks):
    model.eval()
    validation_loss = 0
   
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
            data       = batch.to(device)
            labels     = batch.y.to(device).float()
            outputs = model(data)

            avg_loss = get_loss_task(criterion, len(tasks), outputs, labels, device) 
  
            validation_loss += avg_loss.item()
        
    print('====> Epoch: {} Average Validation Loss: {:.4f}'.format(epoch, validation_loss / len(val_loader)))
    validation_loss = (validation_loss / len(val_loader.dataset) )

    return validation_loss
    
# ##########################################################################################
# Test Function
def test(current_iter, model, criterion, test_loader, tasks=None, use_test = True):
    model.eval()
    test_loss = 0
    preds = [[] for i in range(len(tasks))]
    labels_ = [[] for i in range(len(tasks))]

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):

            data       = batch.to(device)
            labels     = batch.y.to(device).float()

            outputs = model(data)
            pred_list, label_list = get_prob_task(len(tasks), outputs, labels, device)

            # avg_loss =  get_loss_task(criterion, len(tasks), outputs, labels, device)
            # test_loss += avg_loss.item()
            for i in range(len(tasks)):
                if len(label_list[i]) != 0:
                    preds[i] = preds[i] + pred_list[i]
                    labels_[i] = labels_[i] + label_list[i]

    if use_test:
        assert len([labels_[i] for i in range(len(tasks) ) if len(labels_[i]) != 0]) == len(tasks), "Some tasks have no valid data"
    # test_loss = (test_loss / len(test_loader.dataset) )

    # print("Performance of model at epoch {} on test dataset:  {}".format(current_iter, test_loss))
    # print('====> Epoch: {} Average Test Loss: {:.4f}'.format(current_iter, test_loss))
    return test_loss, preds, labels_

