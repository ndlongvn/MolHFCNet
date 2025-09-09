import os
import numpy as np
import torch
from munch import Munch
import pandas as pd 
import os
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_auc_score, cohen_kappa_score, balanced_accuracy_score, mean_squared_error, mean_absolute_error, r2_score

def get_perform_regression(labels, probs, task=None):
    actual_arr = np.array(labels)
    predicted_arr = np.array(probs)
    # rmse = np.sqrt(np.mean((actual_arr - predicted_arr)**2))
    rmse = np.sqrt(mean_squared_error(actual_arr, predicted_arr))
    r2= r2_score(actual_arr, predicted_arr)
    mae = mean_absolute_error(actual_arr, predicted_arr)
    # print("RMSE:", rmse)
    return rmse, mae, r2

def get_perform_binary(labels, probs, task=None):
    trn_roc = roc_auc_score(labels, probs)
    trn_prc = metrics.auc(precision_recall_curve(labels, probs)[1],
                        precision_recall_curve(labels, probs)[0])
    predicted_labels = []
    for prob in probs: 
        predicted_labels.append(np.round(prob))

    trn_acc = accuracy_score(labels, predicted_labels)
    trn_ba  =balanced_accuracy_score(labels, predicted_labels)
    trn_mcc =matthews_corrcoef(labels, predicted_labels)
    trn_ck  = cohen_kappa_score(labels, predicted_labels)
    
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision   = tp / (tp + fp)
    f1 = 2*precision*sensitivity / (precision + sensitivity)
   
    perform = [trn_roc, trn_prc, trn_acc, trn_ba, trn_mcc, trn_ck, sensitivity, specificity, precision, f1]
    # print(f"AUC= {trn_roc} , PR_AUC={trn_prc}")
    return perform


def get_perform_multitask(y_label_list, y_pred_list, tasks):
    trn_roc =  np.array([roc_auc_score(y_label_list[i], y_pred_list[i]) for i in range(len(tasks))])
    trn_prc =  np.array([metrics.auc(precision_recall_curve(y_label_list[i], y_pred_list[i])[1],
                        precision_recall_curve(y_label_list[i], y_pred_list[i])[0]) for i in range(len(tasks))])
   
    predicted_labels = [[] for i in range(len(tasks))]
    for i in range(len(tasks)):
        for prob in y_pred_list[i]: 
            predicted_labels[i].append(np.round(prob))

    trn_acc = np.array([accuracy_score(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_ba  = np.array([balanced_accuracy_score(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_mcc = np.array([matthews_corrcoef(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_ck  =  np.array([cohen_kappa_score(y_label_list[i], predicted_labels[i]) for i in range(len(tasks))])
    trn_sensitivity, trn_specificity, trn_precision, trn_f1 = [], [], [], []
    for i in range(len(tasks)):
        tn, fp, fn, tp = confusion_matrix(y_label_list[i], predicted_labels[i]).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision   = tp / (tp + fp)
        f1 = 2*precision*sensitivity / (precision + sensitivity)
        trn_sensitivity.append(sensitivity)
        trn_specificity.append(specificity)
        trn_precision.append(precision)
        trn_f1.append(f1)

    trn_sensitivity, trn_specificity, trn_precision, trn_f1 = np.array(trn_sensitivity), np.array(trn_specificity), np.array(trn_precision), np.array(trn_f1)
    perform = [trn_roc, trn_prc, trn_acc, trn_ba, trn_mcc, trn_ck, trn_sensitivity, trn_specificity, trn_precision, trn_f1]
    perform = [np.mean(perform[i]) for i in range(len(perform))]
      
    return perform

def get_perform_multi_regression(labels_list, preds_list, tasks, cal_all=False):
    # Calculate metrics for each task
    rmse_list = []
    r2_list = []
    mae_list = []
    for task_id in range(len(tasks)):
        actual_arr = np.array(labels_list[task_id])
        predicted_arr = np.array(preds_list[task_id])
        if len(actual_arr)==0 or len(predicted_arr)==0:
            continue
        rmse = np.sqrt(mean_squared_error(actual_arr, predicted_arr))
        r2 = r2_score(actual_arr, predicted_arr)
        mae = mean_absolute_error(actual_arr, predicted_arr)
        rmse_list.append(rmse)
        r2_list.append(r2)
        mae_list.append(mae)
    
    # Calculate overall metrics
    all_labels = np.concatenate(labels_list)
    all_preds = np.concatenate(preds_list)
    overall_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    overall_r2 = r2_score(all_labels, all_preds)
    overall_mae = mean_absolute_error(all_labels, all_preds)
    
    if cal_all:
        return overall_rmse, overall_mae, overall_r2
    
    return rmse_list, mae_list, r2_list

def parse_args(args, **kwargs):
    args = Munch({"epoch": 0}, **args)
    kwargs = Munch({"no_cuda": False, "debug": False}, **kwargs)
    args.device = "cuda" if torch.cuda.is_available() and not kwargs.no_cuda else "cpu"
    
    if "decoder_args" not in args or args.decoder_args is None:
        args.decoder_args = {}
    if "model_path" in args:
        args.out_path = os.path.join(args.model_path, args.name)
        os.makedirs(args.out_path, exist_ok=True)
    return args


import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

