import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def compute_f1(predictions: list, truth: list, f: bool):
    '''
    compute evaluation metrics
    :param predictions: list of model predictions
    :param truth: list of ground truth
    :param f: compute F1 for each target or other measures for all targets
    :return: evaluation metrics
    '''
    rounded_preds = torch.nn.functional.softmax(predictions)
    values, indices = torch.max(rounded_preds, 1)

    y_pred = np.array(indices.cpu().numpy())
    y_true = np.array(truth.cpu().numpy())
    result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])

    if f:
        f1_average = (result[2][0] + result[2][2]) / 2  # average F1 score of Favor and Against
        return f1_average
    else:
        correct = (indices == truth).float()
        accuracy = correct.sum() / len(correct)
        return accuracy, result[0], result[1]  # accuracy, precision, recall
