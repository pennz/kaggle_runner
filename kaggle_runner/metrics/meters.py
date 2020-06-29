import numpy as np
import torch

import torch.nn as nn
import sklearn
from kaggle_runner.datasets.coders import compute_iou_batch
from kaggle_runner.metrics.metrics import metric
from kaggle_runner.predictors import predict
from .metrics import matthews_correlation


class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.append(dice)
        self.dice_pos_scores.append(dice_pos)
        self.dice_neg_scores.append(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)
        dice_neg = np.mean(self.dice_neg_scores)
        dice_pos = np.mean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)

        return dices, iou

# # + colab={} colab_type="code" id="swiDMY2l3bHb"
class RocAucMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([])
        self.y_true_float = np.array([], dtype=np.float)
        self.y_pred = np.array([])
        self.score = 0
        self.mc_score = 0
        self.aux_part = 0

    def update(self, y_true, y_pred, aux_part=0):
        #y_true_ = y_true
        y_true = y_true[:,:2].cpu().numpy().argmax(axis=1)
        y_true_float = y_true.astype(np.float)
        y_pred = nn.functional.softmax(y_pred[:,:2], dim=1).data.cpu().numpy()[:,1]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_true_float = np.hstack((self.y_true_float, y_true_float))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        try:
            self.score = sklearn.metrics.roc_auc_score(self.y_true, self.y_pred, labels=np.array([0, 1]))
        except Exception:
            self.score = 0
        self.mc_score = matthews_correlation(self.y_true_float, self.y_pred)
        self.aux_part = aux_part

    @property
    def avg(self):
        return self.score
    @property
    def mc_avg(self):
        return self.mc_score

# # + colab={} colab_type="code" id="sj4NmKFm3bHj"
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
