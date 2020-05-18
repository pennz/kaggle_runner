import tensorflow as tf
from tensorflow.python.ops import math_ops

import torch
import torch.nn as nn
from torch.nn import functional as F

from .defaults import (ALPHA, FOCAL_LOSS_BETA_NEG_POS, FOCAL_LOSS_GAMMA,
                       FOCAL_LOSS_GAMMA_NEG_POS)


def toxic_custom_mimic_loss(predictions, labels, subgroups, power=5.0,
                            score_function=F.binary_cross_entropy_with_logits):
    """
    Just reference this

    https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/103280
    """

    subgroup_positive_mask = subgroups & (labels.unsqueeze(-1) >= 0.5)
    subgroup_negative_mask = subgroups & ~(labels.unsqueeze(-1) >= 0.5)
    background_positive_mask = ~subgroups & (labels.unsqueeze(-1) >= 0.5)
    background_negative_mask = ~subgroups & ~(labels.unsqueeze(-1) >= 0.5)

    bpsn_mask = (background_positive_mask | subgroup_negative_mask).float()
    bnsp_mask = (background_negative_mask | subgroup_positive_mask).float()
    subgroups = subgroups.float()

    bce = score_function(predictions,labels, reduction="none")

    sb = (bce.unsqueeze(-1) * subgroups).sum(0).div(subgroups.sum(0).clamp(1.)).pow(power).mea().pow(1/power)
    bpsn = (bce.unsqueeze(-1) * bpsn_mask).sum(0).div(bpsn_mask.sum(0).clamp(1.)).pow(power).mean().pow(1/power)
    bnsp = (bce.unsqueeze(-1) * bnsp_mask).sum(0).div(bnsp_mask.sum(0).clamp(1.)).pow(power).mean().pow(1/power)
    loss = (bce.mean() + sb + bpsn + bnsp) /4

    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not target.size() == input.size():
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        def dice_loss(input, target):
            input = torch.sigmoid(input)
            smooth = 1.0
            iflat = input.view(-1)
            tflat = target.view(-1)
            intersection = (iflat * tflat).sum()

            return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

        loss = self.alpha * self.focal(input, target) - torch.log(
            dice_loss(input, target)
        )

        return loss.mean()


def binary_crossentropy_with_focal_seasoned(
    y_true,
    logit_pred,
    beta=FOCAL_LOSS_BETA_NEG_POS,
    gamma=FOCAL_LOSS_GAMMA_NEG_POS,
    alpha=ALPHA,
    custom_weights_in_y_true=True,
):
    """
    :param alpha:weight for positive classes **loss**. default to 1- true positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or hyperparameter.
    :param custom_weights_in_y_true:
    :return:
    """
    balanced = gamma * logit_pred + beta
    y_pred = math_ops.sigmoid(balanced)
    # only use gamma in this layer, easier to split out factor

    return binary_crossentropy_with_focal(
        y_true,
        y_pred,
        gamma=0,
        alpha=alpha,
        custom_weights_in_y_true=custom_weights_in_y_true,
    )


def binary_crossentropy_with_focal(
    y_true, y_pred, gamma=FOCAL_LOSS_GAMMA, alpha=ALPHA, custom_weights_in_y_true=True
):
    """
    https://arxiv.org/pdf/1708.02002.pdf

    $$ FL(p_t) = -(1-p_t)^{\gamma}log(p_t) $$
    $$ p_t=p\: if\: y=1$$
    $$ p_t=1-p\: otherwise$$

    :param y_true:
    :param y_pred:
    :param gamma: make easier ones weights down
    :param alpha: weight for positive classes. default to 1- true positive cnts / all cnts, alpha range [0,1] for class 1 and 1-alpha
        for calss -1.   In practiceαmay be set by inverse class freqency or hyperparameter.
    :return:
    """
    # assert 0 <= alpha <= 1 and gamma >= 0
    # hyper parameters, just use the one for binary?
    # alpha = 1. # maybe smaller one can help, as multi-class will make the error larger
    # gamma = 1.5 # for our problem, try different gamma

    # for binary_crossentropy, the implementation is in  tensorflow/tensorflow/python/keras/backend.py
    #       bce = target * alpha* (1-output+epsilon())**gamma * math_ops.log(output + epsilon())
    #       bce += (1 - target) *(1-alpha)* (output+epsilon())**gamma * math_ops.log(1 - output + epsilon())
    # return -bce # binary cross entropy
    eps = tf.keras.backend.epsilon()

    if custom_weights_in_y_true:
        custom_weights = y_true[:, 1:2]
        y_true = y_true[:, :1]

    if 1.0 - eps <= gamma <= 1.0 + eps:
        bce = alpha * math_ops.multiply(
            1.0 - y_pred, math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        )
        bce += (1 - alpha) * math_ops.multiply(
            y_pred, math_ops.multiply(
                (1.0 - y_true), math_ops.log(1.0 - y_pred + eps))
        )
    elif 0.0 - eps <= gamma <= 0.0 + eps:
        bce = alpha * math_ops.multiply(y_true, math_ops.log(y_pred + eps))
        bce += (1 - alpha) * math_ops.multiply(
            (1.0 - y_true), math_ops.log(1.0 - y_pred + eps)
        )
    else:
        gamma_tensor = tf.broadcast_to(
            tf.constant(gamma), tf.shape(input=y_pred))
        bce = alpha * math_ops.multiply(
            math_ops.pow(1.0 - y_pred, gamma_tensor),
            math_ops.multiply(y_true, math_ops.log(y_pred + eps)),
        )
        bce += (1 - alpha) * math_ops.multiply(
            math_ops.pow(y_pred, gamma_tensor),
            math_ops.multiply(
                (1.0 - y_true), math_ops.log(1.0 - y_pred + eps)),
        )

    if custom_weights_in_y_true:
        return math_ops.multiply(-bce, custom_weights)
    else:
        return -bce
