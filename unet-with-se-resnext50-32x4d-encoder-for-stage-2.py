# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# **this kernel contains my stage-2 submissions**
#

# **Hello kagglers**
# this is my first kaggle competition,as a beginner what i tried is as follows :
# 1.thanks @rishabhiitbhu for your public kernel,i used code from that kernel,which is this one : https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch
# 2.changed his architecture from unet with resnet34 to unet with se_resnext50_32x4d
#
# in stage-1 i also tried a lot more different architectures in this kernel like linknet with resnet101,unet with resnet101,linknet with vgg11 etc etc,but none of them performed well than this unet with vgg11  in stage-1 public leaderboard,of course there were few architectures i tried like unet with senet with radams and i saw the graph is better than it was before,but as those encoders are large,i ran out of memory in kaggle kernel,if i had a gpu,i am optimistic those models would get us medal hahaha,anyway what are the models from segmentation model pytorch github ripository you  tried for this competition? and what the outcome of each?if you can remember please share with me in the comment box,it will help me a lot for my upcoming competitions
#
#
# **most importantly**
# 1. @rishabhiitbhu is using equal number of pneumothorax and non pneumothorax samples for training,i have decided to increase the non-pneumothorax sample a bit to see if it does well in terms of overall prediction
# 2. i have used radams here
# 3. after 20th epoch which this model stopped training because : "RuntimeError: DataLoader worker (pid(s) 1094) exited unexpectedly" so i  reduced the size of num_works and  trained the network again
#

# ****This Kernel uses UNet architecture with se_resnext50_32x4d encoder, I've used [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library which has many inbuilt segmentation architectures. This kernel is inspired by [Yury](https://www.kaggle.com/deyury)'s discussion thread [here](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/99440#591985). I've used snippets from multiple other public kernels I've given due credits at the end of this notebook.
#
# What's down below?
#
# * UNet with imagenet pretrained se_resnext50_32x4d architecture
# * Training on 512x512 sized images/masks with Standard Augmentations
# * MixedLoss (weighted sum of Focal loss and dice loss)
# * Gradient Accumulution

# **ChangeLog**
# 1. version 1 contains my exact model for this competition's stage 2 submission
# 2. in version 3 i will try speckle noise or multiplicative noise (implemented in albumentation few days ago after i requested for the implementation),check here : https://github.com/albu/albumentations/issues/439
# 3. in version 4 i will try fpn instead of unet
# 4. version 7 - FPN with inceptionresnetv2

# !pip install -U git+https://github.com/albu/albumentations

import glob

# +
import math

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import os
import pdb
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader, Dataset, sampler
from tqdm import tqdm_notebook as tqdm

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
from albumentations import (
    Blur,
    Compose,
    ElasticTransform,
    GaussNoise,
    GridDistortion,
    HorizontalFlip,
    IAAEmboss,
    MultiplicativeNoise,
    Normalize,
    OneOf,
    OpticalDistortion,
    RandomBrightnessContrast,
    RandomGamma,
    RandomRotate90,
    Resize,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensor
from trainer import Trainer

warnings.filterwarnings("ignore")
# -


A.MultiplicativeNoise()

# !pip install git+https://github.com/qubvel/segmentation_models.pytorch > /dev/null 2>&1 # Install segmentations_models.pytorch, with no bash output.


# ## Utility functions

# +
def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(" ")])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start:end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    rle = " ".join([str(r) for r in rle])
    return rle


# -

# ## Dataloader

# +
class SIIMDataset(Dataset):
    def __init__(self, df, fnames, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, size, mean, std)
        self.gb = self.df.groupby("ImageId")
        self.fnames = fnames

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df[" EncodedPixels"].tolist()
        image_path = os.path.join(self.root, image_id + ".png")
        image = cv2.imread(image_path)
        mask = np.zeros([1024, 1024])
        if annotations[0] != "-1":
            for rle in annotations:
                mask += run_length_decode(rle)
        mask = (mask >= 1).astype("float32")  # for overlap cases
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        return image, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                GaussNoise(),
                A.MultiplicativeNoise(multiplier=1.5, p=1),
            ]
        )
    list_transforms.extend(
        [Resize(size, size), Normalize(mean=mean, std=std, p=1), ToTensor(), ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms


def provider(
    fold,
    total_folds,
    data_folder,
    df_path,
    phase,
    size,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=2,
):
    df_all = pd.read_csv(df_path)
    df = df_all.drop_duplicates("ImageId")
    df_with_mask = df[df[" EncodedPixels"] != "-1"]
    df_with_mask["has_mask"] = 1
    df_without_mask = df[df[" EncodedPixels"] == "-1"]
    df_without_mask["has_mask"] = 0
    df_without_mask_sampled = df_without_mask.sample(
        len(df_with_mask) + 1500, random_state=2019
    )  # random state is imp
    df = pd.concat([df_with_mask, df_without_mask_sampled])

    # NOTE: equal number of positive and negative cases are chosen.

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=43)
    train_idx, val_idx = list(kfold.split(df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%

    fnames = df["ImageId"].values

    image_dataset = SIIMDataset(
        df_all, fnames, data_folder, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader


# -

print(os.listdir("../input/"))

sample_submission_path = "../input/siimmy/stage_2_sample_submission.csv"
train_rle_path = "../input/mysiim/train-rle.csv"
data_folder = "../input/siimpng/siimpng/train_png"
test_data_folder = "../input/siim_stage2_png"

ab = glob.glob("../input/siimpng/siimpng/train_png/*.png")
len(ab)

a = pd.read_csv(train_rle_path)
len(a)

# ### Dataloader sanity check

dataloader = provider(
    fold=0,
    total_folds=5,
    data_folder=data_folder,
    df_path=train_rle_path,
    phase="train",
    size=512,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    batch_size=16,
    num_workers=2,
)

batch = next(iter(dataloader))  # get a batch from the dataloader
images, masks = batch

# plot some random images in the `batch`
idx = random.choice(range(16))
plt.imshow(images[idx][0], cmap="bone")
plt.imshow(masks[idx][0], alpha=0.2, cmap="Reds")
plt.show()
if len(np.unique(masks[idx][0])) == 1:  # only zeros
    print("Chosen image has no ground truth mask, rerun the cell")


# ## Losses
#
# This kernel uses a weighted sum of Focal Loss and Dice Loss, let's call it MixedLoss

# +
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
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
        loss = self.alpha * self.focal(input, target) - torch.log(
            dice_loss(input, target)
        )
        return loss.mean()


# -

# ## Some more utility functions
#
# Here are some utility functions for calculating IoU and Dice scores

# +
def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype("uint8")
    return preds


def metric(probability, truth, threshold=0.5, reduction="none"):
    """Calculates dice of positive and negative images seperately"""
    """probability and truth must be torch tensors"""
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
        dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
        dice = dice.mean().item()

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice, dice_neg, dice_pos, num_neg, num_pos


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
        dice, dice_neg, dice_pos, _, _ = metric(
            probs, targets, self.base_threshold)
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


def epoch_log(phase, epoch, epoch_loss, meter, start):
    """logging the metrics at the end of an epoch"""
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print(
        "Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f"
        % (epoch_loss, dice, dice_neg, dice_pos, iou)
    )
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    """computes iou for one ground truth mask and predicted mask"""
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    """computes mean iou for a batch of ground truth masks and predicted masks"""
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


# -

# ![](http://)## FPN  with inceptionresnetv2 model
# Let's take a look at the model

model = smp.FPN("inceptionresnetv2", encoder_weights="imagenet")

model  # a *deeper* look

# **Radams**


class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError(
                        "RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(
                        p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = self.buffer[int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = (
                            group["lr"]
                            * math.sqrt(
                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)
                            )
                            / (1 - beta1 ** state["step"])
                        )
                    else:
                        step_size = group["lr"] / (1 - beta1 ** state["step"])
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(-group["weight_decay"]
                                     * group["lr"], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss


# -

# ## Model Training and validation

model_trainer = Trainer(model)
model_trainer.start()

# +
# PLOT TRAINING
losses = model_trainer.losses
dice_scores = model_trainer.dice_scores  # overall dice
iou_scores = model_trainer.iou_scores


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores["train"])),
             scores["train"], label=f"train {name}")
    plt.plot(range(len(scores["train"])), scores["val"], label=f"val {name}")
    plt.title(f"{name} plot")
    plt.xlabel("Epoch")
    plt.ylabel(f"{name}")
    plt.legend()
    plt.show()


plot(losses, "BCE loss")
plot(dice_scores, "Dice score")
plot(iou_scores, "IoU score")


# -

# ## Test prediction

# +
class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [Normalize(mean=mean, std=std, p=1),
             Resize(size, size), ToTensor(), ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples


def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


# -

size = 512
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
num_workers = 8
batch_size = 16
best_threshold = 0.5
min_size = 3500
device = torch.device("cuda:0")
df = pd.read_csv(sample_submission_path)
testset = DataLoader(
    TestDataset(test_data_folder, df, size, mean, std),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
model = model_trainer.net  # get the model from model_trainer object
model.eval()
state = torch.load("./model.pth", map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])
encoded_pixels = []
for i, batch in enumerate(tqdm(testset)):
    preds = torch.sigmoid(model(batch.to(device)))
    preds = (
        preds.detach().cpu().numpy()[:, 0, :, :]
    )  # (batch_size, 1, size, size) -> (batch_size, size, size)
    for probability in preds:
        if probability.shape != (1024, 1024):
            probability = cv2.resize(
                probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR
            )
        predict, num_predict = post_process(
            probability, best_threshold, min_size)
        if num_predict == 0:
            encoded_pixels.append("-1")
        else:
            r = run_length_encode(predict)
            encoded_pixels.append(r)
df["EncodedPixels"] = encoded_pixels
df.to_csv("submission.csv", columns=["ImageId", "EncodedPixels"], index=False)

df.head()

#
#
# `segmentation_models_pytorch` has got many other segmentation models implemented, try them out :)
#
# I've learnt a lot from fellow kagglers, I've borrowed a lot of code from you guys, special shout-out to [@Abhishek](https://www.kaggle.com/abhishek), [@Yury](https://www.kaggle.com/deyury), [Heng](https://www.kaggle.com/hengck23), [Ekhtiar](https://www.kaggle.com/ekhtiar), [lafoss](https://www.kaggle.com/iafoss), [Siddhartha](https://www.kaggle.com/meaninglesslives) and many other kagglers :)
#
# Kaggle is <3
