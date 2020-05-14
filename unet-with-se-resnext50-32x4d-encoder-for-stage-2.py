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

import glob

# + _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19"
import os
import random
import subprocess
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm

import albumentations as A
import cv2
import segmentation_models_pytorch as smp
import torch
from kaggle_runner.data_providers import provider
from kaggle_runner.datasets.coders import run_length_encode

# from kaggle_runner.datasets.siim_dataset import SIIMDataset
from kaggle_runner.datasets.mock_dataset import MockDataset
from kaggle_runner.plots import plot
from kaggle_runner.post_processers import post_process
from kaggle_runner.runners.trainer import Trainer
from torch.utils.data import DataLoader  # TODO optimize this

# +
# from albumentations.pytorch import ToTensor

warnings.filterwarnings("ignore")
# -

A.MultiplicativeNoise()

# ## Utility functions

# ## Dataloader

print(os.listdir("../input/"))

sample_submission_path = "../input/siimmy/stage_2_sample_submission.csv"
train_rle_path = "../input/mysiim/train-rle.csv"
data_folder = "../input/siimpng/siimpng/train_png"
test_data_folder = "../input/siim_stage2_png"

ab = glob.glob("../input/siimpng/siimpng/train_png/*.png")
len(ab)

a = pd.read_csv(train_rle_path)
len(a)

# ### Dataloader sanity check, not used in trainer

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


# ## Some more utility functions
#
# Here are some utility functions for calculating IoU and Dice scores

# ![](http://)## FPN  with inceptionresnetv2 model
# Let's take a look at the model

model = smp.FPN("inceptionresnetv2", encoder_weights="imagenet")

model  # a *deeper* look

# **Radams**


# -

# ## Model Training and validation

model_trainer = Trainer(model, data_folder=data_folder, df_path=train_rle_path)
model_trainer.start()

# +
# PLOT TRAINING
losses = model_trainer.losses
dice_scores = model_trainer.dice_scores  # overall dice
iou_scores = model_trainer.iou_scores

plot(losses, "BCE loss")
plot(dice_scores, "Dice score")
plot(iou_scores, "IoU score")


# -

# ## Test prediction

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
    MockDataset(test_data_folder, df, size, mean, std),
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

# TODO put to kaggleKernel class, the predict part
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
