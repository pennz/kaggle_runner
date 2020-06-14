import torch
from torch import nn
from fastai import *
from fastai import vision
from fastai.basic_data import *
from fastai.basic_train import Learner
from fastai.callbacks import CSVLogger
from fastai.core import *
from fastai.torch_core import *

from .kernel import KaggleKernel
from .. import logger


class FastAIKernel(KaggleKernel):
    """Can't instantiate abstract class FastAIKernel with abstract methods
    build_and_set_model, check_predict_details, peek_data, set_loss,
    set_metrics
    """

    def build_and_set_model(self):
        self.model = None

    def check_predict_details(self):
        assert False

    def peek_data(self):
        assert False

    def set_loss(self, loss_func):
        self.model_loss = loss_func

    def set_metrics(self, metrics):
        self.model_metrics = metrics

    def __init__(self, **kargs):
        super(FastAIKernel, self).__init__(logger=logger)
        self.developing = True
        self.learner = None

        for required in ['loss_func', 'metrics']:
            assert required in kargs

        for k,v in kargs.items():
            setattr(self, k, v)

    def setup_learner(self, data=None, model=None, opt_func=None, loss_func=None, metrics=None):
        data = self.data if hasattr(self, 'data') and self.data is not None else data
        model = self.model if hasattr(self, 'model') and self.model is not None else model
        opt_func = self.opt_func if hasattr(self, 'opt_func') and self.opt_func is not None else AdamW
        loss_func = self.model_loss if hasattr(self, 'model_loss') and self.model_loss is not None else loss_func
        metrics = self.model_metrics if hasattr(self, 'model_metrics') and self.model_metrics is not None else metrics

        return Learner(data, model, opt_func, loss_func, metrics, bn_wd=False)


# to get all files from a directory
import os

# to easier work with paths
from pathlib import Path

# to read and manipulate .csv-files
import pandas as pd
# -

INPUT = Path("../input/digit-recognizer")
os.listdir(INPUT)

# Let's look at 'train.csv' and 'test.csv':

train_df = pd.read_csv(INPUT/"train.csv")
train_df.head(3)

test_df = pd.read_csv(INPUT/"test.csv")
test_df.head(3)

# (nice to know: the input folder of Kaggle Competitions is always read-only, so if we want to add data or create folders, we have to do so outside of the input folder)

TRAIN = Path("../train")
TEST = Path("../test")

# Create training directory

for index in range(10):
    try:
        os.makedirs(TRAIN/str(index))
    except:
        pass

# Test whether creating the training directory was successful
sorted(os.listdir(TRAIN))

#Create test directory
try:
    os.makedirs(TEST)
except:
    pass

import numpy as np

# import PIL to display images and to create images from arrays
from PIL import Image

def saveDigit(digit, filepath):
    digit = digit.reshape(28,28)
    digit = digit.astype(np.uint8)

    img = Image.fromarray(digit)
    img.save(filepath)


# -

# save training images

for index, row in train_df.iterrows():

    label,digit = row[0], row[1:]

    folder = TRAIN/str(label)
    filename = f"{index}.jpg"
    filepath = folder/filename

    digit = digit.values

    saveDigit(digit, filepath)

# save testing images

for index, digit in test_df.iterrows():

    folder = TEST
    filename = f"{index}.jpg"
    filepath = folder/filename

    digit = digit.values

    saveDigit(digit, filepath)

# ##  Display images
#
# To check whether everything worked as expected, let's take a look at a few images from each folder.

# + _kg_hide-input=true
# import matplotlib to arrange the images properly
import matplotlib.pyplot as plt

def displayTrainingData():
    fig = plt.figure(figsize=(5,10))

    for rowIndex in range(1, 10):
        subdirectory = str(rowIndex)
        path = TRAIN/subdirectory
        images = os.listdir(path)

        for sampleIndex in range(1, 6):
            randomNumber = random.randint(0, len(images)-1)
            image = Image.open(path/images[randomNumber])
            ax = fig.add_subplot(10, 5, 5*rowIndex + sampleIndex)
            ax.axis("off")

            plt.imshow(image, cmap='gray')

    plt.show()

def displayTestingData():
    fig = plt.figure(figsize=(5, 10))

    paths = os.listdir(TEST)


    for i in range(1, 51):
        randomNumber = random.randint(0, len(paths)-1)
        image = Image.open(TEST/paths[randomNumber])

        ax = fig.add_subplot(10, 5, i)
        ax.axis("off")

        plt.imshow(image, cmap='gray')
    plt.show()


# transforms
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(
    path = TRAIN,
    test = TEST,
    valid_pct = 0.2,
    bs = 16,
    size = 28,
    #num_workers = 0,
    ds_tfms = tfms
)

# Let's perform normalization to make the CNN converge faster. fast.ai already defined the variable mnist_stats, that we can use to normalize our data. Alternatively, we can call normalize() without any paramters. In this case fast.ai simply calculates the exact stats needed for the dataset at hand.

mnist_stats

# + _kg_hide-output=true
data.normalize(mnist_stats)
# -

# all the classes in data
print(data.classes)

learn = cnn_learner(data, base_arch=models.resnet18, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)

learn.fit_one_cycle(cyc_len=5)

def test_learner_init():
    l = FastAIKernel(loss_func=None, metrics=None)
    assert l is not None

def test_learner_fit():
    k = FastAIKernel(loss_func=None, metrics=None)
    k.setup_learner()
    k.leaner.fit_one_cycle() # just test basic function

def test_basic_model():
    m = nn.Dropout(p=0.2)
    input = torch.randn(20, 16)
    output = m(input)
    assert sum(sum(input.numpy())) != sum(sum(output.numpy()))
