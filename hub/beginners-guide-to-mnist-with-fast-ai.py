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

# <br>
# <hr>
# **“You don't have to be great to start, but you have to start to be great.”** ~ Zig Ziglar
# <hr>
# <br>
#
# Hey friends,<br>
# this is a gentle introduction into Image Classification with the Python library fast.ai.
#
# What do you need to get started?
# - basic coding skills in Python
#
# [click if you don't know how to code at all](https://www.codecademy.com/learn/learn-python-3)<br>
# [click if you know some coding, but not Python](https://developers.google.com/edu/python/)<br>
# [click if you don't know the fast.ai course](https://course.fast.ai/videos/?lesson=1)
#
# Feel free to fork and tweak constants to improve accuracy. Just by doing so, I was able to get a score of 99.32% easily.
#
# Check out the [Q&A section](#Questions-and-Answers) of this notebook. If you have any questions or certain explanations weren't completely clear, let me know in the comments. I'm happy to help everyone :)
#
# 1. [Preparation](#Preparation)<br>
#     a.) [Explore Kaggle competitions page](#Explore-Kaggle-competitions-page)<br>
#     b.) [Setup environment](#Setup-environment)<br>
#     c.) [Explore data](#Explore-data)<br>
#     d.) [Data wrangling](#Data-wrangling)<br>
#     e.) [Display images](#Display-images)<br>
#     f.) [Load data into DataBunch](#Load-data-into-DataBunch)
# 3. [Training](#Training)
# 4. [Evaluation](#Evaluation)
# 5. [Prediction](#Prediction)
# 6. [Questions and Answers](#Questions-and-Answers)
#
# [click for the fast.ai documentation](https://docs.fast.ai)
#
#

# Be aware that you don't see all the code immediately. Some code is hidden! This is on purpose, we're focusing on the the most important aspects and you don't have to understand everything in detail yet.
# Nevertheless you can show the code by clicking on the **Code** button.
# <img src="https://i.imgur.com/IaxIqS1.gif">

# # Preparation

# ## Explore Kaggle competitions page
# Before you start coding anything, it is really important to get a thorough understanding of the competition. Kaggle already provides the most important information on the competition page ([https://www.kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer)). So the first thing you do when you start a new competition, is to read through the entire page and every tab within.
#
# **I will give you a quick overview of all the important information:**
#
# ### Goal of the competition
# "The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is."
#
# ### Evaluation
# "This competition is evaluated on the categorization accuracy of your predictions" <br>
# So the used metric is: Correctly classified Images / Total number of images
#
# ### Data ([read more here](https://www.kaggle.com/c/digit-recognizer/data))
#
#
# There are 3 different files: train.csv, test.csv, sample_submission.csv
#
# #### train.csv
# - greyscale images from zero through nine
# - file contains all necessary information for training the model
# - each row is one image
# - first row of each image is the label. It tells us which digit is shown.
# - other 784 rows are the pixels for each digit, they should be read like this
#
# `000 001 002 003 ... 026 027
# 028 029 030 031 ... 054 055
# 056 057 058 059 ... 082 083
#  |   |   |   |  ...  |   |
# 728 729 730 731 ... 754 755
# 756 757 758 759 ... 782 783`
#
# #### test.csv
# - greyscale images from zero through nine
# - structure is the same as in train.csv, but there are no labels
# - these 28000 images are used later to test how good our model is
#
# #### sample_submission.csv
# - show us, how to structure our prediction results to submit them to the competition
# - we need two columns: ImageId and Label
# - the rows don't need to be ordered
# - the submission file should look like this: <br> <br>
#     `ImageId, Label` <br>
#     `1, 3` <br>
#     `2, 4` <br>
#     `3, 9` <br>
#     `4, 1` <br>
#     `5, 7` <br>
#     `(27995 more lines)`
#

# ## Setup environment
# First of all make sure you enabled GPU so the model trains faster <br><br>
# Then import the fast.ai library

# + _kg_hide-input=true
# the following three lines are suggested by the fast.ai course
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# + _kg_hide-input=true
# hide warnings
import warnings
warnings.simplefilter('ignore')
# -

# the fast.ai library, used to easily build neural networks and train them
from fastai import *
from fastai.vision import *

# ## Explore Data
# I already summarized all the data we're given for the competition, but let's still check out the files:

# +
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

# Okay, perfect! The data looks just as expected.

# ## Data Wrangling
# ** = Getting the data into the right format**
#
# Looking at the [fast.ai documentation](https://docs.fast.ai/vision.data.html#ImageDataBunch) we can quickly see, that fast.ai only accepts image files for Computer Vision. In this competition we were not offered images, but .csv files containing the pixel values for each pixel of each image. If we want to use fast.ai we have to create images from the data we have.
#
# Fast.ai accepts image data in different formats. We will use the from_folder function of the ImageDataBunch class to load in the data. To do this we need all images in the following structure:
#
# `path\
#   train\
#     0\
#       ___.jpg
#       ___.jpg
#       ___.jpg
#     1\
#       ___.jpg
#       ___.jpg
#     2\
#       ...
#     3\
#       ...
#     ...
#   test\
#     ___.jpg
#     ___.jpg
#     ...
# `
#
# Let's first create the folder structure!
#
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

# Okay, all folders are created! The next step is to create the images inside of the folders from 'train.csv' and 'test.csv'. We will use the Image module from PIL to do this.
#
# we have to reshape each numpy array to have the desired dimensions of the image (28x28)
#
# `000 001 002 003 ... 026 027
# 028 029 030 031 ... 054 055
# 056 057 058 059 ... 082 083
#  |   |   |   |  ...  |   |
# 728 729 730 731 ... 754 755
# 756 757 758 759 ... 782 783`
#
# then we use the fromarray function to create a .jpg image from the numpy array and save it into the desired folder

# +
# import numpy to reshape array from flat (1x784) to square (28x28)
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

# + _kg_hide-input=true
print('samples of training data')
displayTrainingData()

# + _kg_hide-input=true
print('samples of testing data')
displayTestingData()
# -

# Let's also look at one image in more detail:

# + _kg_hide-input=true
image_path = TEST/os.listdir(TEST)[9]
image = Image.open(image_path)
image_array = np.asarray(image)


fig, ax = plt.subplots(figsize=(15, 15))

img = ax.imshow(image_array, cmap='gray')

for x in range(28):
    for y in range(28):
        value = round(image_array[y][x]/255.0, 2)
        color = 'black' if value > 0.5 else 'white'
        ax.annotate(s=value, xy=(x, y), ha='center', va='center', color=color)

plt.axis('off')
plt.show()
# -

# ## Load data into DataBunch
# Now that we have the right folder structure and images inside of the folders we can continue. Before training a model in fast.ai, we have to load the data into a [DataBunch](https://docs.fast.ai/basic_data.html#DataBunch), in this case, we use a ImageDataBunch, a special version of the DataBunch. Fast.ai offers different functions to create a DataBunch. We will use the from_folder method of the ImageDataBunch class to create the dataset.<br><br>
# There are different hyperparameters we can tweak to make the model perform better:
#
# - [valid_pct](#What-are-Train,-Test-and-Validation-datasets?)
# - [size](#What-image-size-should-I-choose?)
# - [num_workers](#What-is-multiprocessing?)
# - [ds_tfms](#What-are-transforms-and-which-transforms-should-I-use?)
# - [bs (batch size)](#What-is-the-batch-size?)

# transforms
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(
    path = str(TRAIN),
    test = str(TEST),
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

# # Training

# **First we have to create the CNN**
# <br>The next step is to create a convolutional neural network, short CNN. The most important thing about you need to know about CNNs is that we feed the model images and it outputs a probability for each possible category, so in this competition the digits from 0 through 9.

# <img src="https://i.imgur.com/Dm6wnCb.png">

# There are many different types of CNNs. For now we will only use one type of CNN, ResNets. The come in different sizes. There is **resnet18**, **resnet34** and a few more. At the moment, you just have to select the size.
#
# In fast.ai creating a CNN is really easy. We can use the cnn_learner object to do so and just have to pass the data that we want to feed into the CNN later and which architecture *(base_arch)* we want to use:

learn = cnn_learner(data, base_arch=models.resnet18, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)

# But as I said, there are many more architectures, you can use. Check out [How to select a CNN architecture?](#How-to-set-a-CNN-architecture?) to learn more.

# The model that we just created is already [pretrained](#What-is-transfer-learning?). In fast.ai this is the default setting.

# We can tweak one parameter when creating the CNN to make it perform better:
# - [ps (dropout propability)](#What-is-the-dropout-propability-and-how-high-should-I-set-it?)

# **Now it's time to train the neural network.**<br>
# We do so in fast.ai using the fit_one_cycle() function.<br>
# Training a CNN means, we feed it with data to predict which digit is shown. Then we compare the predictions with the actual results and update the CNN so that it better classifies the images later.
#
# There are different hyperparameters we can tweak to make the model perform better:
# - [cyc_len (number of epochs)](#What-are-epochs-and-how-many-epochs-should-I-train-my-CNN?)
# - max_lr (learning rate)
# - moms (momentum)

learn.fit_one_cycle(cyc_len=5)

# # Evaluation
# Create a ClassificationInterpretation object to evaluate your results.

interp = ClassificationInterpretation.from_learner(learn)

# Plot the 9 images with the highest loss. These are the images the CNN was most sure about, but still got wrong.

interp.plot_top_losses(9, figsize=(7, 7))

# A good way to summarize the performance of a classification algorithm is to create a confusion matrix. Confusion Matricies are used to understand which classes are most easily confused. As labeled on the axis, the x-axis shows the predicted classes and the y-axis the actual classes. So if (4/7)=10 it means that it happened 10 times that the CNN predicted a 7 but in reality if was a 4.

interp.plot_confusion_matrix()

# # Prediction
# Get the predictions on the test set. <br>
# learn.get_preds() returns a propability distribution over all possible classes for every given image.

class_score, y = learn.get_preds(DatasetType.Test)

# That means that for every image in the test set it predicts how likely each class is. In this case the highest value is obviously 1

probabilities = class_score[0].tolist()
[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]

# But we want the CNN to predict only one class. The class with the highest probability. Argmax returns the index of the highest value.

class_score = np.argmax(class_score, axis=1)

# This is exactly what we want.

class_score[0].item()

# The last step is creating the submission file. <br>
# "sample_submission.csv" is showing us the desired format

sample_submission =  pd.read_csv(INPUT/"sample_submission.csv")
display(sample_submission.head(2))
display(sample_submission.tail(2))

# Columns the submission file has to have:
#
# - ImageId: index in the test set, starting from 1, going up to 28000
# - Label: a prediction which digit the image shows

# remove file extension from filename
ImageId = [os.path.splitext(path)[0] for path in os.listdir(TEST)]
# typecast to int so that file can be sorted by ImageId
ImageId = [int(path) for path in ImageId]
# # +1 because index starts at 1 in the submission file
ImageId = [ID+1 for ID in ImageId]

submission  = pd.DataFrame({
    "ImageId": ImageId,
    "Label": class_score
})
# submission.sort_values(by=["ImageId"], inplace = True)
submission.to_csv("submission.csv", index=False)
display(submission.head(3))
display(submission.tail(3))

# # Questions and Answers
#
# - [What is MNIST?](#What-is-MNIST?)
# - [How to submit?](#How-to-submit?)
# - [What are Train, Test and Validation datasets?](#What-are-Train,-Test-and-Validation-datasets?)
# - [What is the batch size?](#What-is-the-batch-size?)
# - [What image size should I choose?](#What-image-size-should-I-choose?)
# - [What is multiprocessing?](#What-is-multiprocessing?)
# - [What are transforms and which transforms should I use?](#What-are-transforms-and-which-transforms-should-I-use?)
# - [What is transfer learning?](#What-is-transfer-learning?)
# - [How to improve your score?](#How-to-improve-your-score?)
# - [How to set a CNN architecture?](#How-to-set-a-CNN-architecture?)
# - [What is the best CNN architecture?](#What-is-the-best-CNN-architecture?)

# ## What is MNIST?
# MNIST is the perfect dataset to get started learning more about pattern recognition and machine learning. That is why people call it the "Hello World of machine learning".
# It's a large database of handwritten digits. There are a total of 70.000 grayscale images, each is 28x28 pixels. They show 10 different classes representing the numbers from 0 to 9. The dataset is split into 60.000 images in the training set and 10.000 in the test set.
# This competition is based on the MNIST dataset. However, the train-test distribution is different. Here, there are 42.000 images in the training set and 28.000 images in the test set.
# MNIST was published by the godfather of CNNs, Yann LeCun. You can find the original dataset [here](http://yann.lecun.com/exdb/mnist/)

# ## How to train a CNN?
# You need an image and the label (=Which number is this?). For example an image of a nine and a nine.<br>
# You feed in the image. The CNN predicts a number. You compare the predicted number to the actual number (the label). Then you update the CNN so that it better predicts this image the next time.

# ## What is the difference between the train and the test dataset?
#
# When you do machine learning often you have two different groups of data. You have a training dataset and a testing dataset.
#
# Let's explain this concept with the MNIST dataset! In the training dataset you have images for which you already know the label.
#
#
#
# With the training dataset you teach the model how a nine (and every other digit) looks like. Once you're done teaching your model how each digit looks like, you take the other dataset, the test dataset. You use the test dataset to check how good your model really is, by letting it predict data it has never seen before. To do so, you only feed the images to the CNN, not the labels. Then the CNN predicts the labels. Now you calculate the accuracy: Simply compare the predicted labels with the actual labels.
#
# #### Remember:
#
# Whenever we train a CNN we need to split the data into 2 parts:
# - training set: used to teach the model how each class looks like
# - test set: test accuracy on never before seen data

# ## What is the validation set?
#
#
#
# In this kernel we only have a training and a test set. That's why we split the test set to get a validation set. We do this with the 'valid_pct' parameter. This is one of the parameters you could tune to increase the accuracy.
#
# To learn more about this read [this stackoverflow post](https://stackoverflow.com/a/13623707)

# ## What to keep in mind when you split data into train, validation and test set?
# - all three datasets should have the same distribution over classes. In MNIST this would mean that when 5% of images in the training set are showing the digit two, also 5% of image in the test and validation set should show a two. But not that all classes should exist the same number of times.

# ## What is the batch size?
# The batch size refers to the number of images in one batch. Everytime an entire batch of images is passed through the neural network the weights of the neural network are updated
#
# Why would you increase the batch size:
# - to improve accuracy of the model
#
# Why would you decrease the batch size:
# - to train the network faster
# - to reduce the memory used
#
# A good value for the batch size is 16

# ### What are epochs and how many epochs should I train my CNN?
# How often we feed the entire data to our CNN. Train the CNN as long as the accuracy is improving

# ### What is the dropout propability and how high should I set it?
# In dropout you randomly deactivate nodes. That way the CNN has to learn redundant representations. This works well to reduce overfitting.

# ## What image size should I choose?
# (Most) CNNs need images of the same size. By setting the size parameter we tell fast.ai to make all images that size.
#
# Bigger images, result in more calculations and thus slower speed, but the accuracy improves. Smaller images on the other hand reduce accuracy but improve speed. Don't make the trainig images bigger than the original images, as this would only be a waste of time.
#
# Our data is already of shape 28x28.
#
# We could make the images smaller than 28x28. This would decrease the training time, but also decrease the accuracy. Because our CNN trains in a reasonable amount of time there is no reason to decrease the image size
#
# Never make the training image bigger than the original image.

# ## What is multiprocessing?
# In Computer Science there is this thing called multiprocessing. This means that we have two or more things happending at the same time. A computer typically has multiple CPUs and we make use of exactly that. Every CPU can run one process at a time. Ususally when we do multiprocessing every CPU gets its own task. Exactly that is the default for ImageDataBunch: number workers = number of CPUs. This works fine for Linux, but makes a lot of problems in Windows. If you're on Windows, set num_workers to 0, if you're on Linux, don't set anything, then it defaults to the number of CPUs
#
# If you use a cloud solution and are not sure which operating system is used, execute the following code

import platform; platform.system()

# ## What are transforms and which transforms should I use?
# To make models generalize better we can use so called transforms. They randomly change the image slightly. For example a bit of zoom or rotation. Fortunately, we don't really have to deal with transforms a lot in fast.ai. The package offers a convenient function called get_transforms() that returns pretty good values for transformations. In the case of digit recognition we want to tranform the data as much as possible so that it generalizes better, but only so much that the image would still be recognized by a human being.
#
# One parameter we definitely have to change for that reason is do_flip. If do_flip is set to True, random horizontal flips (with a probability of 0.5) would be applied to the images. This would result in images like this:

# + _kg_hide-input=true
flip_tfm = RandTransform(tfm=TfmPixel (flip_lr), kwargs={}, p=1, resolved={}, do_run=True, is_random=True, use_on_y=True)
folder = TRAIN/"3"
filename = os.listdir(folder)[0]
img = open_image(TRAIN/folder/filename)
display(img)
display(img.apply_tfms(flip_tfm))
# -

# We don't want that! This would confuse our CNN. Therefore we set do_flip to False

tfms = get_transforms(do_flip=False)

# ## What is normalization?
#
# Normalization is a technique use to make neural network converge faster. It sppeds up learning and makes the network converge faster.

# ## What is transfer learning?
# Transfer learning is a popular method in Deep Learning. It allows you to build accurate models fast.
# When you train a neural network, it (hopefully) gets better and better at finding patterns in data.
# Researchers found out that patterns a neural network learned in one problem can be helpful when solving another problem.
# In computer vision, the most common form of transfer learning is using **pretrained CNNs**. You simply take a CNN from one problem, with all its weights and biases, and use it as a baseline for another problem.
# Then you only have to [finetune the model](https://www.kaggle.com/christianwallenwein/beginners-guide-to-mnist-with-fast-ai#What-is-fine-tuning?). This works surprisingly good.
# Most **pretrained CNN's** for classification problems have been trained on ImageNet.
#
# Check out the [How to set a CNN architecture](http://)-section to learn how to use **pretrained models** for our problem.

# ## What is fine-tuning?
# Finetuning is a technique used in [transfer learning](#What-is-transfer learning?). It is used to optimize a pretrained model for the new task given.
#
# Tips for fine-tuning:
#
# ### Replace the last layer
# Most pretrained CNNs were trained on ImageNet. Imagenet has 1000 categories and therefore outputs probabilities for 1000 classes. Our task is to predict only 10 digits in MNIST. Therefore we have to replace the last layer, so that it only outputs 10 probabilities, one for each digit.
#
# ### Use a smaller learning rate
# When we train a CNN, at the beginning the weights are completely random. Compare these to pretrained weights and you will find that the pretrained
# weights are much better. Because the weights are already so good, we don't have to change the model that much.
# It is common to set a learning rate 10 times smaller than used for training usual, non-pretrained, models.
#
# ### Freeze weights of the first few layers
# The first few convolutional layers of a CNN capture universal features. These features could be edges or curves. They are so basic, that every model used for object classification needs to understand them. We don't want to change those low-level features, but rather focus on high-level features that are specific to the problem at hand. Therefore you want to freeze the weights for the first few layers and only train on the last layers.

# ## How to improve your score?
# **99.5%**: Good CNN + pooling layers + data augmentation + dropout+ batch normalization + decaying learning rate + advanced optimization
#
# **99.7%**: When you train the same CNN multiple times, you will not get the same results. So if you build a great CNN, train it 10 times and submit the results 10 times, you will probably beat 99.7%. Another approach is to build an ensemble of (great) CNNs. This means that you train multiple good, but different CNNs and combine their predictions.
#
# **99.75%**: This is the **best score reached inside of Kaggle Kernels**. Check out [this kernel](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist).
#
# **99.79%**: This is the **best score ever achieved** . It was obtained by Yann LeCun and colleagues from NYU in [this paper](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf).
#
# **99.9% - 100.0%**: To achieve a perfect or nearly **perfect score**, you have to train your CNN on the entire MNIST dataset (70.000 images). That way the CNN already "knows" the correct label for each image and simply has to remember it.
#
# The information here is an excerpt from [this post](https://www.kaggle.com/c/digit-recognizer/discussion/61480).

# ## How to set a CNN architecture?
# There are different opportunities. You could
# - [use a predefined architecture from fast.ai](#How-to-use-a-predefined-architecture-from-fast.ai?)
# - [use a predefined architecture from PyTorch](#How-to-use-a-predefined-architecture-from-PyTorch?)
# - [use a custom PyTorch architecture](#How-to-use-a-custom-PyTorch-architecture?)

# ### How to use a predefined architecture from fast.ai?
# At the time of writing, the following architectures are available:
#
# - resnet18, resnet34, resnet50, resnet101, resnet152
# - squeezenet1_0, squeezenet1_1
# - densenet121, densenet169, densenet201, densenet161
# - vgg16_bn, vgg19_bn
# - alexnet
#
# To change the architecture just change the base_arch parameter in cnn_learner to the desired architecture.

learn = cnn_learner(data, base_arch=models.densenet169, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)

# In fast.ai, the default is to use pretrained models. If you don't want to use a model pretrained on ImageNet, pass **pretrained=False** to the architecture like this:

learn = cnn_learner(data, base_arch=models.densenet169, pretrained=False, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)

# Check out the [fast.ai Computer Vision models zoo](https://docs.fast.ai/vision.models.html#Computer-Vision-models-zoo). Maybe there are new architectures available.

# ### How to use a predefined architecture from PyTorch?
#
# To do so, you first have to import the necessary submodule from PyTorch like this:

import torchvision.models

# At the time of writing, the following architectures are available:
#
# - resnet18
# - alexnet
# - vgg16
# - squeezenet1_0
# - densenet161
# - **inception_v3** (requires a tensor of size N x 3 x 299 x 299)
# - **googlenet**
# - **shufflenet_v2_x1_0**
# - **mobilenet_v2**
# - **resnext50_32x4d**
# - **wide_resnet50_2**
# - **mnasnet1_0**
#
# When we want to use an architecture from PyTorch, we can't simply use a cnn_learner object. Use the following code and replace google_net() with the desired architecture:

learn = Learner(data, torchvision.models.googlenet(), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)

# To use a model pretrained on ImageNet, pass **pretrained=True** to the architecture like this:

learn = Learner(data, torchvision.models.googlenet(pretrained=True), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)


# Check out the [torchvision.models documentation](https://pytorch.org/docs/stable/torchvision/models.html). Maybe there are new architectures available.

# ### How to use a custom PyTorch architecture?
# To use a custom PyTorch architecture, you first have to define it. I will not go into detail about how to do it. But there are wonderful tutorials that can show you how to do it. The skeleton of the class looks like this:

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # here you instantiate all the layers of the neural network and the activation function

    def forward(self, x):
        # here you define the forward propagation

        return x


# When you are finished, the CNN could look something like this:

# +
# set the batch size
batch_size = 16

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input is 28 pixels x 28 pixels x 3 channels
        # our original data was grayscale, so only one channel, but fast.ai automatically loads in the data as RGB
        self.conv1 = nn.Conv2d(3,16, 3, padding=1)
        self.conv2 = nn.Conv2d(16,32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(7*7*32, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        print (x.size())
        # x (28x28x3)
        x = self.conv1(x)
        # x (28x28x16)
        x = self.pool(x)
        # x (14x14x16)
        x = self.relu(x)

        x = self.conv2(x)
        # x (14x14x32)
        x = self.pool(x)
        # x (7x7x32)
        x = self.relu(x)

        # flatten images in batch
        print(x.size())
        x = x.view(-1,7*7*32)
        print(x.size())
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        return x


# -

# And to try the created CNN use the following code:

learn = Learner(data, CNN(), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)

# ## What is the best CNN architecture?
# According to [this post](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist) by [Chris Deotte](https://www.kaggle.com/cdeotte) the best architecture is:
#
# 784 input nodes
#
# > [32C3-32C3-32C5S2]
#
# - 2x (convolutional layer with 32 feature maps, 3x3 filter and stride 1)
# - convolutional layer with 32 feature maps, 5x5 filter and stride 2
# > [64C3-64C3-64C5S2]
#
# - 2x (convolutional layer with 64 feature maps, 3x3 filter and stride 1)
# - convolutional layer with 64 feature maps, 5x5 filter and stride 2
#
# 128 fully connected dense layers
#
# 10 output nodes
#
# with 40% dropout, batch normalization, and data augmentation added

# ## How to submit?
# Now we're throught the entire process of how to create a CNN with fast.ai to recognize digits. To submit a file to the the competition, you have two different options.
#
# Code in Kaggle Kernel
# 1. go to your kernel
# 2. commit the kernel
# 3. go back to all of your kernels
# 4. select the kernel again
# 5. scroll down to Output
# 6. click on 'Submit to Competition'
# <img src="https://i.imgur.com/CaFZm43.gif">
#
# Code locally on PC
# 1. go to [the competition](https://www.kaggle.com/c/digit-recognizer)
# 2. click on ['Submit Predictions'](https://www.kaggle.com/c/digit-recognizer/submit)
# 3. upload your submission file
# 4. add a description
# 5. click on 'Make Submission'
# <img src="https://i.imgur.com/m3W1BCS.gif">
#

# Fun Fact
# ![](https://images.squarespace-cdn.com/content/v1/5c293b5d55b02c783a5d8747/1551704286537-JG06FA61IJM7JJ94TAKX/ke17ZwdGBToddI8pDm48kPx25wW2-RVvoRgxIT6HShBZw-zPPgdn4jUwVcJE1ZvWQUxwkmyExglNqGp0IvTJZUJFbgE-7XRK3dMEBRBhUpxp8SK2yZDC8sLtVprOyP9xilpMqmr5qElvqElFYURlwnSzTaHLYeySc8Xfr8nFWbQ/Butterflies-taste-with-their-fet.gif?format=1000w)
# [source](https://www.learnsomethingeveryday.co.uk/#/4-march-2019/)

# TODO:
#
# - image augmentation https://www.kaggle.com/anisayari/generate-more-training-data-fastai
# - created images to zip file https://www.kaggle.com/anisayari/generate-more-training-data-fastai
# - better explanations
# - explain CNNs better
# - explain normalization
# - change channels from RGB to Grayscale in images to improve computation time
# - explain feed forward and backpropagation
# - explain how to make the model train faster
# - group Q&A questions: general questions, training, etc.
# - check spelling
# - visualize concepts
# - make tutorial  simpler
# - explain dropout better
# - explain train and test set
# - expain train and validation set
# - supervised vs unsupervised learning
