import warnings
warnings.simplefilter('ignore')

from fastai import *
from fastai.vision import *
from fastai.vision import get_transforms

import os

from pathlib import Path

import pandas as pd

INPUT = Path("../input/digit-recognizer")
os.listdir(INPUT)

train_df = pd.read_csv(INPUT/"train.csv")

test_df = pd.read_csv(INPUT/"test.csv")

TRAIN = Path("../train")
TEST = Path("../test")

for index in range(10):
    try:
        os.makedirs(TRAIN/str(index))
    except:
        pass

sorted(os.listdir(TRAIN))

try:
    os.makedirs(TEST)
except:
    pass

from PIL import Image

def saveDigit(digit, filepath):
    digit = digit.reshape(28,28)
    digit = digit.astype(np.uint8)

    img = Image.fromarray(digit)
    img.save(filepath)


for index, row in train_df.iterrows():

    label,digit = row[0], row[1:]

    folder = TRAIN/str(label)
    filename = f"{index}.jpg"
    filepath = folder/filename

    digit = digit.values

    saveDigit(digit, filepath)


for index, digit in test_df.iterrows():

    folder = TEST
    filename = f"{index}.jpg"
    filepath = folder/filename

    digit = digit.values

    saveDigit(digit, filepath)


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

print('samples of training data')
displayTrainingData()

print('samples of testing data')
displayTestingData()


image_path = TEST/os.listdir(TEST)[9]
image = Image.open(image_path)
image_array = np.asarray(image)


#fig, ax = plt.subplots(figsize=(15, 15))

#img = ax.imshow(image_array, cmap='gray')

#for x in range(28):
#    for y in range(28):
#        value = round(image_array[y][x]/255.0, 2)
#        color = 'black' if value > 0.5 else 'white'
#        ax.annotate(s=value, xy=(x, y), ha='center', va='center', color=color)
#
#plt.axis('off')
#plt.show()


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


print(mnist_stats)

data.normalize(mnist_stats)

print(data.classes)





learn = cnn_learner(data, base_arch=models.resnet18, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)





learn.fit_one_cycle(cyc_len=5)


interp = ClassificationInterpretation.from_learner(learn)


interp.plot_top_losses(9, figsize=(7, 7))


interp.plot_confusion_matrix()













import platform; platform.system()


flip_tfm = RandTransform(tfm=TfmPixel (flip_lr), kwargs={}, p=1, resolved={}, do_run=True, is_random=True, use_on_y=True)
folder = TRAIN/"3"
filename = os.listdir(folder)[0]
img = open_image(TRAIN/folder/filename)
display(img)
display(img.apply_tfms(flip_tfm))


tfms = get_transforms(do_flip=False)







learn = cnn_learner(data, base_arch=models.densenet169, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)


learn = cnn_learner(data, base_arch=models.densenet169, pretrained=False, metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)



import torchvision.models


learn = Learner(data, torchvision.models.googlenet(), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)


learn = Learner(data, torchvision.models.googlenet(pretrained=True), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # here you instantiate all the layers of the neural network and the activation function

    def forward(self, x):
        # here you define the forward propagation

        return x



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

learn = Learner(data, CNN(), metrics=accuracy, model_dir="/tmp/models", callback_fns=ShowGraph)
