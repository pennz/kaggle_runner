import os

import cv2
from torch.utils.data import DataLoader, Dataset, sampler

from albumentations import (Blur, Compose, ElasticTransform, GaussNoise,
                            GridDistortion, HorizontalFlip, IAAEmboss,
                            MultiplicativeNoise, Normalize, OneOf,
                            OpticalDistortion, RandomBrightnessContrast,
                            RandomGamma, RandomRotate90, Resize,
                            ShiftScaleRotate, Transpose, VerticalFlip)
from albumentations.pytorch import ToTensor


class MockDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [Normalize(mean=mean, std=std, p=1), Resize(size, size), ToTensor(),]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".png")
        image = cv2.imread(path)
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples
