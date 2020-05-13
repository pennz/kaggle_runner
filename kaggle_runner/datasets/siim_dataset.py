import os

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset, sampler

from kaggle_runner.datasets.coders import run_length_decode
from kaggle_runner.datasets.transfomers import get_transforms


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
