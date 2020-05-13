import cv2

import albumentations as A
from albumentations import (Blur, Compose, ElasticTransform, GaussNoise,
                            GridDistortion, HorizontalFlip, IAAEmboss,
                            MultiplicativeNoise, Normalize, OneOf,
                            OpticalDistortion, RandomBrightnessContrast,
                            RandomGamma, RandomRotate90, Resize,
                            ShiftScaleRotate, Transpose, VerticalFlip)
from albumentations.pytorch import ToTensor


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
        [Resize(size, size), Normalize(mean=mean, std=std, p=1), ToTensor(),]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms
