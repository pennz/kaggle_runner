import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from kaggle_runner.datasets.siim_dataset import SIIMDataset


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

    image_dataset = SIIMDataset(df_all, fnames, data_folder, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader
