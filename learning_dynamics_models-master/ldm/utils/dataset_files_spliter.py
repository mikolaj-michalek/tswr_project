import torch
from pathlib import Path
import os


def split_dataset_files(path, train_ratio, val_ratio):
    list_files = list(path.glob("*.csv"))

    def extract_episode_number(file_path):
        file_name = Path(file_path).stem
        return int(file_name.split("_")[-1])

    list_files.sort(key=extract_episode_number)

    # split the dataset
    n_files = len(list_files)
    n_train = int(n_files * train_ratio)
    n_val = int(n_files * val_ratio)
    n_test = n_files - n_train - n_val

    rng_split = torch.Generator().manual_seed(444)
    train_files, val_files, test_files = torch.utils.data.random_split(
        list_files, [n_train, n_val, n_test], generator=rng_split
    )

    return train_files, val_files, test_files
