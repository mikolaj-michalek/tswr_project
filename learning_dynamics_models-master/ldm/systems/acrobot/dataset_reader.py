import torch
import numpy as np
import pandas as pd
import wandb
from pathlib import Path

from ldm.utils.dataset_files_spliter import split_dataset_files


class AcrobotDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 #mode: str, # "train", "val", "test"
                 observation_window: int,
                 prediction_horizon: int,
                 stride: int,
                 device: str,):

        mode = dataset_path.split("/")[-1]
        dataset_path = dataset_path.replace(f"/{mode}", "")
        dataset_path = Path(dataset_path)
        
        train_files, val_files, test_files = split_dataset_files(
            path=dataset_path,
            train_ratio=0.7,
            val_ratio=0.2,
        )

        if mode == "train":
            files = train_files
        elif mode == "val":
            files = val_files
        elif mode == "test":
            files = test_files
        else:
            raise ValueError("mode should be one of ['train', 'val', 'test']")

        t_max = np.inf

        segment_length = observation_window + prediction_horizon

        list_files = list(files)

        def extract_episode_number(file_path):
            file_name = Path(file_path).stem
            return int(file_name.split("_")[-1])

        list_files.sort(key=extract_episode_number)

        M_cols = ["q1", "q2", "dq1", "dq2", "u"]
        U_cols = ["u"]
        X_cols = ["q1", "q2", "dq1", "dq2"]

        list_of_tensors_M = []
        list_of_tensors_U = []
        list_of_tensors_X0 = []
        list_of_tensors_X = []

        for file in list_files:

            df = pd.read_csv(file)
            df = df[df["t"] < t_max]  # cut off the end of the episode

            episode_length = len(df)

            for i in range(observation_window, episode_length - prediction_horizon, stride):
                M = df[M_cols].iloc[i - observation_window: i].to_numpy()

                U = df[U_cols].iloc[i: i + prediction_horizon].to_numpy()

                X = df[X_cols].iloc[i: i + prediction_horizon].to_numpy()

                M = torch.from_numpy(M).float().unsqueeze(0)
                U = torch.from_numpy(U).float().unsqueeze(0)
                X = torch.from_numpy(X).float().unsqueeze(0)
                X0 = X[:, 0, :].unsqueeze(0)

                list_of_tensors_M.append(M)
                list_of_tensors_U.append(U)
                list_of_tensors_X0.append(X0)
                list_of_tensors_X.append(X)

        self.M = torch.cat(list_of_tensors_M, dim=0).contiguous()
        self.U = torch.cat(list_of_tensors_U, dim=0).contiguous()
        self.X0 = torch.cat(list_of_tensors_X0, dim=0).contiguous()
        self.X = torch.cat(list_of_tensors_X, dim=0).contiguous()

        # make dtype float32
        self.M = self.M.float()
        self.U = self.U.float()
        self.X0 = self.X0.float()
        self.X = self.X.float()

        assert self.M.shape[0] == self.U.shape[0] == self.X0.shape[0] == self.X.shape[0]
        self.length = self.M.shape[0]
        
        # calac max channel values  dq1, dq2
        self.max_dq1 = self.X[:, :, 2].abs().max()
        self.max_dq2 = self.X[:, :, 3].abs().max()
        print(f"Mode : {mode}")
        print("max_dq1", self.max_dq1)
        print("max_dq2", self.max_dq2)
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            torch.Tensor: M      [time_dim, measurement_dim] -> time_dim = observation window
            torch.Tensor: U      [time_dim, control_dim]
            torch.Tensor: X0     [1, state_dim]

            torch.Tensor: X      [time, state_dim] -> time_dim = prediciton horizon
        """
        return self.M[idx], self.U[idx], self.X0[idx], self.X[idx]
