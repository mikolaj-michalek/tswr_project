import os
import torch


def read_datasets(system, batch_size, observation_window_len, rollout_len, stride, device, num_workers=0, shift_u1=0, shift_u2=0, dataset=None):
    if system == "f1tenth":
        from ldm.systems.f1tenth.dataset_reader import F1tenthDataset
        if dataset == "new":
            dataset_path = os.path.join(os.path.dirname(__file__), "../..", "datasets", "f1tenth", "260130_mpc_expert_train.csv")
        elif dataset == "new_filtered":
            dataset_path = os.path.join(os.path.dirname(__file__), "../..", "datasets", "f1tenth", "260130_mpc_expert_sgf_p5_w25_train.csv")
        elif dataset == "monza":
            dataset_path = os.path.join(os.path.dirname(__file__), "../..", "datasets", "f1tenth", "monza_train.csv")
        else:
            dataset_path = os.path.join(os.path.dirname(__file__), "../..", "datasets", "f1tenth", "lab_rss1_train_fix.csv")
        dataset_class = F1tenthDataset
    elif system == "vw_golf":
        from ldm.systems.vw_golf.dataset_reader import VWGolfDataset
        #dataset_path = os.path.join(os.path.dirname(__file__), "../..", "datasets", "vehicle_dynamics", "oct_8_9", "train.csv")
        dataset_path = os.path.join(os.path.dirname(__file__), "../..", "datasets", "vehicle_dynamics", "oct", "train.csv")
        dataset_class = VWGolfDataset
    elif system == "acrobot":
        from ldm.systems.acrobot.dataset_reader import AcrobotDataset
        dataset_path = os.path.join(os.path.dirname(__file__), "../..", "datasets", "acrobot", "train")
        dataset_class = AcrobotDataset
    else:
        raise ValueError(f"Unknown system: {system}")

    train_dataset = dataset_class(
        dataset_path=dataset_path,
        observation_window=observation_window_len,
        prediction_horizon=rollout_len,
        stride=stride,
        device=device,
        shift_u1=shift_u1,
        shift_u2=shift_u2
    )

    val_dataset = dataset_class(
        dataset_path=dataset_path.replace("train", "val"),
        observation_window=observation_window_len,
        prediction_horizon=rollout_len,
        stride=stride,
        device=device,
        shift_u1=shift_u1,
        shift_u2=shift_u2,
    )

    test_dataset = dataset_class(
        dataset_path=dataset_path.replace("train", "test"),
        observation_window=observation_window_len,
        prediction_horizon=rollout_len,
        stride=stride,
        device=device,
        shift_u1=shift_u1,
        shift_u2=shift_u2,
    )

    # Dataset loaders
    train_dataset = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )

    test_dataset = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        #shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
    )
    
    return train_dataset, val_dataset, test_dataset