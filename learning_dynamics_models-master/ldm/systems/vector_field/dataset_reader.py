import torch

class DummyVFDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 observation_window: int,
                 prediction_horizon: int, # in samples
                 stride: int,
                 device: str):

        n_examples = 3
        time = 2000
        dt = 0.01
        if "train" in dataset_path:
            #angle = torch.linspace(-torch.pi/2, torch.pi/2, time + 1)
            t = torch.linspace(0, time*dt, time + 1)
            r1 = 1.0
            x1 = torch.cos(t) * r1
            y1 = torch.sin(t) * r1
            r2 = 10.0
            x2 = torch.cos(t + torch.pi) * r2
            y2 = torch.sin(t + torch.pi) * r2
            x = torch.stack([x1, x2], dim=0)
            y = torch.stack([y1, y2], dim=0)
        elif "val" in dataset_path:
            #angle = torch.linspace(-torch.pi/2, torch.pi/2, time + 1)
            r = 5.0
            t = torch.linspace(0, time*dt, time + 1)
            x1 = torch.cos(t) * r
            y1 = torch.sin(t) * r
            x2 = torch.cos(t + torch.pi) * r
            y2 = torch.sin(t + torch.pi) * r
            x = torch.stack([x1, x2], dim=0)
            y = torch.stack([y1, y2], dim=0)
        else:
            raise ValueError(f"Unknown dataset path: {dataset_path}")

        self.X = torch.stack([x, y], dim=-1)  # [n_examples, time, 2]

        #import matplotlib.pyplot as plt
        #plt.plot(self.X[0, :, 0], self.X[0, :, 1], label='Example 1')
        #plt.plot(self.X[1, :, 0], self.X[1, :, 1], label='Example 2')
        #plt.title('Trajectories of Examples')
        #plt.xlabel('X Position')
        #plt.ylabel('Y Position')
        #plt.legend()
        #plt.show()

        index_list = []
        
        i = 0
        example_idx = 0
        while True:
            if i + prediction_horizon + 1 >= len(self.X[example_idx]):
                i = 0
                example_idx += 1
                if example_idx >= y.shape[0]:
                    break
                        
            index_list.append((example_idx, i, i + prediction_horizon))
            i += 1

        self.index_tensor = torch.tensor(index_list, dtype=torch.long)
        self.length = (self.index_tensor).shape[0]
        
        # index_tensor and data to device
        self.index_tensor = self.index_tensor.to(device)        
        self.X = self.X.to(device)
        print(f"Dataset length: {self.length}")
        print(f"Prediction horizon: {prediction_horizon}")
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index 

        Returns:
            torch.Tensor: M      not used in this case = None
            torch.Tensor: U      [time_dim, 1] = zeros
            torch.Tensor: X0     [1, state_dim]
            torch.Tensor: X      [time, state_dim] -> time_dim = prediciton horizon
        """
        example_idx, t, t_plus_tp = self.index_tensor[idx, :].unbind(-1)
        
        return torch.zeros(1), \
               torch.zeros_like(self.X[example_idx, t:t_plus_tp, :1]), \
               self.X[example_idx, t, :].unsqueeze(0), \
               self.X[example_idx, t+1:t_plus_tp + 1, :]    
