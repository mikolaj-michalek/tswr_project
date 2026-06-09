import numpy as np
import torch
import pandas as pd

class VWGolfDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 observation_window: int, # in samples
                 prediction_horizon: int, # in samples
                 stride: int, # in samples
                 device: str,
                 shift_u1: int = 0,
                 shift_u2: int = 0,):

        # Observation columns
        M_cols = ["v_x", "v_y", "r", 
                  "omega_wheels_rear", "omega_wheels_front", "delta"]
        
        # Control columns
        U_cols = ["omega_wheels_rear", "omega_wheels_front", "delta"]

        # State vector columns
        X_cols = ["v_x", "v_y", "r"]
        
        self.df = pd.read_csv(dataset_path)
        #self.df = self.df.drop(self.df[self.df.run_id == 14].index)
        #self.df = self.df[:10000]
        self.M = torch.from_numpy(self.df[M_cols].to_numpy()).float().contiguous()
        self.U = torch.from_numpy(self.df[U_cols].to_numpy()).float().contiguous()
        self.X = torch.from_numpy(self.df[X_cols].to_numpy()).float().contiguous()

        self.shift_u1 = shift_u1
        self.shift_u2 = shift_u2

        #a = 0
        #for idx in [150, 500, 1000]:
        #    #idx = 100
        #    xu = torch.cat([self.X, self.U], dim=-1)
        #    diffs = (xu[idx, None] - xu).square().sum(-1)
        #    idxs = torch.where(diffs < 0.01)[0].numpy()
        #    X = self.X.numpy()

        #    import matplotlib.pyplot as plt
        #    plt.subplot(221)
        #    plt.plot(X[idxs, 0], X[idxs, 1], 'g.')
        #    plt.plot(X[idxs+1, 0], X[idxs+1, 1], 'r.')
        #    for idx in idxs:
        #        dX = np.stack([X[idx, 0], X[idx+1, 0]], axis=-1)
        #        dY = np.stack([X[idx, 1], X[idx+1, 1]], axis=-1)
        #        plt.plot(dX, dY, 'b')
        #    plt.subplot(222)
        #    plt.plot(X[idxs, 0], X[idxs, 2], 'g.')
        #    plt.plot(X[idxs+1, 0], X[idxs+1, 2], 'r.')
        #    for idx in idxs:
        #        dX = np.stack([X[idx, 0], X[idx+1, 0]], axis=-1)
        #        dY = np.stack([X[idx, 2], X[idx+1, 2]], axis=-1)
        #        plt.plot(dX, dY, 'b')
        #    plt.subplot(223)
        #    plt.plot(X[idxs, 1], X[idxs, 2], 'g.')
        #    plt.plot(X[idxs+1, 1], X[idxs+1, 2], 'r.')
        #    for idx in idxs:
        #        dX = np.stack([X[idx, 1], X[idx+1, 1]], axis=-1)
        #        dY = np.stack([X[idx, 2], X[idx+1, 2]], axis=-1)
        #        plt.plot(dX, dY, 'b')
        #    plt.show()


        #import matplotlib.pyplot as plt
        #plt.hist(self.df["v_x"].to_numpy(), bins=100)
        #plt.title("v_x distribution")
        #plt.xlabel("v_x [m/s]")
        #plt.ylabel("Count")
        #plt.show()

#        import matplotlib.pyplot as plt
#        import numpy as np
#        for i in range(-20, 20):
#            H = 1000
#            yaw = self.df["yaw"] + i * 0.01
#            global_v_x = self.df["v_x"] * np.cos(yaw) - self.df["v_y"] * np.sin(yaw)
#            global_v_y = self.df["v_x"] * np.sin(yaw) + self.df["v_y"] * np.cos(yaw)
#            dxdt = np.diff(self.X[:H, 0]) / 0.01
#            dydt = np.diff(self.X[:H, 1]) / 0.01
#            error = np.mean((dxdt - global_v_x[:H-1])**2 + (dydt - global_v_y[:H-1])**2)
#            print(f"Yaw offset: {i*0.01:.2f}, error: {error:.4f}")
#
#        for i in range(20):
#            H = 1000
#            yaw = self.df["yaw"]
#            global_v_x = self.df["v_x"] * np.cos(yaw) - self.df["v_y"] * np.sin(yaw)
#            global_v_y = self.df["v_x"] * np.sin(yaw) + self.df["v_y"] * np.cos(yaw)
#            global_v_x = global_v_x[i:]
#            global_v_y = global_v_y[i:]
#            dxdt = np.diff(self.X[:H-i, 0]) / 0.01
#            dydt = np.diff(self.X[:H-i, 1]) / 0.01
#            error = np.mean((dxdt - global_v_x[:H-i-1])**2 + (dydt - global_v_y[:H-i-1])**2)
#            print(f"Time offset: {i*0.01:.2f}, error: {error:.4f}")
#
#        yaw = self.df["yaw"]
#        global_v_x = self.df["v_x"] * np.cos(yaw) - self.df["v_y"] * np.sin(yaw)
#        global_v_y = self.df["v_x"] * np.sin(yaw) + self.df["v_y"] * np.cos(yaw)
#        yawp = yaw + 0.08
#        global_v_xp = self.df["v_x"] * np.cos(yawp) - self.df["v_y"] * np.sin(yawp)
#        global_v_yp = self.df["v_x"] * np.sin(yawp) + self.df["v_y"] * np.cos(yawp)
#        yawm = yaw - 0.08
#        global_v_xm = self.df["v_x"] * np.cos(yawm) - self.df["v_y"] * np.sin(yawm)
#        global_v_ym = self.df["v_x"] * np.sin(yawm) + self.df["v_y"] * np.cos(yawm)
#        H = 1000
#        plt.subplot(321)
#        plt.plot(0.01 * np.arange(H-1), np.diff(self.X[:H, 0]) / 0.01, label="dx/dt")
#        plt.plot(0.01 * np.arange(H), global_v_x[:H], label="global_v_x")
#        plt.legend()
#        plt.subplot(322)
#        plt.plot(0.01 * np.arange(H-1), np.diff(self.X[:H, 1]) / 0.01, label="dy/dt")
#        plt.plot(0.01 * np.arange(H), global_v_y[:H], label="global_v_y")
#        plt.legend()
#        plt.subplot(323)
#        to = 7
#        plt.plot(0.01 * np.arange(H-to-1), np.diff(self.X[:H-to, 0]) / 0.01, label="dx/dt")
#        plt.plot(0.01 * np.arange(H-to), global_v_x[to:H], label="global_v_x time offset")
#        plt.legend()
#        plt.subplot(324)
#        plt.plot(0.01 * np.arange(H-to-1), np.diff(self.X[:H-to, 1]) / 0.01, label="dy/dt")
#        plt.plot(0.01 * np.arange(H-to), global_v_y[to:H], label="global_v_y time offset")
#        plt.legend()
#        plt.subplot(325)
#        plt.plot(0.01 * np.arange(H-1), np.diff(self.X[:H, 0]) / 0.01, label="dx/dt")
#        plt.plot(0.01 * np.arange(H), global_v_xm[:H], label="global_v_x-")
#        plt.legend()
#        plt.subplot(326)
#        plt.plot(0.01 * np.arange(H-1), np.diff(self.X[:H, 1]) / 0.01, label="dy/dt")
#        plt.plot(0.01 * np.arange(H), global_v_ym[:H], label="global_v_y-")
#        plt.legend()
#        plt.show()

        index_list = []
        droped_data = 0
        
        for i in range(observation_window, len(self.df) - prediction_horizon - 1, stride):
    
            # i - current time sample      
            t_minus_to = i - observation_window # start of observation window
            t_plus_tp = i + prediction_horizon # end of prediction horizon
            
            t_plus_tp_check = t_plus_tp + max(shift_u1, shift_u2)
            if t_plus_tp_check >= len(self.df):
                continue
            if self.df["run_id"].iloc[t_minus_to] != self.df["run_id"].iloc[t_plus_tp_check]:
            #if self.df["run_id"].iloc[t_minus_to] != self.df["run_id"].iloc[t_plus_tp]:
                droped_data += 1
                continue
                        
            index_list.append((t_minus_to, i, t_plus_tp))

        #print(f"Droped data: {droped_data / (len(index_list)+droped_data) * 100}%")
                
        self.index_tensor = torch.tensor(index_list, dtype=torch.long)
        self.length = (self.index_tensor).shape[0]
        
        # index_tensor and data to device
        self.index_tensor = self.index_tensor.to(device)        
        self.M = self.M.to(device)
        self.U = self.U.to(device)
        self.X = self.X.to(device)
        #print(f"Dataset length: {self.length}")
        #print(f"Observation window: {observation_window}, Prediction horizon: {prediction_horizon}, Stride: {stride}")
        
        # df max abs values
        self.max_values = self.df.abs().max().to_dict() 
        #print(f"Max values: {self.max_values}")
        
        
    #####################################################################################
    #                                   Dataset interface                               #
    #####################################################################################

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
        t_minus_to, t, t_plus_tp = self.index_tensor[idx, :].unbind(-1)
        # No shifts right now
        #U = torch.stack([self.U[t + self.shift_u1:t_plus_tp + self.shift_u1, 0],
        #                 self.U[t + self.shift_u2:t_plus_tp + self.shift_u2, 1]], dim=-1)
        U = self.U[t:t_plus_tp, :]
        
        return self.M[t_minus_to:t, :], \
               U, \
               self.X[t, :].unsqueeze(0), \
               self.X[t+1:t_plus_tp + 1, :]    
