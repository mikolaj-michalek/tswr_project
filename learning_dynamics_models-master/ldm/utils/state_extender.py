import torch

class DummyStateExtenderModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(DummyStateExtenderModel, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, obs_history):
        batch_size = obs_history.shape[0]
        return torch.zeros((batch_size, 1, self.hidden_size), device=obs_history.device), None

class CNNStateExtenderModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(CNNStateExtenderModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=5, stride=2)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.conv2 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, stride=2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.conv3 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=5, stride=2)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, obs_history):
        x = obs_history.permute(0, 2, 1)  # (batch_size, input_size, time_steps)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch_size, hidden_size)
        return x.unsqueeze(1), None  # (batch_size, 1, hidden_size)
    

class StateExtender(torch.nn.Module):
    def __init__(self, extender_type: str, input_dim: int, output_dim: int, preprocessor):
        super(StateExtender, self).__init__()
        self.extender_type = extender_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.preprocessor = preprocessor
        self.hidden_dim = 32
        # Initialize the state extender model based on the type
        if extender_type == "gru":
            self.model = torch.nn.GRU(input_size=input_dim, hidden_size=self.hidden_dim, batch_first=True)
        elif extender_type == "lstm":
            self.model = torch.nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, batch_first=True)
        elif extender_type == "cnn":
            self.model = CNNStateExtenderModel(input_size=input_dim, hidden_size=self.hidden_dim)
        elif extender_type == "dummy":
            self.model = DummyStateExtenderModel(input_size=input_dim, hidden_size=self.hidden_dim)
        else:
            raise ValueError(f"Unknown state extender type: {extender_type}")

        self.fc = torch.nn.Linear(self.hidden_dim, output_dim)

    def forward(self, obs_history, X0):
        # Preprocess observations
        processed_obs = self.preprocessor(obs_history)
        # Pass through the state extender model
        hidden_states, _ = self.model(processed_obs)
        output = 2. * self.fc(hidden_states[:, -1, :])  # (batch_size, hidden_dim)
        # Return the last output in the sequence stacked with X0
        return torch.cat([X0, output.unsqueeze(1)], dim=-1)  # (batch_size, output_dim + state_dim)