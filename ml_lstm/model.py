import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("✅ 使用 Apple M1 MPS 加速")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("✅ 使用 CUDA 加速")
        return torch.device("cuda")
    print("⚠️  使用 CPU")
    return torch.device("cpu")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)
