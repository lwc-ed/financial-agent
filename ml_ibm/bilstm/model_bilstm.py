import torch
import torch.nn as nn

class MyBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(MyBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 取最後一個時間步的雙向輸出
        last_time_step_out = lstm_out[:, -1, :] 
        out = self.fc1(last_time_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out