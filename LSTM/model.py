import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.LSTM(input_size=hidden_sizes[i-1], hidden_size=hidden_sizes[i], batch_first=True))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_sizes[-1], output_size)  # output 3 sources

    def forward(self, x):
        out = x
        for layer in self.layers:
            out, _ = layer(out)
            out = self.dropout(out)
        out = self.fc(out)
            
         # Apply constraint to output: sum of the output channels should be the size of the input channel
        sum_out = torch.sum(out, dim=-1, keepdim=True)  # sum_y.shape = (batch_size, seq_len, 1)
        out = out * x.sum(dim=-1, keepdim=True) / sum_out  # Adjust y to satisfy constraint

        return out

