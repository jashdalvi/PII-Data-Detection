import torch.nn as nn
import torch

class LSTMHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers, dropout = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(in_features,
                            hidden_dim,
                            n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout)
        self.out_features = hidden_dim

    def forward(self, x):
        self.lstm.flatten_parameters()
        hidden, (_, _) = self.lstm(x)
        out = hidden
        return out