import torch
import torch.nn as nn


class LSTMBackbone(nn.Module):
    """
    LSTM backbone untuk data time-series AMR 3-fasa.

    Input:
        [B, T, F]
        B = batch
        T = timestep
        F = jumlah fitur

    Output:
        [B, hidden_dim]
    """

    def __init__(
        self,
        input_dim=31,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    ):
        super(LSTMBackbone, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.projection = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.output_dim = hidden_dim

    def forward(self, x):
        """
        x:
            [B, T, F]
        """

        # LSTM output:
        # [B, T, hidden_dim * directions]
        output, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            # Ambil hidden state terakhir dari:
            # forward dan backward
            h_forward = h_n[-2]
            h_backward = h_n[-1]

            h = torch.cat(
                [h_forward, h_backward],
                dim=1
            )
        else:
            h = h_n[-1]

        # [B, hidden_dim]
        z = self.projection(h)

        return z
    