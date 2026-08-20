import torch
import torch.nn as nn

from models.lstm_time import LSTMBackbone


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_dim=31,
        hidden_dim=128,
        projection_dim=128,
        num_layers=2,
        dropout=0.2,
        bidirectional=True
    ):
        super(LSTMModel, self).__init__()

        self.backbone = LSTMBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.features_dim = hidden_dim

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):

        # Input:
        # [B, T, F]
        features = self.backbone(x)

        # Projection untuk contrastive learning
        output = self.projector(features)

        return output