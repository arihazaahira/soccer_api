import torch
import torch.nn as nn


class LSTMScorer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # regression score

    def forward(self, x, lengths=None):
        # x: (B, T, F). lengths optional for variable length (not strictly required here)
        packed_out, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # last layer hidden state
        return out.squeeze(-1)


class CNN1DScorer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x, lengths=None):
        # x: (B, T, F) where F=input_dim
        x = x.transpose(1, 2)  # -> (B, F, T)
        x = self.conv(x).squeeze(-1)
        out = self.fc(x)
        return out.squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, F)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # (B, num_classes)
        return out


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, lengths=None):
        # x: (B, T, F) where F=input_dim
        x = x.transpose(1, 2)  # -> (B, F, T)
        x = self.conv(x).squeeze(-1)
        logits = self.fc(x)
        return logits


class CNNLSTMClassifier(nn.Module):
    """Lightweight CNN+LSTM hybrid for sequence classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_channels: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x, lengths=None):
        # x: (B, T, F)
        x = x.transpose(1, 2)  # -> (B, F, T)
        feat = self.cnn(x)
        feat = feat.transpose(1, 2)  # -> (B, T, C)
        lstm_out, _ = self.lstm(feat)
        pooled = lstm_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits