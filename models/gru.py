"""
GRU-based model for univariate time series forecasting.
Lighter alternative to LSTM: fewer parameters, similar expressive power.
Uncertainty via MC Dropout (same approach as rnn.py).

The only architectural difference vs rnn.py: nn.GRU instead of nn.LSTM.
GRU has a single hidden state (no separate cell state), so the forward pass
returns (output, h_n) instead of (output, (h_n, c_n)).

Dependencies: torch
  pip install torch
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_NAME = "GRU"

SEQ_LEN     = 30
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 32
MAX_EPOCHS  = 150
LR          = 1e-3
PATIENCE    = 15
MC_SAMPLES  = 300


class _GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru  = nn.GRU(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.drop = nn.Dropout(DROPOUT)
        self.fc   = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.gru(x)              # GRU: no separate cell state
        out    = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class GRUModel:
    def __init__(self, net, mean_: float, std_: float, last_seq_norm: np.ndarray):
        self.net       = net
        self.mean_     = mean_
        self.std_      = std_
        self._last_seq = last_seq_norm

    def predict(self, h: int) -> dict:
        self.net.train()   # keep dropout active for MC Dropout

        all_preds = np.empty((MC_SAMPLES, h), dtype=np.float32)
        with torch.no_grad():
            for s in range(MC_SAMPLES):
                seq = self._last_seq.copy()
                for t in range(h):
                    x    = torch.tensor(seq[-SEQ_LEN:], dtype=torch.float32).view(1, SEQ_LEN, 1)
                    step = self.net(x).item()
                    seq  = np.append(seq, step)
                    all_preds[s, t] = step

        mean_n = all_preds.mean(axis=0)
        std_n  = all_preds.std(axis=0)
        mean   = mean_n * self.std_ + self.mean_
        std    = std_n  * self.std_

        return {
            "mean":     mean,
            "lower_80": mean - 1.28 * std,
            "upper_80": mean + 1.28 * std,
            "lower_95": mean - 1.96 * std,
            "upper_95": mean + 1.96 * std,
        }


def _make_windows(arr: np.ndarray):
    X, Y = [], []
    for i in range(len(arr) - SEQ_LEN):
        X.append(arr[i:i + SEQ_LEN])
        Y.append(arr[i + SEQ_LEN])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train(y_train, y_val=None) -> GRUModel:
    values = y_train.values.astype(np.float32)
    mean_  = values.mean()
    std_   = values.std() + 1e-8
    norm   = (values - mean_) / std_

    X_tr, Y_tr = _make_windows(norm)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).unsqueeze(-1), torch.from_numpy(Y_tr)),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    val_loader = None
    if y_val is not None:
        norm_val = (y_val.values.astype(np.float32) - mean_) / std_
        context  = np.concatenate([norm[-SEQ_LEN:], norm_val])
        X_v, Y_v = _make_windows(context)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_v).unsqueeze(-1), torch.from_numpy(Y_v)),
            batch_size=BATCH_SIZE,
        )

    net       = _GRUNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(net(xb), yb).backward()
            optimizer.step()

        if val_loader is not None:
            net.eval()
            with torch.no_grad():
                val_loss = np.mean([criterion(net(xb), yb).item() for xb, yb in val_loader])
            if val_loss < best_loss:
                best_loss  = val_loss
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}  (best val loss: {best_loss:.6f})")
                    break
        elif epoch == MAX_EPOCHS:
            print(f"  Training finished ({MAX_EPOCHS} epochs)")

    if best_state is not None:
        net.load_state_dict(best_state)

    return GRUModel(net, float(mean_), float(std_), last_seq_norm=norm[-SEQ_LEN:])
