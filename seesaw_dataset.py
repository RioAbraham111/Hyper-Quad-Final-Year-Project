import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class SeesawDataset(Dataset):
    def __init__(self, csv_path, window_size=50, fit_scaler=True,
                 x_scaler=None, y_scaler=None):

        self.window_size = window_size

        # Load CSV
        df = pd.read_csv(csv_path)

        self.theta = df["theta"].values
        self.theta_dot = df["theta_dot"].values
        self.tau = df["tau"].values

        self.J = df["J"].values
        self.b = df["b"].values
        self.k = df["k"].values

        # Build sliding windows
        X, y = self._build_windows()

        # Scale data
        if fit_scaler:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()

            X = self.x_scaler.fit_transform(X)
            y = self.y_scaler.fit_transform(y)
        else:
            self.x_scaler = x_scaler
            self.y_scaler = y_scaler

            X = self.x_scaler.transform(X)
            y = self.y_scaler.transform(y)

        # Convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def _build_windows(self):
        X, y = [], []

        for i in range(self.window_size, len(self.theta)):
            X.append(
                np.concatenate([
                    self.theta[i-self.window_size:i],
                    self.theta_dot[i-self.window_size:i],
                    self.tau[i-self.window_size:i],
                ])
            )
            y.append([self.J[i], self.b[i], self.k[i]])

        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

