import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# Folder containing the 81 CSV files
data_folder = "seesaw_simulations"

# List all CSV files
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

# Feature extraction function
def extract_features(df):
    # Remove time column
    theta = df["theta"].values
    theta_dot = df["theta_dot"].values
    tau = df["tau"].values

    features = []

    for signal in [theta, theta_dot, tau]:
        mean_val = np.mean(signal)
        std_val  = np.std(signal)
        max_val  = np.max(signal)
        min_val  = np.min(signal)
        rms_val  = np.sqrt(np.mean(signal**2))

        features.extend([mean_val, std_val, max_val, min_val, rms_val])

    return np.array(features)


# Build dataset
X_list = []

for file in csv_files:
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)

    features = extract_features(df)
    X_list.append(features)

# Convert to numpy array
xObs = torch.tensor(np.vstack(X_list), dtype=torch.float32)

print("Feature matrix shape:", xObs.shape)

y_data = pd.read_csv("simulation_parameters.csv")
y_numeric = y_data[["J", "b", "k", "tau_limit"]].astype(float)
yObs = torch.tensor(y_numeric.values, dtype=torch.float32)
print("Target matrix shape:", yObs.shape)

nInput  = 15
nHidden = 50
nOutput = 4


class MLPexplicit(nn.Module):
    '''
    Multi-layer perceptron for non-linear regression.
    '''
    def __init__(self, nInput, nHidden, nOutput):
        super(MLPexplicit, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.linear1 = nn.Linear(self.nInput, self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nHidden)
        self.linear4 = nn.Linear(self.nHidden, self.nOutput)
        self.ReLU    = nn.ReLU()

    def forward(self, x):
        h1 = self.ReLU(self.linear1(x))
        h2 = self.ReLU(self.linear2(h1))
        output = self.linear4(h2)
        return(output)

model = MLPexplicit(nInput, nHidden, nOutput)


class nonLinearRegressionData(Dataset):
    def __init__(self, xObs, yObs):
        self.xObs = xObs.float()
        self.yObs = yObs.float()

    def __len__(self):
        return len(self.xObs)

    def __getitem__(self, idx):
        return self.xObs[idx], self.yObs[idx]


# instantiate Dataset object for current training data
d = nonLinearRegressionData(xObs, yObs)

# instantiate DataLoader
#    we use the 4 batches of 25 observations each (full data  has 100 observations)
#    we also shuffle the data
train_dataloader = DataLoader(d, batch_size=20 , shuffle=True)

##################################################
## training the model
##################################################

# Define the loss function and optimizer
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
nTrainSteps = 3000

for epoch in range(nTrainSteps):
    current_loss = 0.0

    for inputs, targets in train_dataloader:
        optimizer.zero_grad()

        outputs = model(inputs)                 # <-- FIXED
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()

        current_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {current_loss:.4f}")

model.eval()
with torch.no_grad():
    yPred = model(xObs)

print("yPred shape:", yPred.shape)
print("First 5 predictions:\n", yPred[:5])
print("First 5 targets:\n", yObs[:5])

yPred_np = yPred.numpy()
yObs_np  = yObs.numpy()

labels = ["Output 1", "Output 2", "Output 3"]

for i in range(3):
    plt.figure()
    plt.scatter(range(len(yObs_np)), yObs_np[:, i], label="True", alpha=0.6)
    plt.scatter(range(len(yPred_np)), yPred_np[:, i], label="Pred", alpha=0.6)
    plt.title(labels[i])
    plt.legend()
    plt.show()


