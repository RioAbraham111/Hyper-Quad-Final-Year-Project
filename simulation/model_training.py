import torch
import torch.nn as nn
import torch.optim as optim
import MLP_model as MLP_model
import numpy as np
import matplotlib.pyplot as plt

# this is where we have to get the data set from the simulation.py file
data = np.load("simulation_data_for_exp.npz")
X = data['x_data']
y = data['y_data']

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

X_train = X.permute(2, 0, 1).reshape(5000, 8000)

X_mean = X_train.mean(dim=0)
X_std = X_train.std(dim=0) + 1e-8

X_train_norm = (X_train - X_mean) / X_std

y_mean = y_train.mean(dim=0)
y_std = y_train.std(dim=0) + 1e-8

y_train_norm = (y_train - y_mean) / y_std

# Initialize the model
input_size = 8000
hidden_size = 128
output_size = 3
model = MLP_model.MLP(input_size, hidden_size, output_size)

losses = model.fit(X_train_norm, y_train_norm, loss_function=nn.MSELoss(), optimizer=optim.Adam(model.parameters(), lr=0.01), nTrainSteps=2000)
torch.save({
    "model_state_dict": model.state_dict(),
    "X_mean": X_mean,
    "X_std": X_std,
    "y_mean": y_mean,
    "y_std": y_std,
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size
}, "system_identifier_ML.pth")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid()
plt.show()

