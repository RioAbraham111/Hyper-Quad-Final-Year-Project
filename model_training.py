import torch
import torch.nn as nn
import torch.optim as optim
import MLP_model
import numpy as np
import matplotlib.pyplot as plt

# this is where we have to get the data set from the simulation.py file
data = np.load("successful_model/attempt#1/simulation_data.npz")
X = data['x_data']
y = data['y_data']

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

X_train = X.permute(2, 0, 1).reshape(1000, 600)
X_train = (X_train - X_train.mean(dim=0)) / (X_train.std(dim=0) + 1e-8)

# Initialize the model
input_size = 600
hidden_size = 128
output_size = 3
model = MLP_model.MLP(input_size, hidden_size, output_size)

losses = model.fit(X_train, y_train, loss_function=nn.MSELoss(), optimizer=optim.Adam(model.parameters(), lr=0.01), nTrainSteps=1000)
torch.save(model.state_dict(), "mlp_model.pth")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid()
plt.show()

