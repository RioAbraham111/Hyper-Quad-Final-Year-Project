import torch
import torch.nn as nn
import torch.optim as optim
import MLP_model
import numpy as np
import matplotlib.pyplot as plt

# this is where we have to get the data set from the simulation.py file
data = np.load("simulation_data.npz")
X_train = data['x_data']
y_train = data['y_data']

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# Initialize the model
input_size = 200
hidden_size = 50
output_size = 3
model = MLP_model.MLP(input_size, hidden_size, output_size)

losses = model.train(X_train, y_train, loss_function=nn.MSELoss(), optimizer=optim.Adam(model.parameters(), lr=0.01), nTrainSteps=1000)
torch.save(model.state_dict(), "mlp_model.pth")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.show()

