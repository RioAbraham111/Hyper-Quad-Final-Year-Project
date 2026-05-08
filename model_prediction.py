import MLP_model
import torch
import numpy as np
import matplotlib.pyplot as plt

# load model
model = MLP_model.MLP(600, 128, 3)
model.load_state_dict(torch.load("successful_model/attempt#1/mlp_model.pth"))
model.eval()

# load data
data = np.load("successful_model/attempt#1/simulation_data.npz")
X = data['x_data']   # (200,3,250)
y = data['y_data']   # (250,3)

X_full = torch.tensor(X, dtype=torch.float32)
X_full = X_full.permute(2, 0, 1).reshape(1000, 600)

X_mean = X_full.mean(dim=0)
X_std = X_full.std(dim=0) + 1e-8

ypred = []
yactual = []
# test few samples
for index in range(23, 30):

    theta = X[:, 0, index]
    theta_dot = X[:, 1, index]
    tau = X[:, 2, index]

    y_pred = model.predict(theta, theta_dot, tau, X_mean, X_std)

    ypred.append(y_pred)
    yactual.append(y[index])

ypred = torch.stack(ypred).squeeze().numpy()
yactual = torch.tensor(yactual).numpy()

# Plotting
plt.figure(figsize=(12, 6))
labels = ['J', 'b', 'k']
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.scatter(range(23, 30), yactual[:, i], color='blue', label='Actual')
    plt.scatter(range(23, 30), ypred[:, i], color='red', label='Predicted')
    plt.title(labels[i] + ': Predicted vs Actual')
    plt.legend()
    plt.grid()
plt.show()




