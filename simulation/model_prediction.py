import MLP_model as MLP_model
import torch
import numpy as np
import matplotlib.pyplot as plt

# load model
checkpoint = torch.load("system_identifier_ML.pth", map_location="cpu")

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]

model = MLP_model.MLP(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

X_mean = checkpoint["X_mean"]
X_std = checkpoint["X_std"]
y_mean = checkpoint["y_mean"]
y_std = checkpoint["y_std"]

# load data
data = np.load("simulation_data_for_exp.npz")
X = data['x_data']
y = data['y_data']

X_full = torch.tensor(X, dtype=torch.float32)
X_full = X_full.permute(2, 0, 1).reshape(5000, 8000)
y_full = torch.tensor(y, dtype=torch.float32)
    
X_norm = (X_full - X_mean) / X_std

with torch.no_grad():
    y_pred = model(X_norm)
    y_pred = y_pred * y_std + y_mean 

# Convert to numpy
y_actual_np = y_full.numpy()
y_pred_np = y_pred.numpy()

# -----------------------------
# Regression metrics
# -----------------------------
labels = ["J", "b", "k"]

mae = np.mean(np.abs(y_pred_np - y_actual_np), axis=0)
rmse = np.sqrt(np.mean((y_pred_np - y_actual_np) ** 2, axis=0))
mape = np.mean(np.abs((y_pred_np - y_actual_np) / y_actual_np), axis=0) * 100

# R2 score
ss_res = np.sum((y_actual_np - y_pred_np) ** 2, axis=0)
ss_tot = np.sum((y_actual_np - np.mean(y_actual_np, axis=0)) ** 2, axis=0)
r2 = 1 - ss_res / ss_tot

# -----------------------------
# Print results
# -----------------------------
print("Model Evaluation on Simulation Data")
print("-----------------------------------")

for i, label in enumerate(labels):
    print(f"{label}:")
    print(f"  MAE  = {mae[i]:.6f}")
    print(f"  RMSE = {rmse[i]:.6f}")
    print(f"  MAPE = {mape[i]:.2f}%")
    print(f"  R²   = {r2[i]:.4f}")
    print()

print("Average performance:")
print(f"  Average MAE  = {np.mean(mae):.6f}")
print(f"  Average RMSE = {np.mean(rmse):.6f}")
print(f"  Average MAPE = {np.mean(mape):.2f}%")
print(f"  Average R²   = {np.mean(r2):.4f}")

errors = y_pred_np - y_actual_np

plt.figure(figsize=(12, 4))

for i, label in enumerate(labels):
    plt.subplot(1, 3, i + 1)
    plt.hist(errors[:, i], bins=30, alpha=0.7)
    plt.xlabel(f"Prediction Error in {label}")
    plt.ylabel("Count")
    plt.title(f"{label} Error Distribution")
    plt.grid(True)

plt.figure(figsize=(12, 4))

for i, label in enumerate(labels):
    plt.subplot(1, 3, i + 1)
    plt.scatter(y_actual_np[:, i], y_pred_np[:, i], alpha=0.5)

    min_val = min(y_actual_np[:, i].min(), y_pred_np[:, i].min())
    max_val = max(y_actual_np[:, i].max(), y_pred_np[:, i].max())

    plt.plot([min_val, max_val], [min_val, max_val], "k--")
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"{label}: Predicted vs Actual")
    plt.grid(True)


plt.tight_layout()
plt.show()




