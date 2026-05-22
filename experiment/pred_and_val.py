import simulation.MLP_model as MLP_model
import torch
import numpy as np
import pandas as pd


# -----------------------------
# Load model
# -----------------------------
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
# -----------------------------
# Load experimental data
# -----------------------------
data = np.load("experimental_data_for_prediction.npz", allow_pickle=True)

X = data["x_data"]          # shape: (time_steps, features, samples)
meta_data = data["meta_data"]

time_steps = X.shape[0]
num_features = X.shape[1]
num_samples = X.shape[2]

print("X shape:", X.shape)

# -----------------------------
# Prepare input for MLP
# -----------------------------
X_full = torch.tensor(X, dtype=torch.float32)

# Shape: (samples, time_steps, features)
X_full = X_full.permute(2, 0, 1)

# Flatten: (samples, time_steps * features)
X_full = X_full.reshape(num_samples, time_steps * num_features)

# Normalisation
# Ideally, use training X_mean and X_std.
X_norm = (X_full - X_mean) / X_std

# -----------------------------
# Predict J, b, k
# -----------------------------
with torch.no_grad():
    y_pred = model(X_norm)
y_pred = y_pred * y_std + y_mean 

J_pred = y_pred[:, 0].numpy()
b_pred = y_pred[:, 1].numpy()
k_pred = y_pred[:, 2].numpy()

print("Predicted parameters:")
for i in range(num_samples):
    print(f"Trial {i}: J={J_pred[i]:.6f}, b={b_pred[i]:.6f}, k={k_pred[i]:.6f}")

# -----------------------------
# Save prediction results to CSV
# -----------------------------

# meta_data expected format:
# [trial_id, theta_ref_deg, kp, ki, kd, file_name]

trial_id = meta_data[:, 0]
theta_ref_deg = meta_data[:, 1]
kp = meta_data[:, 2]
ki = meta_data[:, 3]
kd = meta_data[:, 4]
csv_file = meta_data[:, 5]

results_df = pd.DataFrame({
    "csv_file": csv_file,
    "trial_id": trial_id,
    "theta_ref_deg": theta_ref_deg,
    "kp": kp,
    "ki": ki,
    "kd": kd,
    "J_pred": J_pred,
    "b_pred": b_pred,
    "k_pred": k_pred
})

results_df.to_csv("experimental_parameter_predictions.csv", index=False)

print("Saved results to experimental_parameter_predictions.csv")