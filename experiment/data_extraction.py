import os
import re
import numpy as np
import pandas as pd

# Folder containing your experimental CSV files
csv_folder = "experiment_data"

# Output file
output_file = "experimental_data_for_prediction.npz"

# Number of samples per trial
num_samples = 2000

# Torque conversion
# Use your estimated/calibrated maximum torque
tau_max = 0.15  # Nm

x_data = []
meta_data = []

csv_files = sorted([
    f for f in os.listdir(csv_folder)
    if f.endswith(".csv")
])

for file_name in csv_files:
    file_path = os.path.join(csv_folder, file_name)

    # --------------------------------------------------
    # Extract theta reference from filename
    # Example: pid_trial_angle_0deg_20260521_153000.csv
    # Example: pid_trial_angle_-20deg_20260521_153000.csv
    # --------------------------------------------------
    match = re.search(r"angle_(-?\d+(?:\.\d+)?)deg", file_name)

    if match is None:
        print(f"Skipping {file_name}: could not find angle in filename")
        continue

    theta_ref_deg = float(match.group(1))
    theta_ref_rad = np.deg2rad(theta_ref_deg)

    # Read CSV
    df = pd.read_csv(file_path)

    # Take first 2000 data rows
    df = df.iloc[:num_samples]

    if len(df) < num_samples:
        print(f"Skipping {file_name}: only {len(df)} rows")
        continue

    # Extract experimental signals
    theta_deg = df["theta"].to_numpy(dtype=float)
    theta_dot_deg = df["theta_dot"].to_numpy(dtype=float)
    control_output = df["control_output"].to_numpy(dtype=float)

    # Convert angle signals to radians
    theta_rad = np.deg2rad(theta_deg)
    theta_dot_rad = np.deg2rad(theta_dot_deg)

    # Convert control output percentage to estimated torque [Nm]
    maximum_thrust = 9.842
    l1 = 0.27575
    l2 = 0.27575
    tau = (12 + control_output) * maximum_thrust * l1 - (12 - control_output) * maximum_thrust * l2

    # Create theta_ref signal with same length as theta
    theta_ref_data = np.ones_like(theta_rad) * theta_ref_rad

    # Stack features in the same order used during training
    # Features: theta, theta_dot, tau, theta_ref
    trial_data = np.stack(
        (theta_rad, theta_dot_rad, tau, theta_ref_data),
        axis=1
    )

    x_data.append(trial_data)

    # Save metadata for checking
    trial_id = df["trial_id"].iloc[0]
    kp = df["kp"].iloc[0]
    ki = df["ki"].iloc[0]
    kd = df["kd"].iloc[0]

    meta_data.append([
        trial_id,
        theta_ref_deg,
        kp,
        ki,
        kd,
        file_name
    ])

# Convert to numpy array
# Current shape: (samples, time_steps, features)
x_data = np.array(x_data, dtype=np.float32)

# Convert to model/simulation format:
# shape: (time_steps, features, samples)
x_data = np.transpose(x_data, (1, 2, 0))

meta_data = np.array(meta_data, dtype=object)

np.savez(
    output_file,
    x_data=x_data,
    meta_data=meta_data
)

print("Saved:", output_file)
print("x_data shape:", x_data.shape)
print("meta_data shape:", meta_data.shape)
print("Features: theta_rad, theta_dot_rad, tau_Nm, theta_ref_rad")