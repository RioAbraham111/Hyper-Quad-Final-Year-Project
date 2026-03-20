import os
import numpy as np
import pandas as pd

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
X = np.vstack(X_list)

print("Feature matrix shape:", X.shape)

y_data = pd.read_csv("simulation_parameters.csv", )

yObs = y_data.values[:, 1:]  # Assuming the first column is an index or identifier, and the next three columns are the target parameters
