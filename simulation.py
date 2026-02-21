import numpy as np
import csv
import itertools
import os

folder_name = "seesaw_simulations"
os.makedirs(folder_name, exist_ok=True)

# -----------------------------
# Parameter Sets (Logical Values)
# -----------------------------
J_values = [0.01, 0.05, 0.1]
b_values = [0.05, 0.2, 0.5]
k_values = [0.5, 1.0, 2.0]
tau_limits = [2, 5, 10]

# PID Gains (fixed for comparison)
Kp = 30
Ki = 10
Kd = 5

theta_ref = 0.5

# Simulation settings
dt = 0.001
T = 5
steps = int(T/dt)

param_file = os.path.join(folder_name, "simulation_parameters.csv")

with open(param_file, "w", newline="") as pfile:
    param_writer = csv.writer(pfile)
    param_writer.writerow([
        "simulation_file",
        "J",
        "b",
        "k",
        "tau_limit"
    ])

    sim_number = 1

    # Loop through all parameter combinations
    for J, b, k, tau_max in itertools.product(
        J_values, b_values, k_values, tau_limits
    ):

        # Initial conditions
        theta = 0
        theta_dot = 0
        integral = 0
        prev_error = 0

        # Storage
        time_data = []
        theta_data = []
        theta_dot_data = []
        tau_data = []

        for i in range(steps):
            t = i * dt

            # PID error
            error = theta_ref - theta
            integral += error * dt
            derivative = (error - prev_error) / dt

            tau = Kp * error + Ki * integral + Kd * derivative

            # Apply torque saturation
            tau = max(min(tau, tau_max), -tau_max)

            # Dynamics
            theta_ddot = (tau - b * theta_dot - k * theta) / J

            # Integrate (Euler)
            theta_dot += theta_ddot * dt
            theta += theta_dot * dt

            prev_error = error

            # Store
            time_data.append(t)
            theta_data.append(theta)
            theta_dot_data.append(theta_dot)
            tau_data.append(tau)

        # Save simulation CSV
        sim_filename = f"sim_{sim_number}.csv"
        sim_path = os.path.join(folder_name, sim_filename)

        with open(sim_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "theta", "theta_dot", "tau"])
            for i in range(len(time_data)):
                writer.writerow([
                    time_data[i],
                    theta_data[i],
                    theta_dot_data[i],
                    tau_data[i]
                ])

        # Save parameter mapping
        param_writer.writerow([
            sim_filename,
            J,
            b,
            k,
            tau_max
        ])

        print(f"Saved {sim_filename}")
        sim_number += 1

print("All simulations complete.")
