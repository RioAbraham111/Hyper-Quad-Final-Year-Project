import numpy as np
import matplotlib.pyplot as plt

simulation_runs = 1000
# Performance settings
maximum_overshoot = 0.05 # 5% overshoot target
settling_time = 2.0  # seconds
zeta = -np.log(maximum_overshoot) / np.sqrt(np.pi**2 + (np.log(maximum_overshoot))**2)
wn = 4 / (zeta * settling_time)
alpha = 5 * wn  # heuristic (faster third pole)

#System parameters
J = np.random.uniform(0.01, 0.03, simulation_runs)
b = np.random.uniform(0.02, 0.08, simulation_runs)
k = np.random.uniform(0.0, 0.01, simulation_runs)
y_data = np.stack((J, b, k), axis=1)

# Controller gains (PID)
# Kp = J*(wn**2 + 2*zeta*wn*alpha) - k
# Kd = J*(2*zeta*wn + alpha) - b
# Ki = J*(alpha * wn**2)
Kp = np.array([1.5] * simulation_runs)
Ki = np.array([1.7] * simulation_runs)
Kd = np.array([0.3] * simulation_runs)

#desired setpoint
theta_ref = 2.5

x_data = []
for index in range(simulation_runs):
    # initial conditions
    theta = 0
    theta_dot = 0
    prev_error = 0
    integral = 0

    theta_data = []
    theta_dot_data = []
    tau_data = []

    # Simulation settings
    dt = 0.01
    T = 2
    steps = int(T/dt)

    for i in range(steps):
        # PID error
        error = theta_ref - theta
        integral += error * dt
        derivative = (error - prev_error) / dt
        tau = Kp[index] * error + Ki[index] * integral + Kd[index] * derivative

        # Dynamics
        theta_ddot = (tau - b[index] * theta_dot - k[index] * theta) / J[index]
        # Integrate (Euler)
        theta_dot += theta_ddot * dt
        theta += theta_dot * dt
        prev_error = error

        # Store
        theta_data.append(theta)
        theta_dot_data.append(theta_dot)
        tau_data.append(tau)

    sim_data = np.stack((theta_data, theta_dot_data, tau_data), axis=1)
    x_data.append(sim_data)

x_data = np.stack(x_data, axis=2) # shape: (time_steps, features, samples)
np.savez("simulation_data.npz", x_data=x_data, y_data=y_data)