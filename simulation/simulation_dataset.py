import numpy as np
import matplotlib.pyplot as plt

simulation_runs = 5000

#System parameters
J = np.random.uniform(0.0025, 0.008, simulation_runs)
b = np.random.uniform(0.001, 0.03, simulation_runs)
k = np.random.uniform(0.005, 0.08, simulation_runs)
# J = np.random.uniform(0.0005, 0.08, simulation_runs)
# b = np.random.uniform(0.0001, 0.1, simulation_runs)
# k = np.random.uniform(0.0005, 0.1, simulation_runs)
y_data = np.stack((J, b, k), axis=1)

Kp = 0.0015
Ki = 0.0002
Kd = 0.0005

x_data = []

progress_bar = 0
for index in range(simulation_runs):
    if index % (simulation_runs//10) == 0:
        print(f"Progress: {progress_bar}%")
        progress_bar += 10

    # initial conditions
    if index % 2 == 0:
        theta = -50 * np.pi / 180
    else:
        theta = 0

    if index % 3 == 0:
        multi_step = True
    elif index % 5 == 0:
        multi_step = False
        sine_wave = True
        sine_wave_array = np.sin(np.linspace(0, 2*np.pi*int(np.random.uniform(3, 10)), 2000)) * (40 * np.pi / 180)
    else:
        multi_step = False
        sine_wave = False

    theta_ref = np.random.uniform(-40, 40) * np.pi / 180

    theta_dot = 0
    prev_error = theta_ref - theta
    integral = 0

    theta_data = []
    theta_dot_data = []
    tau_data = []
    theta_ref_data = []

    # Simulation settings
    dt = 0.005
    time_span = 10
    steps = int(time_span/dt)

    for i in range(steps):
        if multi_step:
            if i*dt < int(np.random.uniform(2, 4)):
                theta_ref = np.random.uniform(-40, 40) * np.pi / 180
            elif i*dt < int(np.random.uniform(4, 6)):
                theta_ref = np.random.uniform(-40, 40) * np.pi / 180
            else:
                theta_ref = np.random.uniform(-40, 40) * np.pi / 180

        elif sine_wave:
            theta_ref = sine_wave_array[i]

        # PID error
        error = theta_ref - theta
        integral += error * dt
        derivative = (error - prev_error) / dt
        tau = Kp * error + Ki * integral + Kd * derivative

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
        theta_ref_data.append(theta_ref)

    sim_data = np.stack((theta_data, theta_dot_data, tau_data, theta_ref_data), axis=1)
    x_data.append(sim_data)

x_data = np.stack(x_data, axis=2) # shape: (time_steps, features, samples)
np.savez("simulation_data_for_exp.npz", x_data=x_data, y_data=y_data)

print("Simulation complete. Data saved to 'simulation_data_for_exp.npz'")