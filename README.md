# Hyper-Quad Final Year Project

**Integrating Physics-Based Modelling and Machine Learning for System Identification and Control: A Quadcopter Attitude Case Study**

This repository contains the code developed for the Hyper-Quad Monash University Engineering Final Year Project. The project investigates a hybrid approach that combines physics-based modelling, experimental data, and machine learning to identify the dynamic parameters of a quadcopter attitude system.

The completed work focuses primarily on a **1-degree-of-freedom (1-DoF) seesaw test rig**. A **3-DoF quadcopter test platform** was mechanically assembled, but full experimental testing and multi-axis control were not completed within the project period.

## Start Here

New project members should follow this order:

1. Read the final paper and final presentation in the project Google Drive.
2. Review the system architecture, software flowchart, and firmware flowchart.
3. Clone this repository and inspect the 1-DoF firmware and GUI.
4. Reproduce the simulation dataset generation and model-training workflow.
5. Run the 1-DoF system safely and collect one experimental CSV trial.
6. Reproduce one experimental parameter prediction.
7. Only after reproducing the existing 1-DoF workflow, begin improvements or expansion to the 3-DoF platform.

## Project Resources

- **GitHub repository:** https://github.com/RioAbraham111/Hyper-Quad-Final-Year-Project
- **Google Drive:** https://drive.google.com/drive/folders/1C4ajOvmcAlzWm-CTQPlkggHm8bSVhtrT?usp=sharing

The Google Drive contains the final paper, final presentation, figures, CAD assemblies, progress documents, system diagrams, and other large project files that are not suitable for GitHub.

## Project Objective

The 1-DoF plant is represented by:

\[
J\ddot{\theta}+b\dot{\theta}+k\theta=\tau
\]

where:

- \(J\) is the rotational inertia in kg·m²
- \(b\) is the viscous damping coefficient in N·m·s/rad
- \(k\) is the restoring stiffness coefficient in N·m/rad
- \(\theta\) is the angular position
- \(\tau\) is the applied control torque

A multilayer perceptron is trained using simulated closed-loop responses to estimate:

\[
[J,b,k]
\]

from time-series data containing angular position, angular velocity, control input, and reference angle.

## Current Project Status

### Completed

- 1-DoF physics-based dynamic model
- Closed-loop simulation dataset generation
- 1-DoF experimental seesaw test rig
- Arduino-based encoder measurement and PID control
- Python Tkinter GUI for control, monitoring, and CSV logging
- Constant-angle and sine-wave reference modes
- Encoder calibration, median filtering, and low-pass filtering
- MLP architecture, training, and parameter prediction pipeline
- Experimental-data preprocessing
- Preliminary parameter prediction and response-validation workflow
- Mechanical assembly of the 3-DoF platform
- Final paper and final presentation

### Partially Completed

- Simulation-to-experiment transfer
- Quantitative validation across repeated experimental trials
- Reliable torque conversion from motor command to physical torque
- Encoder robustness and long-duration stability
- Parameter-based PID retuning
- 3-DoF sensing, firmware, and controller integration

### Not Completed

- Full 3-DoF experimental testing
- Validated motor-command-to-torque calibration
- Real-time parameter estimation
- Adaptive or automatically tuned control
- Full nonlinear and cross-axis system identification

## Repository Structure

```text
Hyper-Quad-Final-Year-Project/
├── 1DOF/
│   ├── Firmware/
│   │   ├── PID_tuning.ino
│   │   ├── PID_tuning_homing.ino
│   │   └── microcontroller/
│   │       ├── encoder.ino
│   │       └── motor_control.ino
│   └── Software/
│       ├── PID_tuning.py
│       ├── serial_read_test1.py
│       └── test.py
├── simulation/
│   ├── simulation_dataset.py
│   ├── MLP_model.py
│   ├── model_training.py
│   └── model_prediction.py
├── experiment/
│   ├── data_extraction.py
│   └── pred_and_val.py
├── successful_model/
│   ├── attempt#1/
│   └── attempt#2/
├── MLP Example/
├── archieve/
└── README.md
```

## Important Files

| File | Purpose |
|---|---|
| `1DOF/Firmware/PID_tuning_homing.ino` | Integrated 1-DoF firmware with motor control, encoder calibration, filtering, PID control, setpoint and sine-reference commands |
| `1DOF/Software/PID_tuning.py` | Tkinter GUI for serial connection, PID updates, reference commands, live plots and CSV saving |
| `simulation/simulation_dataset.py` | Generates synthetic closed-loop training data from randomly sampled \(J\), \(b\), and \(k\) values |
| `simulation/MLP_model.py` | Defines the multilayer perceptron architecture |
| `simulation/model_training.py` | Normalises the simulation data, trains the MLP and saves the model checkpoint |
| `simulation/model_prediction.py` | Tests parameter prediction on simulated samples |
| `experiment/data_extraction.py` | Converts experimental CSV trials into the model input format |
| `experiment/pred_and_val.py` | Loads the trained model, predicts \(J\), \(b\), and \(k\), and saves the prediction results |
| `successful_model/` | Earlier saved model attempts and simulation datasets; retain for reference but clearly identify the final model before future work |
| `MLP Example/` | Early learning/example material; not the final identification workflow |
| `archieve/` | Older files; this folder should eventually be renamed to `archive/` |

## Hardware Overview

The 1-DoF system uses:

- Arduino Mega
- Incremental quadrature encoder
- Two BLDC motors and propellers
- Two electronic speed controllers
- Carbon-fibre seesaw structure
- Bench power supply
- Computer running the Python GUI

Current firmware pin assignments include:

| Function | Pin |
|---|---:|
| Motor 1 ESC signal | 5 |
| Motor 2 ESC signal | 7 |
| Encoder Channel A | 2 |
| Encoder Channel B | 3 |

Confirm all wiring against the actual rig before applying motor power.

## Safety

This rig uses exposed high-speed propellers. Treat every powered test as hazardous.

- Secure the test rig before connecting motor power.
- Remove propellers for software-only testing.
- Inspect propellers and mechanical fasteners before each test.
- Keep hands, cables, clothing, and loose objects outside the propeller area.
- Use a current-limited bench power supply where possible.
- Keep the GUI `STOP` command immediately accessible.
- Begin with conservative PID gains and small reference angles.
- Do not operate the platform alone when testing major firmware or control changes.
- Disconnect motor power before changing wiring or mechanical components.

## Software Requirements

Recommended tools:

- Python 3.x
- Arduino IDE
- Git

Core Python packages:

```bash
pip install numpy pandas matplotlib torch pyserial
```

`tkinter` is commonly included with Python. On some Linux systems it must be installed separately.

A clean virtual environment is recommended:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

## Clone the Repository

```bash
git clone https://github.com/RioAbraham111/Hyper-Quad-Final-Year-Project.git
cd Hyper-Quad-Final-Year-Project
```

## 1-DoF Experimental Workflow

### 1. Upload the firmware

Open the integrated Arduino firmware:

```text
1DOF/Firmware/PID_tuning_homing.ino
```

Before uploading, verify:

- Arduino board and serial port
- Motor signal pins
- Encoder pins
- Encoder calibration counts
- Motor directions
- Safe motor limits

### 2. Start the GUI

From the repository root:

```bash
python "1DOF/Software/PID_tuning.py"
```

The GUI supports:

- Serial-port selection
- PID gain updates
- Constant-angle references
- Sine-wave references
- Start, stop, idle and encoder-calibration commands
- Live plots of controller output, angle, and angular velocity
- CSV saving

The serial baud rate is currently `115200`.

### 3. Calibrate the encoder

The firmware uses a two-position calibration process when the `ZERO` command is sent:

1. Move the platform to the minimum calibration angle and press `ZERO`.
2. Move the platform to the maximum calibration angle and press `ZERO` again.
3. Confirm that the GUI reports calibration completion.

Check the physical calibration convention before operating the motors. The current firmware maps the calibration endpoints to approximately \(-51^\circ\) and \(+51^\circ\), while reference inputs are constrained to \(-45^\circ\) to \(+45^\circ\).

### 4. Run and save a trial

1. Set conservative PID gains.
2. Enter a safe reference angle or sine input.
3. Start the trial.
4. Stop immediately if the rig oscillates excessively or approaches a mechanical limit.
5. Save the CSV file.

Typical CSV columns are:

```text
trial_id,time_ms,theta,theta_dot,control_output,kp,ki,kd
```

The GUI generates filenames containing the reference input, for example:

```text
pid_trial_angle_-20deg_YYYYMMDD_HHMMSS.csv
```

## Simulation and MLP Workflow

### Simulation configuration

The current simulation script uses approximately:

- 5000 simulation runs
- 10 seconds per run
- 0.005 second timestep
- 2000 samples per run
- Four features: \(\theta\), \(\dot{\theta}\), \(\tau\), and \(\theta_{ref}\)

Current parameter ranges are:

```text
J: 0.0025 to 0.008 kg·m²
b: 0.001 to 0.03 N·m·s/rad
k: 0.005 to 0.08 N·m/rad
```

### 1. Generate the simulation dataset

The scripts currently use relative paths. The simplest existing workflow is:

```bash
cd simulation
python simulation_dataset.py
```

This creates:

```text
simulation_data_for_exp.npz
```

### 2. Train the model

From the same `simulation` directory:

```bash
python model_training.py
```

This creates:

```text
system_identifier_ML.pth
```

The saved checkpoint contains:

- Model weights
- Input mean and standard deviation
- Output mean and standard deviation
- Input, hidden, and output sizes

The normalisation values must always be loaded with the model.

## Experimental Prediction Workflow

### 1. Prepare experimental CSV files

Create an `experiment_data` folder and place the raw experimental CSV files inside it. Filenames must contain the reference angle in this format:

```text
pid_trial_angle_-20deg_YYYYMMDD_HHMMSS.csv
```

### 2. Extract model-ready experimental data

The current script uses paths relative to the working directory. Review the path variables at the top of `experiment/data_extraction.py` before running it.

```bash
python experiment/data_extraction.py
```

Expected output:

```text
experimental_data_for_prediction.npz
```

### 3. Run parameter prediction

`experiment/pred_and_val.py` expects the following files in its active working directory:

```text
system_identifier_ML.pth
experimental_data_for_prediction.npz
```

Run it from the repository root after checking the paths:

```bash
python -m experiment.pred_and_val
```

Expected output:

```text
experimental_parameter_predictions.csv
```

## Critical Limitations and Known Issues

### Motor command is not a validated torque measurement

The experimental `control_output` is a differential motor command, not a directly measured torque in N·m. The conversion in `experiment/data_extraction.py` is an estimate and must not be treated as validated ground truth.

Future work should experimentally determine a relationship such as:

\[
\tau=f(u)
\]

or, preferably:

\[
\tau=f(u,\omega,V)
\]

where \(u\) is motor command, \(\omega\) is motor or propeller speed, and \(V\) is supply voltage.

### Simulation-to-reality mismatch

The simulation does not fully represent:

- ESC and motor dynamics
- Propeller aerodynamics
- Motor dead zone
- Static friction
- Nonlinear damping
- Structural flexibility
- Sensor noise and quantisation
- Control saturation
- Electrical and serial delays

### Encoder reliability

The encoder previously produced occasional jumps and offsets. The current firmware includes valid-transition checking, a median filter, low-pass filtering, and two-point calibration, but further validation is required.

### Sampling-rate comment mismatch

The firmware uses:

```cpp
const unsigned long SAMPLE_TIME_MS = 5;
```

A 5 ms period corresponds to **200 Hz**, although an inline source-code comment currently says 100 Hz. The comment should be corrected.

### Relative file paths

Several scripts depend on the current working directory and manually placed `.npz` or `.pth` files. Future maintainers should refactor these paths using `pathlib` and paths relative to each script or the repository root.

### Model and dataset naming

The `successful_model` folder contains multiple attempts named `attempt#1` and `attempt#2`, both with generically named files. Before further work, identify the final validated checkpoint and rename it clearly, for example:

```text
system_identifier_ML_final.pth
simulation_dataset_final.npz
```

## Recommended Future Work

Recommended priority order:

1. Reproduce one complete 1-DoF experimental trial.
2. Reproduce the simulation dataset and MLP training workflow.
3. Reproduce one parameter prediction from experimental data.
4. Calibrate motor command against physical thrust and torque.
5. Improve encoder reliability and verify long-duration repeatability.
6. Quantify model validation using RMSE, MAE, parameter variation, and reconstructed-response comparisons.
7. Improve the physics model using a learned residual term:

   \[
   \tau=J\ddot{\theta}+b\dot{\theta}+k\theta+\tau_{residual}
   \]

8. Use the identified model to support PID retuning.
9. Integrate sensing and control on the 3-DoF platform.
10. Investigate axis coupling and multi-axis system identification.

## Handover Acceptance Test

The handover is successful when a new project member can independently:

- Access the GitHub repository and Google Drive
- Identify the final paper, final presentation, CAD files, and system diagrams
- Upload the 1-DoF firmware
- Start the Python GUI
- Calibrate and read the encoder
- Run one safe 1-DoF trial
- Save and interpret a CSV file
- Generate a simulation dataset
- Train or load the MLP
- Predict \(J\), \(b\), and \(k\) from one trial
- Explain the torque-calibration and simulation-to-reality limitations
- Identify the next recommended development task

## Google Drive Guide

The Drive is currently divided into:

```text
HyperQuad FYP/
├── FYP S2 2025/
└── FYP S1 2026/
```

Important locations include:

| Location | Contents |
|---|---|
| `FYP S2 2025/1DOF_assembly.zip` | 1-DoF CAD assembly |
| `FYP S2 2025/3DofTestBenchWDrone.zip` | 3-DoF test-platform CAD assembly |
| `FYP S2 2025/System Interaction Diagram.drawio` | Early system interaction diagram |
| `FYP S1 2026/Final Paper` | Final submitted paper |
| `FYP S1 2026/Final Presentation` | Final presentation |
| `FYP S1 2026/Figures` | Final figures and illustrations |
| `FYP S1 2026/Firmware Flowchart.drawio` | Embedded-system logic |
| `FYP S1 2026/Software Flowchart.drawio` | Software workflow |
| `FYP S1 2026/System Architecture.drawio` | Overall project architecture |
| `FYP S1 2026/FYP S1 2026 Master` | Project planning and progress record |

Some files are owned by other group members. Important CAD and diagram files should be copied or transferred to a permanent project-controlled location before student accounts or permissions change.

## Recommended Repository Cleanup

Before final handover:

- Add a `.gitignore`
- Remove `__pycache__/`
- Remove `.DS_Store`
- Remove unnecessary `.vscode` user settings
- Rename `archieve/` to `archive/`
- Add a `requirements.txt`
- Identify and clearly rename the final model and dataset
- Move obsolete files into `archive/`
- Add brief README files inside `1DOF/`, `simulation/`, and `experiment/`
- Create a tagged release such as `v1.0-final-handover`

## Contributors

This project was completed as part of the Monash University Engineering Final Year Project under the supervision of **Dr Mohamed Tolba**.

Project contributors and ownership details should be completed by the project team before final handover.

## Academic Use

This repository is an academic project archive and research handover resource. Future users should cite the final paper and acknowledge the original project team when reusing the methodology, code, figures, datasets, or hardware designs.
