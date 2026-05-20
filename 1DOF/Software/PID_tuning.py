import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import serial
import serial.tools.list_ports as serial_ports
import csv
from datetime import datetime
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PIDMonitorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("1-DOF Drone Seesaw PID Monitor")
        self.root.geometry("1100x700")

        # Serial
        self.ser = None
        self.connected = False

        # Data storage
        self.max_display_points = 2000
        self.time_data = deque(maxlen=self.max_display_points)
        self.theta_data = deque(maxlen=self.max_display_points)
        self.control_data = deque(maxlen=self.max_display_points)
        self.theta_dot_data = deque(maxlen=self.max_display_points)

        self.all_rows = []

        self.trial_active = False
        self.current_trial_id = 0

        # GUI variables
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="115200")

        self.kp_var = tk.StringVar(value="0.0")
        self.ki_var = tk.StringVar(value="0.0")
        self.kd_var = tk.StringVar(value="0.0")

        # Desired angle / reference variables
        self.setpoint_var = tk.StringVar(value="0.0")

        self.sine_offset_var = tk.StringVar(value="0.0")
        self.sine_amplitude_var = tk.StringVar(value="10.0")
        self.sine_frequency_var = tk.StringVar(value="0.1")

        self.status_var = tk.StringVar(value="Disconnected")
        self.latest_theta_var = tk.StringVar(value="Theta: ---")
        self.latest_control_var = tk.StringVar(value="Control: ---")
        self.latest_pid_var = tk.StringVar(value="Kp: ---   Ki: ---   Kd: ---")
        self.latest_theta_dot_var = tk.StringVar(value="Theta dot: ---")

        self.reference_filename_label = "angle_0deg"

        self.create_widgets()
        self.refresh_ports()

        # Start non-threaded update loop
        self.root.after(10, self.update_loop)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---------------- Serial section ----------------
        ttk.Label(control_frame, text="Serial Port").pack(anchor="w")

        port_row = ttk.Frame(control_frame)
        port_row.pack(fill=tk.X, pady=5)

        self.port_box = ttk.Combobox(
            port_row,
            textvariable=self.port_var,
            width=22,
            state="readonly"
        )
        self.port_box.pack(side=tk.LEFT)

        ttk.Button(
            port_row,
            text="Refresh",
            command=self.refresh_ports
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Baud Rate").pack(anchor="w", pady=(10, 0))
        ttk.Entry(control_frame, textvariable=self.baud_var, width=15).pack(anchor="w", pady=5)

        ttk.Button(
            control_frame,
            text="Connect",
            command=self.connect_serial
        ).pack(fill=tk.X, pady=5)

        ttk.Button(
            control_frame,
            text="Disconnect",
            command=self.disconnect_serial
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=10)

        # ---------------- PID section ----------------
        # ---------------- PID section ----------------
        ttk.Label(control_frame, text="PID Gains").pack(anchor="w")

        pid_frame = ttk.Frame(control_frame)
        pid_frame.pack(fill=tk.X, pady=5)

        # Left side: labels + entries
        ttk.Label(pid_frame, text="Kp").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(pid_frame, textvariable=self.kp_var, width=8).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(pid_frame, text="Ki").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(pid_frame, textvariable=self.ki_var, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(pid_frame, text="Kd").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(pid_frame, textvariable=self.kd_var, width=8).grid(row=2, column=1, padx=5, pady=2)

        # Right side: button
        ttk.Button(
            pid_frame,
            text="Update PID",
            command=self.update_pid,
            width=14
        ).grid(row=0, column=2, rowspan=3, padx=8, pady=2, sticky="ns")

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        # ---------------- Constant setpoint section ----------------
        ttk.Label(control_frame, text="Desired Angle").pack(anchor="w")

        setpoint_frame = ttk.Frame(control_frame)
        setpoint_frame.pack(fill=tk.X, pady=5)

        ttk.Label(setpoint_frame, text="Angle").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(setpoint_frame, textvariable=self.setpoint_var, width=8).grid(row=0, column=1, padx=5, pady=2)

        ttk.Button(
            setpoint_frame,
            text="Set Angle",
            command=self.set_constant_angle,
            width=14
        ).grid(row=0, column=2, padx=8, pady=2)

        # ---------------- Sine reference section ----------------
        ttk.Label(control_frame, text="Sine Reference").pack(anchor="w", pady=(5, 0))

        sine_frame = ttk.Frame(control_frame)
        sine_frame.pack(fill=tk.X, pady=5)

        ttk.Label(sine_frame, text="Offset").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(sine_frame, textvariable=self.sine_offset_var, width=8).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(sine_frame, text="Amp").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(sine_frame, textvariable=self.sine_amplitude_var, width=8).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(sine_frame, text="Freq").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(sine_frame, textvariable=self.sine_frequency_var, width=8).grid(row=2, column=1, padx=5, pady=2)

        ttk.Button(
            sine_frame,
            text="Set Sine",
            command=self.set_sine_angle,
            width=14
        ).grid(row=0, column=2, rowspan=3, padx=8, pady=2, sticky="ns")

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        # ---------------- Trial buttons ----------------
        trial_button_frame = ttk.Frame(control_frame)
        trial_button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            trial_button_frame,
            text="START",
            command=self.start_trial,
            width=12
        ).grid(row=0, column=0, padx=3, pady=3, sticky="ew")

        ttk.Button(
            trial_button_frame,
            text="STOP",
            command=self.stop_trial,
            width=12
        ).grid(row=0, column=1, padx=3, pady=3, sticky="ew")

        ttk.Button(
            trial_button_frame,
            text="IDLE",
            command=self.idle_motor,
            width=12
        ).grid(row=1, column=0, padx=3, pady=3, sticky="ew")

        ttk.Button(
            trial_button_frame,
            text="ZERO",
            command=self.zero_encoder,
            width=12
        ).grid(row=1, column=1, padx=3, pady=3, sticky="ew")

        ttk.Button(
            trial_button_frame,
            text="STATUS",
            command=self.request_status,
            width=12
        ).grid(row=2, column=0, columnspan=2, padx=3, pady=3, sticky="ew")

        trial_button_frame.columnconfigure(0, weight=1)
        trial_button_frame.columnconfigure(1, weight=1)

        ttk.Separator(control_frame).pack(fill=tk.X, pady=8)

        # ---------------- Data buttons ----------------
        data_button_frame = ttk.Frame(control_frame)
        data_button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            data_button_frame,
            text="Save CSV",
            command=self.save_csv,
            width=12
        ).grid(row=0, column=0, padx=3, pady=3, sticky="ew")

        ttk.Button(
            data_button_frame,
            text="Clear Plot",
            command=self.clear_plot,
            width=12
        ).grid(row=0, column=1, padx=3, pady=3, sticky="ew")

        data_button_frame.columnconfigure(0, weight=1)
        data_button_frame.columnconfigure(1, weight=1)

        # ---------------- Status labels ----------------
        ttk.Label(control_frame, textvariable=self.status_var).pack(anchor="w", pady=3)
        ttk.Label(control_frame, textvariable=self.latest_theta_var).pack(anchor="w", pady=3)
        ttk.Label(control_frame, textvariable=self.latest_theta_dot_var).pack(anchor="w", pady=3)
        ttk.Label(control_frame, textvariable=self.latest_control_var).pack(anchor="w", pady=3)
        ttk.Label(control_frame, textvariable=self.latest_pid_var).pack(anchor="w", pady=3)


        # ---------------- Serial monitor ----------------
        ttk.Label(control_frame, text="Serial Messages").pack(anchor="w", pady=(10, 0))

        self.serial_text = tk.Text(control_frame, height=12, width=40)
        self.serial_text.pack(fill=tk.BOTH, expand=True)

        # ---------------- Matplotlib plot ----------------
        self.fig, (self.ax_control, self.ax_theta, self.ax_theta_dot) = plt.subplots(
            3, 1,
            figsize=(7, 7),
            sharex=True
        )

        TITLE_FONT = 6
        LABEL_FONT = 5
        TICK_FONT = 5
        LEGEND_FONT = 5

        # Plot 1: controller output
        self.line_control, = self.ax_control.plot([], [], label="Control Output")
        self.ax_control.set_ylabel("Control", fontsize=LABEL_FONT)
        self.ax_control.set_title("Controller Output", fontsize=TITLE_FONT)
        self.ax_control.grid(True)
        self.ax_control.legend(loc="upper right", fontsize=LEGEND_FONT)
        self.ax_control.tick_params(axis="both", labelsize=TICK_FONT)

        # Plot 2: theta
        self.line_theta, = self.ax_theta.plot([], [], label="Theta")
        self.ax_theta.set_ylim(-55, 55)
        self.ax_theta.set_ylabel("Theta [deg]", fontsize=LABEL_FONT)
        self.ax_theta.set_title("Theta", fontsize=TITLE_FONT)
        self.ax_theta.grid(True)
        self.ax_theta.legend(loc="upper right", fontsize=LEGEND_FONT)
        self.ax_theta.tick_params(axis="both", labelsize=TICK_FONT)

        # Plot 3: theta dot
        self.line_theta_dot, = self.ax_theta_dot.plot([], [], label="Theta Dot")
        self.ax_theta_dot.set_xlabel("Time [s]", fontsize=LABEL_FONT)
        self.ax_theta_dot.set_ylabel("Theta Dot [deg/s]", fontsize=LABEL_FONT)
        self.ax_theta_dot.set_title("Theta Dot", fontsize=TITLE_FONT)
        self.ax_theta_dot.grid(True)
        self.ax_theta_dot.legend(loc="upper right", fontsize=LEGEND_FONT)
        self.ax_theta_dot.tick_params(axis="both", labelsize=TICK_FONT)

        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def refresh_ports(self):
        ports = serial.tools.list_ports.comports()
        port_names = [port.device for port in ports]

        self.port_box["values"] = port_names

        if port_names:
            self.port_var.set(port_names[0])
        else:
            self.port_var.set("")

    def connect_serial(self):
        if self.connected:
            messagebox.showinfo("Already connected", "Serial is already connected.")
            return

        port = self.port_var.get()

        if not port:
            messagebox.showerror("No port selected", "Please select a serial port.")
            return

        try:
            baud = int(self.baud_var.get())

            self.ser = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=0
            )

            self.connected = True
            self.status_var.set(f"Connected to {port} at {baud} baud")
            self.log_message(f"Connected to {port}")

        except Exception as e:
            messagebox.showerror("Connection error", str(e))
            self.status_var.set("Connection failed")

    def disconnect_serial(self):
        if self.ser is not None:
            try:
                if self.connected:
                    self.send_command("STOP")

                self.ser.close()
            except Exception:
                pass

        self.ser = None
        self.connected = False
        self.trial_active = False

        self.status_var.set("Disconnected")
        self.log_message("Disconnected")


    def send_command(self, command):
        if not self.connected or self.ser is None:
            self.log_message("ERROR: Not connected to Arduino")
            return

        try:
            self.ser.write((command + "\n").encode("utf-8"))
            self.log_message(f"Python -> Arduino: {command}")
        except Exception as e:
            self.log_message(f"Serial write error: {e}")

    def update_pid(self):
        try:
            kp = float(self.kp_var.get())
            ki = float(self.ki_var.get())
            kd = float(self.kd_var.get())
        except ValueError:
            messagebox.showerror("Invalid PID", "Kp, Ki, and Kd must be numbers.")
            return

        self.send_command(f"SET {kp} {ki} {kd}")

    def set_constant_angle(self):
        try:
            angle = float(self.setpoint_var.get())
        except ValueError:
            messagebox.showerror("Invalid angle", "Desired angle must be a number.")
            return

        if angle < -45.0 or angle > 45.0:
            messagebox.showerror(
                "Invalid angle",
                "Desired angle must be between -45 and 45 degrees."
            )
            return
        
        self.reference_filename_label = f"angle_{angle:g}deg"

        self.send_command(f"ANGLE {angle}")


    def set_sine_angle(self):
        try:
            offset = float(self.sine_offset_var.get())
            amplitude = float(self.sine_amplitude_var.get())
            frequency = float(self.sine_frequency_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid sine input",
                "Offset, amplitude, and frequency must be numbers."
            )
            return

        if frequency <= 0:
            messagebox.showerror(
                "Invalid frequency",
                "Sine frequency must be greater than 0 Hz."
            )
            return

        if amplitude < 0:
            messagebox.showerror(
                "Invalid amplitude",
                "Sine amplitude must be positive."
            )
            return

        min_angle = offset - amplitude
        max_angle = offset + amplitude

        if min_angle < -45.0 or max_angle > 45.0:
            messagebox.showerror(
                "Invalid sine range",
                f"Sine reference exceeds safe range.\n\n"
                f"Minimum angle: {min_angle:.2f} deg\n"
                f"Maximum angle: {max_angle:.2f} deg\n\n"
                f"Keep the full sine wave between -45 and 45 degrees."
            )
            return
        
        self.reference_filename_label = f"sine_offset_{offset:g}_amp_{amplitude:g}_freq_{frequency:g}Hz"

        self.send_command(f"SINE {offset} {amplitude} {frequency}")

    def start_trial(self):
        self.current_trial_id += 1
        self.send_command("START")

    def stop_trial(self):
        self.send_command("STOP")

    def request_status(self):
        self.send_command("STATUS")

    def idle_motor(self):
        self.send_command("IDLE")

    def zero_encoder(self):
        self.send_command("ZERO")

    def update_loop(self):
        self.read_serial_data()
        self.update_plot()

        # Run this loop again after 50 ms
        self.root.after(10, self.update_loop)

    # ======================================================
    # Read serial data without threading
    # ======================================================
    def read_serial_data(self):
        if not self.connected or self.ser is None:
            return

        try:
            # Read all available lines in the serial buffer
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()

                if line:
                    self.process_serial_line(line)

        except Exception as e:
            self.log_message(f"Serial read error: {e}")

    # ======================================================
    # Process incoming Arduino line
    # ======================================================
    def process_serial_line(self, line):
        # print(f"Received line: {line}")  # Debug print for all incoming lines
        # Arduino text messages
        if line.startswith("READY"):
            self.log_message(line)
            return

        if line.startswith("Commands"):
            self.log_message(line)
            return

        if line.startswith("TRIAL_STARTED"):
            self.trial_active = True
            self.status_var.set("Trial running")
            self.log_message("Trial started")
            return

        if line.startswith("TRIAL_STOPPED"):
            self.trial_active = False
            self.status_var.set("Trial stopped")
            self.log_message("Trial stopped")
            return

        if line.startswith("PID_UPDATED"):
            parts = line.split(",")

            if len(parts) == 4:
                kp = parts[1]
                ki = parts[2]
                kd = parts[3]

                self.latest_pid_var.set(f"Kp: {kp}   Ki: {ki}   Kd: {kd}")
                self.log_message(f"PID updated: Kp={kp}, Ki={ki}, Kd={kd}")
            else:
                self.log_message(line)

            return

        if line.startswith("STATUS"):
            self.log_message(line)
            return

        if line.startswith("ERROR"):
            self.log_message(line)
            return
        
        if line.startswith("THETA_MIN_CAPTURED"):
            self.status_var.set("Theta min captured")
            self.log_message(line)
            return
        
        if line.startswith("THETA_MAX_CAPTURED"):
            self.status_var.set("Theta max captured")
            self.log_message(line)
            return

        if line.startswith("time_ms"):
            return

        # Expected data from Arduino:
        # time_ms,theta,theta_dot,control_output,kp,ki,kd
        parts = line.split(",")

        if len(parts) != 7:
            # Do not spam the GUI if the Arduino sends other text
            return

        try:
            t_ms = float(parts[0])
            theta = float(parts[1])
            control_output = float(parts[2])
            kp = float(parts[3])
            ki = float(parts[4])
            kd = float(parts[5])
            theta_dot = float(parts[6])

            t_sec = t_ms / 1000.0

            # Only reject samples with repeated/backwards time.
            # No theta_dot calculation is done in the GUI.
            if len(self.time_data) > 0 and t_sec <= self.time_data[-1]:
                return

            self.time_data.append(t_sec)
            self.theta_data.append(theta)
            self.control_data.append(control_output)
            self.theta_dot_data.append(theta_dot)

            self.all_rows.append([
                self.current_trial_id,
                t_ms,
                theta,
                theta_dot,
                control_output,
                kp,
                ki,
                kd
            ])

            self.latest_theta_var.set(f"Theta: {theta:.4f}")
            self.latest_theta_dot_var.set(f"Theta dot: {theta_dot:.4f}")
            self.latest_control_var.set(f"Control: {control_output:.4f}")
            self.latest_pid_var.set(f"Kp: {kp:.4f}   Ki: {ki:.4f}   Kd: {kd:.4f}")

        except ValueError:
            self.log_message(f"Could not parse: {line}")

    # ======================================================
    # Plot update
    # ======================================================
    def update_plot(self):
        if len(self.time_data) == 0:
            return

        t = list(self.time_data)

        self.line_control.set_data(t, list(self.control_data))
        self.line_theta.set_data(t, list(self.theta_data))
        self.line_theta_dot.set_data(t, list(self.theta_dot_data))

        self.ax_control.relim()
        self.ax_control.autoscale_view()

        self.ax_theta.relim()
        self.ax_theta.set_ylim(-55, 55)

        self.ax_theta_dot.relim()
        self.ax_theta_dot.autoscale_view()

        # Show only latest 10 seconds
        latest_time = t[-1]
        window = 10.0

        if latest_time > window:
            self.ax_theta_dot.set_xlim(latest_time - window, latest_time)
        else:
            self.ax_theta_dot.set_xlim(0, window)

        self.canvas.draw_idle()

    # ======================================================
    # Save CSV
    # ======================================================
    def save_csv(self):
        if len(self.all_rows) == 0:
            messagebox.showinfo("No data", "No data to save yet.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"pid_trial_{self.reference_filename_label}_{timestamp}.csv"

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV files", "*.csv")]
        )

        if not filename:
            return

        try:
            with open(filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "trial_id",
                    "time_ms",
                    "theta",
                    "theta_dot",
                    "control_output",
                    "kp",
                    "ki",
                    "kd"
                ])
                writer.writerows(self.all_rows)

            self.log_message(f"Saved CSV: {filename}")
            messagebox.showinfo("Saved", f"Data saved to:\n{filename}")

        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # ======================================================
    # Clear plot
    # ======================================================
    def clear_plot(self):
        self.time_data.clear()
        self.theta_data.clear()
        self.control_data.clear()
        self.theta_dot_data.clear()

        self.all_rows.clear()

        self.line_control.set_data([], [])
        self.line_theta.set_data([], [])
        self.line_theta_dot.set_data([], [])

        self.ax_control.relim()
        self.ax_control.autoscale_view()

        self.ax_theta.relim()
        self.ax_theta.autoscale_view()

        self.ax_theta_dot.relim()
        self.ax_theta_dot.autoscale_view()

        self.canvas.draw_idle()

        self.log_message("Plot and data cleared")

    # ======================================================
    # Serial message log
    # ======================================================
    def log_message(self, message):
        self.serial_text.insert(tk.END, message + "\n")
        self.serial_text.see(tk.END)

    # ======================================================
    # Window close
    # ======================================================
    def on_close(self):
        try:
            if self.connected:
                self.send_command("STOP")
            self.disconnect_serial()
        except Exception:
            pass

        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PIDMonitorGUI(root)

    root.protocol("WM_DELETE_WINDOW", app.on_close)

    root.mainloop()
