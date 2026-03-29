import serial
import time

ser = serial.Serial('COM3', 115200, timeout=1)  
time.sleep(2) 

Kp = 1.0
Ki = 0.1
Kd = 0.05

setpoint = 512  # Target value (midpoint of 1–1024)


previous_error = 0
integral = 0
last_time = time.time()

def pid_control(measured_value):
    global previous_error, integral, last_time

    current_time = time.time()
    dt = current_time - last_time

    if dt <= 0.0:
        return 0

    error = setpoint - measured_value

    integral += error * dt

    derivative = (error - previous_error) / dt

    output = (Kp * error) + (Ki * integral) + (Kd * derivative)

    previous_error = error
    last_time = current_time

    return output

while True:
    try:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()

            if line.isdigit():
                analog_value = int(line)

                analog_value = max(1, min(1024, analog_value))

                control_signal = pid_control(analog_value)

                print(f"Input: {analog_value} | Output: {control_signal:.2f}")

    except KeyboardInterrupt:
        print("Exiting...")
        break

ser.close()
