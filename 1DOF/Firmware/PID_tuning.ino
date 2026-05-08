#include <Servo.h>

// =======================================================
// Motor setup
// =======================================================

Servo motor1;
Servo motor2;

const int MOTOR1_PIN = 5;
const int MOTOR2_PIN = 7;

// Motor power is mapped from 0 to 100 percent
const int MOTOR_POWER_MIN = 0;
const int MOTOR_POWER_MAX = 100;

// ESC microsecond range
const int ESC_MIN_US = 1000;
const int ESC_MAX_US = 2000;

// Base motor power
// Both motors will run at this power when PID output is zero
int basePower = 40;

// Limit how much the PID can change motor power
int maxDeltaPower = 30;


// =======================================================
// Encoder setup
// =======================================================

const int ENCODER_PIN_A = 2;
const int ENCODER_PIN_B = 3;

volatile long counter = 0;

// Your physical seesaw angle range
const float THETA_MAX_DEG = 51.0;
const float THETA_MIN_DEG = -51.0;

// IMPORTANT:
// Change this value after measuring your encoder count difference
// from +51 deg to -51 deg.
const float COUNTS_FULL_RANGE = 220.0;

// If the angle direction is wrong, change this to -1
const int ENCODER_DIRECTION = 1;


// =======================================================
// PID setup
// =======================================================

float Kp = 1.5;
float Ki = 0.0;
float Kd = 0.05;

float setpoint = 0.0;

float error = 0.0;
float prevError = 0.0;
float integral = 0.0;
float derivative = 0.0;

const float INTEGRAL_MIN = -100.0;
const float INTEGRAL_MAX = 100.0;


// =======================================================
// Control loop setup
// =======================================================

bool trialRunning = false;

unsigned long trialStartTime = 0;
unsigned long lastControlTime = 0;

const unsigned long SAMPLE_TIME_MS = 10;   // 100 Hz


// =======================================================
// Function prototypes
// =======================================================

void handleSerialCommand(String cmd);
void startTrial();
void stopTrial();

float readTheta();
float computePID(float theta, float dt);
void applyControl(float controlOutput);
void setMotorPower(int power1, int power2);
void stopMotors();

void ai0();
void ai1();


// =======================================================
// Setup
// =======================================================

void setup() {
  Serial.begin(115200);

  // Motor setup
  motor1.attach(MOTOR1_PIN);
  motor2.attach(MOTOR2_PIN);

  stopMotors();

  // Encoder setup
  pinMode(ENCODER_PIN_A, INPUT_PULLUP);
  pinMode(ENCODER_PIN_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_A), ai0, RISING);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_B), ai1, RISING);

  lastControlTime = millis();

  Serial.println("READY");
  Serial.println("Commands:");
  Serial.println("SET Kp Ki Kd");
  Serial.println("START");
  Serial.println("STOP");
  Serial.println("STATUS");
  Serial.println("Output format:");
  Serial.println("time_ms,theta,control_output,kp,ki,kd");
}


// =======================================================
// Main loop
// =======================================================

void loop() {
  // Read commands from Python
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    handleSerialCommand(cmd);
  }

  // Run PID only when trial is active
  if (trialRunning) {
    unsigned long now = millis();

    if (now - lastControlTime >= SAMPLE_TIME_MS) {
      float dt = (now - lastControlTime) / 1000.0;
      lastControlTime = now;

      if (dt <= 0.0) {
        dt = 0.001;
      }

      float theta = readTheta();
      float controlOutput = computePID(theta, dt);

      applyControl(controlOutput);

      unsigned long trialTime = now - trialStartTime;

      // Send data to Python
      Serial.print(trialTime);
      Serial.print(",");
      Serial.print(theta, 6);
      Serial.print(",");
      Serial.print(controlOutput, 6);
      Serial.print(",");
      Serial.print(Kp, 6);
      Serial.print(",");
      Serial.print(Ki, 6);
      Serial.print(",");
      Serial.println(Kd, 6);
    }
  }
}


// =======================================================
// Serial command handling
// =======================================================

void handleSerialCommand(String cmd) {
  cmd.trim();

  String upperCmd = cmd;
  upperCmd.toUpperCase();

  if (upperCmd.startsWith("SET")) {
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    int thirdSpace = cmd.indexOf(' ', secondSpace + 1);

    if (firstSpace < 0 || secondSpace < 0 || thirdSpace < 0) {
      Serial.println("ERROR: Use SET Kp Ki Kd");
      return;
    }

    float newKp = cmd.substring(firstSpace + 1, secondSpace).toFloat();
    float newKi = cmd.substring(secondSpace + 1, thirdSpace).toFloat();
    float newKd = cmd.substring(thirdSpace + 1).toFloat();

    Kp = newKp;
    Ki = newKi;
    Kd = newKd;

    integral = 0.0;
    prevError = 0.0;

    Serial.print("PID_UPDATED,");
    Serial.print(Kp, 6);
    Serial.print(",");
    Serial.print(Ki, 6);
    Serial.print(",");
    Serial.println(Kd, 6);
  }

  else if (upperCmd == "START") {
    startTrial();
  }

  else if (upperCmd == "STOP") {
    stopTrial();
  }

  else if (upperCmd == "STATUS") {
    Serial.print("STATUS,");
    Serial.print(trialRunning ? "RUNNING" : "STOPPED");
    Serial.print(",");
    Serial.print("theta=");
    Serial.print(readTheta(), 6);
    Serial.print(",");
    Serial.print("Kp=");
    Serial.print(Kp, 6);
    Serial.print(",");
    Serial.print("Ki=");
    Serial.print(Ki, 6);
    Serial.print(",");
    Serial.print("Kd=");
    Serial.println(Kd, 6);
  }

  else {
    Serial.println("ERROR: Unknown command");
  }
}


// =======================================================
// Start trial
// =======================================================

void startTrial() {
  if (trialRunning) {
    Serial.println("ERROR: Trial already running");
    return;
  }

  integral = 0.0;
  prevError = 0.0;

  trialStartTime = millis();
  lastControlTime = millis();

  trialRunning = true;

  Serial.println("TRIAL_STARTED");
  Serial.println("time_ms,theta,control_output,kp,ki,kd");
}


// =======================================================
// Stop trial
// =======================================================

void stopTrial() {
  if (!trialRunning) {
    Serial.println("ERROR: Trial already stopped");
    stopMotors();
    return;
  }

  trialRunning = false;

  stopMotors();

  integral = 0.0;
  prevError = 0.0;

  Serial.println("TRIAL_STOPPED");
}


// =======================================================
// Read theta from encoder
// =======================================================

float readTheta() {
  long countSnapshot;

  noInterrupts();
  countSnapshot = counter;
  interrupts();

  float correctedCount = ENCODER_DIRECTION * (float)countSnapshot;

  // At Arduino startup:
  // counter = 0 means theta = +51 deg
  //
  // After moving through full range:
  // counter = COUNTS_FULL_RANGE means theta = -51 deg

  float theta = THETA_MAX_DEG -
                (correctedCount / COUNTS_FULL_RANGE) *
                (THETA_MAX_DEG - THETA_MIN_DEG);

  theta = constrain(theta, THETA_MIN_DEG, THETA_MAX_DEG);

  return theta;
}


// =======================================================
// PID controller
// =======================================================

float computePID(float theta, float dt) {
  error = setpoint - theta;

  integral += error * dt;
  integral = constrain(integral, INTEGRAL_MIN, INTEGRAL_MAX);

  derivative = (error - prevError) / dt;

  float output = Kp * error + Ki * integral + Kd * derivative;

  prevError = error;

  return output;
}


// =======================================================
// Convert PID output to motor power
// =======================================================

void applyControl(float controlOutput) {
  // The PID output is treated as differential motor power.
  // Positive output increases motor1 and decreases motor2.

  int deltaPower = (int)controlOutput;

  deltaPower = constrain(deltaPower, -maxDeltaPower, maxDeltaPower);

  int power1 = basePower + deltaPower;
  int power2 = basePower - deltaPower;

  power1 = constrain(power1, MOTOR_POWER_MIN, MOTOR_POWER_MAX);
  power2 = constrain(power2, MOTOR_POWER_MIN, MOTOR_POWER_MAX);

  setMotorPower(power1, power2);
}


// =======================================================
// Write motor power as ESC PWM
// =======================================================

void setMotorPower(int power1, int power2) {
  power1 = constrain(power1, 0, 100);
  power2 = constrain(power2, 0, 100);

  int pwm1 = map(power1, 0, 100, ESC_MIN_US, ESC_MAX_US);
  int pwm2 = map(power2, 0, 100, ESC_MIN_US, ESC_MAX_US);

  motor1.writeMicroseconds(pwm1);
  motor2.writeMicroseconds(pwm2);
}


// =======================================================
// Stop motors
// =======================================================

void stopMotors() {
  motor1.writeMicroseconds(ESC_MIN_US);
  motor2.writeMicroseconds(ESC_MIN_US);
}


// =======================================================
// Encoder interrupt functions
// =======================================================

void ai0() {
  if (digitalRead(ENCODER_PIN_B) == LOW) {
    counter++;
  } else {
    counter--;
  }
}

void ai1() {
  if (digitalRead(ENCODER_PIN_A) == LOW) {
    counter--;
  } else {
    counter++;
  }
}