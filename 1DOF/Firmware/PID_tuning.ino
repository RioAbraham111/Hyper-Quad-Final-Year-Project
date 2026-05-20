#include <Servo.h>

// =======================================================
// Motor setup
// =======================================================

Servo motor1;
Servo motor2;

const int MOTOR1_PIN = 5;
const int MOTOR2_PIN = 7;

// Motor power is mapped from 0 to 100 percent
const int MOTOR_POWER_MIN = 7;
const int MOTOR_POWER_MAX = 70;

// ESC microsecond range
const int ESC_MIN_US = 1000;
const int ESC_MAX_US = 2000;

// Base motor power
// Both motors will run at this power when PID output is zero
int basePower = 12;

// Limit how much the PID can change motor power
int maxDeltaPower = 30;


// =======================================================
// Encoder setup
// =======================================================

const int ENCODER_PIN_A = 2;
const int ENCODER_PIN_B = 3;

volatile long counter = 0;
volatile uint8_t lastEncoded = 0;

// Your physical seesaw angle range
const float THETA_MAX_DEG = 51.0;
const float THETA_MIN_DEG = -51.0;

// IMPORTANT:
// Change this value after measuring your encoder count difference
// from +51 deg to -51 deg.
long counterAtThetaMin = 0;
long counterAtThetaMax = -220;

bool waitingForMaxCalibration = false;


// If the angle direction is wrong, change this to -1
const int ENCODER_DIRECTION = 1;

// =======================================================
// Encoder median filter
// =======================================================

const int MEDIAN_FILTER_SIZE = 5;

long encoderHistory[MEDIAN_FILTER_SIZE] = {0};
int encoderHistoryIndex = 0;
bool encoderHistoryFilled = false;


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

float thetaDot = 0.0;
float previousTheta = 0.0;

// =======================================================
// Reference / setpoint mode
// =======================================================

enum ReferenceMode {
  CONSTANT_REFERENCE,
  SINE_REFERENCE
};

ReferenceMode referenceMode = CONSTANT_REFERENCE;

float desiredAngle = 0.0;

float sineOffset = 0.0;
float sineAmplitude = 10.0;
float sineFrequencyHz = 0.1;

unsigned long sineStartTime = 0;

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
void idleMode();
void stopTrial();

float readTheta();
float computePID(float theta, float thetaReference, float dt);
float mapFloat();
void applyControl(float controlOutput);
void setMotorPower(int power1, int power2);
void idleMotors();
void stopMotors();

void ai0();
void ai1();

long getMedian(long arr[], int size);

// =======================================================
// Setup
// =======================================================

void setup() {
  Serial.begin(115200);

  // Motor setup
  motor1.attach(MOTOR1_PIN);
  motor2.attach(MOTOR2_PIN);

  stopMotors();

  pinMode(ENCODER_PIN_A, INPUT_PULLUP);
  pinMode(ENCODER_PIN_B, INPUT_PULLUP);

  uint8_t MSB = digitalRead(ENCODER_PIN_A);
  uint8_t LSB = digitalRead(ENCODER_PIN_B);
  lastEncoded = (MSB << 1) | LSB;

  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_A), updateEncoder, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENCODER_PIN_B), updateEncoder, CHANGE);

  lastControlTime = millis();

  Serial.println("READY");
  Serial.println("Commands:");
  Serial.println("SET Kp Ki Kd");
  Serial.println("START");
  Serial.println("STOP");
  Serial.println("IDLE");
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

      float rawPosition = readTheta();
      float theta = mapFloat(rawPosition,
                            counterAtThetaMin,
                            counterAtThetaMax,
                            THETA_MIN_DEG,
                            THETA_MAX_DEG);

      // theta derivative [deg/s]
      thetaDot = (theta - previousTheta) / dt;
      previousTheta = theta;

      float thetaReference = getReferenceAngle();
      float controlOutput = computePID(theta, thetaReference, dt);

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
      Serial.print(Kd, 6);
      Serial.print(",");
      Serial.println(thetaDot,6);
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

  else if (upperCmd.startsWith("ANGLE")) {
    int space = cmd.indexOf(' ');

    if (space < 0) {
      Serial.println("ERROR: Use ANGLE <setpoint>");
      return;
    }

    float newSetpoint = cmd.substring(space + 1).toFloat();
    setpoint = newSetpoint;

    integral = 0.0;   // optional: prevents windup when target changes
    prevError = 0.0;

    Serial.print("SETPOINT_UPDATED,");
    Serial.println(setpoint, 6);
  }
  
  else if (upperCmd.startsWith("SINE")) {
    int firstSpace = cmd.indexOf(' ');
    int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    int thirdSpace = cmd.indexOf(' ', secondSpace + 1);

    if (firstSpace < 0 || secondSpace < 0 || thirdSpace < 0) {
      Serial.println("ERROR: Use SINE offset amplitude frequencyHz");
      return;
    }

    float newOffset = cmd.substring(firstSpace + 1, secondSpace).toFloat();
    float newAmplitude = cmd.substring(secondSpace + 1, thirdSpace).toFloat();
    float newFrequency = cmd.substring(thirdSpace + 1).toFloat();

    if (newAmplitude < 0.0) {
      Serial.println("ERROR: Sine amplitude must be positive");
      return;
    }

    if (newFrequency <= 0.0) {
      Serial.println("ERROR: Sine frequency must be greater than 0");
      return;
    }

    float minAngle = newOffset - newAmplitude;
    float maxAngle = newOffset + newAmplitude;

    if (minAngle < -45.0 || maxAngle > 45.0) {
      Serial.println("ERROR: Sine reference must stay within -45 to 45 degrees");
      return;
    }

    sineOffset = newOffset;
    sineAmplitude = newAmplitude;
    sineFrequencyHz = newFrequency;
    sineStartTime = millis();

    referenceMode = SINE_REFERENCE;

    integral = 0.0;
    prevError = 0.0;

    Serial.print("SINE_UPDATED,");
    Serial.print(sineOffset, 6);
    Serial.print(",");
    Serial.print(sineAmplitude, 6);
    Serial.print(",");
    Serial.println(sineFrequencyHz, 6);
  }


  else if (upperCmd == "START") {
    startTrial();
  }

  else if (upperCmd == "STOP") {
    stopTrial();
  }

  else if (upperCmd == "IDLE") {
    idleMode();
  }

  else if (upperCmd == "STATUS") {
    Serial.print("STATUS,");
    Serial.print(trialRunning ? "RUNNING" : "STOPPED");
    Serial.print(",");
    Serial.print("theta ref="),
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

  else if (upperCmd == "ZERO") {

    long countSnapshot;

    noInterrupts();
    countSnapshot = counter;
    interrupts();

    if (!waitingForMaxCalibration) {

      counterAtThetaMin = countSnapshot;

      Serial.print("THETA_MIN_CAPTURED,");
      Serial.print(counterAtThetaMin);

      Serial.println("MOVE_TO_NEXT_POSITION");

      waitingForMaxCalibration = true;
    }
    else {

      counterAtThetaMax = countSnapshot;

      Serial.print("THETA_MAX_CAPTURED,");
      Serial.print(counterAtThetaMax);

      Serial.println("CALIBRATION_COMPLETE");

      waitingForMaxCalibration = false;
    }
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
// Idle mode
// =======================================================

void idleMode() {
  trialRunning = false;

  idleMotors();

  Serial.println("IDLE_MODE");
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

  encoderHistory[encoderHistoryIndex] = countSnapshot;

  encoderHistoryIndex++;

  if (encoderHistoryIndex >= MEDIAN_FILTER_SIZE) {
    encoderHistoryIndex = 0;
    encoderHistoryFilled = true;
  }

  if (!encoderHistoryFilled) {
    return countSnapshot;
  }

  long temp[MEDIAN_FILTER_SIZE];

  for (int i = 0; i < MEDIAN_FILTER_SIZE; i++) {
    temp[i] = encoderHistory[i];
  }

  return getMedian(temp, MEDIAN_FILTER_SIZE);
}

float mapFloat(float x,
               float in_min,
               float in_max,
               float out_min,
               float out_max) {

  return (x - in_min) *
         (out_max - out_min) /
         (in_max - in_min) +
         out_min;
}

long getMedian(long arr[], int size) {

  for (int i = 0; i < size - 1; i++) {

    for (int j = i + 1; j < size; j++) {

      if (arr[j] < arr[i]) {

        long temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }

  return arr[size / 2];
}

float getReferenceAngle() {
  if (referenceMode == CONSTANT_REFERENCE) {
    return desiredAngle;
  }

  else if (referenceMode == SINE_REFERENCE) {
    float t = (millis() - sineStartTime) / 1000.0;

    float reference =
      sineOffset +
      sineAmplitude * sin(2.0 * PI * sineFrequencyHz * t);

    reference = constrain(reference, -45.0, 45.0);

    return reference;
  }

  return desiredAngle;
}

// =======================================================
// PID controller
// =======================================================

float computePID(float theta, float thetaReference, float dt) {
  error = thetaReference - theta;

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

void idleMotors() {
  int pwm1 = map(7, 0, 100, ESC_MIN_US, ESC_MAX_US);
  int pwm2 = map(7, 0, 100, ESC_MIN_US, ESC_MAX_US);
  motor1.writeMicroseconds(pwm1);
  motor2.writeMicroseconds(pwm2);
}


void stopMotors(){
  motor1.writeMicroseconds(1000);
  motor2.writeMicroseconds(1000);
}

// =======================================================
// Encoder interrupt functions
// =======================================================

void updateEncoder() {
  uint8_t MSB = digitalRead(ENCODER_PIN_A);
  uint8_t LSB = digitalRead(ENCODER_PIN_B);

  uint8_t encoded = (MSB << 1) | LSB;
  uint8_t sum = (lastEncoded << 2) | encoded;

  // Valid clockwise transitions
  if (
    sum == 0b1101 ||
    sum == 0b0100 ||
    sum == 0b0010 ||
    sum == 0b1011
  ) {
    counter++;
  }

  // Valid counter-clockwise transitions
  else if (
    sum == 0b1110 ||
    sum == 0b0111 ||
    sum == 0b0001 ||
    sum == 0b1000
  ) {
    counter--;
  }

  // Invalid transitions are ignored.
  // This helps reject noise/glitches.

  lastEncoded = encoded;
}