volatile long counter = 0;
long lastPosition = 0;

// PID constants
float Kp = 1.5;
float Ki = 0.2;
float Kd = 0.05;

float setpoint = 0;

// PID variables
float error = 0;
float prev_error = 0;
float integral = 0;
float derivative = 0;

unsigned long lastTime = 0;

// Calibration variables
long startCount = 0;
long stopCount = 1;   // avoid divide-by-zero
bool calibrated = false;

void setup() {
  Serial.begin(9600);

  pinMode(2, INPUT_PULLUP);
  pinMode(3, INPUT_PULLUP);

  attachInterrupt(0, ai0, RISING);
  attachInterrupt(1, ai1, RISING);

  lastTime = millis();

  Serial.println("Type 'start' to record minimum position");
  Serial.println("Then rotate encoder and type 'stop'");
}

void loop() {

  // =========================
  // SERIAL COMMANDS
  // =========================
  if (Serial.available()) {

    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // Record starting encoder value
    if (cmd == "start") {

      startCount = counter;

      Serial.print("Start position recorded: ");
      Serial.println(startCount);
    }

    // Record ending encoder value
    else if (cmd == "stop") {

      stopCount = counter;

      calibrated = true;

      Serial.print("Stop position recorded: ");
      Serial.println(stopCount);

      Serial.println("Calibration complete");
      Serial.println("Mapped angle range: -50 to 50 degrees");
    }

    // Otherwise assume numeric setpoint
    else {

      float newSetpoint = cmd.toFloat();

      setpoint = newSetpoint;

      integral = 0;
      prev_error = 0;

      Serial.print("New Setpoint: ");
      Serial.println(setpoint);
    }
  }

  // =========================
  // PID + ANGLE DISPLAY
  // =========================
  if (counter != lastPosition && calibrated) {

    unsigned long now = millis();
    float dt = (now - lastTime) / 1000.0;

    if (dt <= 0) dt = 0.001;

    long position = counter;

    // Map encoder count to angle
    float angle = mapFloat(position,
                           startCount,
                           stopCount,
                           -50.0,
                           50.0);

    // Keep angle bounded
    angle = constrain(angle, -50.0, 50.0);

    // PID calculations
    error = setpoint - angle;

    integral += error * dt;

    derivative = (error - prev_error) / dt;

    float output =
      Kp * error +
      Ki * integral +
      Kd * derivative;

    // Serial output
    Serial.print("Raw Count: ");
    Serial.print(position);

    Serial.print(" | Angle: ");
    Serial.print(angle);

    Serial.print(" deg");

    Serial.print(" | Setpoint: ");
    Serial.print(setpoint);

    Serial.print(" | Error: ");
    Serial.print(error);

    Serial.print(" | Output: ");
    Serial.println(output);

    prev_error = error;
    lastTime = now;
    lastPosition = position;
  }
}

// ===================================
// FLOAT VERSION OF MAP()
// ===================================
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

// ===================================
// ENCODER INTERRUPTS
// ===================================
void ai0() {

  if (digitalRead(3) == LOW) {
    counter++;
  }
  else {
    counter--;
  }
}

void ai1() {

  if (digitalRead(2) == LOW) {
    counter--;
  }
  else {
    counter++;
  }
}