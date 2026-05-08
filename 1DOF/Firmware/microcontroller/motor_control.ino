#include <Servo.h>

Servo motor1;
Servo motor2;

int power1 = 0;
int power2 = 0;

void setup() {

  Serial.begin(9600);

  motor1.attach(5);
  motor2.attach(7);

  motor1.writeMicroseconds(1000);
  motor2.writeMicroseconds(1000);

  Serial.println("Enter motor A and B power:");
}

void loop() {

  if (Serial.available() > 0) {

    String input = Serial.readStringUntil('\n');
    int commaIndex = input.indexOf(',');

    if (commaIndex > 0) {
      String value1 = input.substring(0, commaIndex);
      String value2 = input.substring(commaIndex + 1);

      power1 = value1.toInt();
      power2 = value2.toInt();

      power1 = constrain(power1, 0, 100);
      power2 = constrain(power2, 0, 100);

      int pwm1 = map(power1, 0, 100, 1000, 2000);
      int pwm2 = map(power2, 0, 100, 1000, 2000);

      motor1.writeMicroseconds(pwm1);
      motor2.writeMicroseconds(pwm2);

      Serial.print("Motor 1(%): ");
      Serial.print(power1);
      Serial.print(" PWM: ");
      Serial.print(pwm1);

      Serial.print(" | Motor 2(%): ");
      Serial.print(power2);
      Serial.print(" PWM: ");
      Serial.println(pwm2);

    } else {

      Serial.println("Invalid format. Use: motor1,motor2");

    }
  }
}