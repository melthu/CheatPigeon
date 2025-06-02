#include <Arduino.h>
#include <Servo.h>
#include <math.h>
#include <algorithm>

#define stepPin 5
#define dirPin 3
#define stepPin2 6
#define dirPin2 4
#define servoPin 7

const int MIN_DELAY = 10;  
const int MAX_DELAY = 60;  
const int MAX_RAMP_STEPS = 50;

Servo myServo;

void servoUp() {
  myServo.write(5);  
}

void servoDown() {
  myServo.write(25);
}

void stepPulse(uint8_t pin, int delayMicros) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(delayMicros);
  digitalWrite(pin, LOW);
  delayMicroseconds(delayMicros);
}

void moveStepPairWithRamp(uint8_t dir1, uint8_t dir2, int stepCount) {
  digitalWrite(dirPin, dir1);
  digitalWrite(dirPin2, dir2);

  int rampSteps = min(stepCount / 2, MAX_RAMP_STEPS);

  for (int i = 0; i < stepCount; i++) {
    int delayMicros;
    if (i < rampSteps) {
      delayMicros = map(i, 0, rampSteps, MAX_DELAY, MIN_DELAY);
    } else if (i > stepCount - rampSteps) {
      delayMicros = map(i, stepCount - rampSteps, stepCount, MIN_DELAY, MAX_DELAY);
    } else {
      delayMicros = MIN_DELAY;
    }

    stepPulse(stepPin, delayMicros);
    stepPulse(stepPin2, delayMicros);
  }
}

void moveRight()  { moveStepPairWithRamp(HIGH, HIGH, 1); }
void moveLeft()   { moveStepPairWithRamp(LOW, LOW, 1); }
void moveUp()     { moveStepPairWithRamp(HIGH, LOW, 1); }
void moveDown()   { moveStepPairWithRamp(LOW, HIGH, 1); }

void moveX(int steps) {
  if (steps > 0) {
    for (int i = 0; i < steps; i++) moveRight();
  } else {
    for (int i = 0; i < -steps; i++) moveLeft();
  }
}

void moveY(int steps) {
  if (steps > 0) {
    for (int i = 0; i < steps; i++) moveUp();
  } else {
    for (int i = 0; i < -steps; i++) moveDown();
  }
}

int gcd(int a, int b) {
  while (b != 0) {
    int temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

void move(int dx, int dy) {
  if (dx == 0 && dy == 0) return;

  int x = 0, y = 0;
  int sx = (dx > 0) ? 1 : -1;
  int sy = (dy > 0) ? 1 : -1;

  int absDx = abs(dx);
  int absDy = abs(dy);

  if (absDx > absDy) {
    int err = absDx / 2;
    for (int i = 0; i < absDx; i++) {
      moveX(sx);
      err -= absDy;
      if (err < 0) {
        moveY(sy);
        err += absDx;
      }
    }
  } else {
    int err = absDy / 2;
    for (int i = 0; i < absDy; i++) {
      moveY(sy);
      err -= absDx;
      if (err < 0) {
        moveX(sx);
        err += absDy;
      }
    }
  }
}

bool readMove(int &dx, int &dy, bool &penCommand, bool &penUp) {
  static char buf[64];
  static uint8_t idx = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == ';') {
      buf[idx] = '\0';
      idx = 0;

      // ——— parse token #1: dx ———
      char* p = strtok(buf, ",");
      if (!p) return false;
      dx = atoi(p);

      // ——— parse token #2: dy ———
      p = strtok(NULL, ",");
      if (!p) return false;
      dy = atoi(p);

      // ——— parse token #3: penCommand flag ———
      //   1 = this is a pen operation, 0 = just movement
      p = strtok(NULL, ",");
      penCommand = p ? (atoi(p) != 0) : false;

      // ——— parse token #4: penUp flag ———
      //   1 = lift pen, 0 = lower pen
      p = strtok(NULL, ",");
      penUp = p ? (atoi(p) != 0) : false;

      return true;
    }

    if (idx < sizeof(buf) - 1) {
      buf[idx++] = c;
    }
  }
  return false;
}

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(stepPin2, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(dirPin2, OUTPUT);

  myServo.attach(servoPin);

  Serial.begin(9600);
  Serial.println("ready");
  servoUp();
}

void loop() {

int dx, dy;
  bool penCommand, penUp;
  while (readMove(dx, dy, penCommand, penUp)) {
    if (penCommand) {
      if (penUp) servoUp();
      else servoDown();
    } else {
      move(dx, dy);
    }
  }
  
  if (!Serial.available()) {
    Serial.println("done");
    Serial.flush();
    delay(100);
  }
}
