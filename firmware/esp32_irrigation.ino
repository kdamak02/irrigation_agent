#include <Wire.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

const int soil_pin = 34;
const int water_pin = 35;

const int relay_pin = 26;
const int led_green = 25;
const int led_red = 33;

// Calibration capteur niveau eau
int water_min = 200;
int water_max = 2300;

String pumpStatus = "OFF";

void pumpON() {
  digitalWrite(relay_pin, HIGH);   // HIGH = pompe ON chez toi
  digitalWrite(led_green, HIGH);
  digitalWrite(led_red, LOW);
  pumpStatus = "ON";
}

void pumpOFF() {
  digitalWrite(relay_pin, LOW);    // LOW = pompe OFF chez toi
  digitalWrite(led_green, LOW);
  digitalWrite(led_red, HIGH);
  pumpStatus = "OFF";
}

void setup() {
  Serial.begin(115200);

  pinMode(relay_pin, OUTPUT);
  pinMode(led_green, OUTPUT);
  pinMode(led_red, OUTPUT);

  pumpOFF();

  Wire.begin(21, 22);

  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("Agent IA Ready");
  delay(1500);
  lcd.clear();
}

void loop() {
  int soil_raw = analogRead(soil_pin);
  int water_raw = analogRead(water_pin);

  int soil = map(soil_raw, 4095, 0, 0, 100);
  soil = constrain(soil, 0, 100);

  int water = map(water_raw, water_min, water_max, 0, 100);
  water = constrain(water, 0, 100);

  // Envoi vers Python / Streamlit
  Serial.print("MOISTURE:");
  Serial.print(soil);
  Serial.print(";WATER:");
  Serial.println(water);

  // Reception commande depuis Python / Streamlit
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "ON") {
      pumpON();
    } else if (cmd == "OFF") {
      pumpOFF();
    }
  }

  // LCD compact
  lcd.setCursor(0, 0);
  lcd.print("S:");
  lcd.print(soil);
  lcd.print("% W:");
  lcd.print(water);
  lcd.print("%   ");

  lcd.setCursor(0, 1);
  lcd.print("IA Pump:");
  lcd.print(pumpStatus);
  lcd.print("     ");

  delay(1000);
}