#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// ============================================================
// ESP32 DEV MODULE - SMART WATER QUALITY MONITOR
// Sends pH, turbidity and temperature to laptop over USB Serial.
// ============================================================


// ---------- Sensor pins ----------
#define PH_SENSOR_PIN        34
#define TURBIDITY_SENSOR_PIN 36
#define ONE_WIRE_BUS          5

// ---------- pH calibration ----------
float calibration_value = 21.34 + 1.5;

unsigned long int avgval;
int buffer_arr[10];
int temp_sort;
float ph_act;

// ---------- Turbidity ----------
int turbidity_raw;

// ---------- Temperature ----------
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature temperatureSensor(&oneWire);

unsigned long lastSend = 0;
const unsigned long SEND_INTERVAL = 1000;

float readPH()
{
  for (int i = 0; i < 10; i++) {
    buffer_arr[i] = analogRead(PH_SENSOR_PIN);
    delay(20);
  }

  // Sort readings.
  for (int i = 0; i < 9; i++) {
    for (int j = i + 1; j < 10; j++) {
      if (buffer_arr[i] > buffer_arr[j]) {
        temp_sort = buffer_arr[i];
        buffer_arr[i] = buffer_arr[j];
        buffer_arr[j] = temp_sort;
      }
    }
  }

  // Average middle 6 readings.
  avgval = 0;
  for (int i = 2; i < 8; i++) {
    avgval += buffer_arr[i];
  }

  float volt = (float)avgval * 3.3 / 4095.0 / 6.0;

  // Same formula as the supplied ESP32 program.
  ph_act = -5.70 * volt + calibration_value;

  return ph_act;
}

float readTemperature()
{
  temperatureSensor.requestTemperatures();

  float temperature = temperatureSensor.getTempCByIndex(0);

  if (temperature == DEVICE_DISCONNECTED_C) {
    return NAN;
  }

  return temperature;
}

void setup()
{
  Wire.begin();
  Serial.begin(115200);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  pinMode(PH_SENSOR_PIN, INPUT);
  pinMode(TURBIDITY_SENSOR_PIN, INPUT);

  temperatureSensor.begin();

  Serial.println();
  Serial.println("========================================");
  Serial.println("ESP32 Smart Water Quality Monitor");
  Serial.println("pH + Turbidity + DS18B20");
  Serial.println("========================================");

}

void loop()
{
  float ph = readPH();
  turbidity_raw = analogRead(TURBIDITY_SENSOR_PIN);
  float temperature = readTemperature();

  // Machine-readable line consumed by Flask on COM3:
  // DATA,<ph>,<turbidity_adc>,<temperature>
  if (millis() - lastSend >= SEND_INTERVAL) {
    lastSend = millis();

    if (!isnan(temperature)) {
      Serial.print("DATA,");
      Serial.print(ph, 2);
      Serial.print(",");
      Serial.print(turbidity_raw);
      Serial.print(",");
      Serial.println(temperature, 2);
    } else {
      Serial.println("ERROR,TEMPERATURE_SENSOR_DISCONNECTED");
    }
  }

  delay(100);
}
