#include "agri_models.h"

// -------------------------------------------------------------
// HARDWARE PIN CONFIGURATION
// -------------------------------------------------------------
const int PIN_SOIL       = A0; // Capacitive Soil Moisture (Analog)
const int PIN_TEMP       = A1; // LM35 Linear Temp Sensor (Analog)
const int PIN_HUM        = A2; // Humidity Sensor / Potentiometer
const int PIN_LDR        = A3; // Light Dependent Resistor Divider
const int PIN_COW_WEIGHT = A4; // Potentiometer 1: Cow Weight Calibration
const int PIN_COW_MILK   = A5; // Potentiometer 2: Milk Yield Calibration

const int RELAY_IRRIGATION = 8;  // Relay 1: Drip Irrigation Valve
const int RELAY_MISTING    = 9;  // Relay 2: Overhead Canopy Misters
const int RELAY_ALARM      = 10; // Relay 3: System Warning Buzzer

// -------------------------------------------------------------
// SETUP
// -------------------------------------------------------------
void setup() {
    Serial.begin(9600);
    while(!Serial);

    // Initialize Actuator Outputs (Active LOW Configuration)
    pinMode(RELAY_IRRIGATION, OUTPUT);
    pinMode(RELAY_MISTING, OUTPUT);
    pinMode(RELAY_ALARM, OUTPUT);

    digitalWrite(RELAY_IRRIGATION, HIGH); // OFF
    digitalWrite(RELAY_MISTING, HIGH);    // OFF
    digitalWrite(RELAY_ALARM, HIGH);      // OFF

    Serial.println(F("=================================================="));
    Serial.println(F(" EdgeAgri-AI: Bare-Metal TinyML System Initialized"));
    Serial.println(F(" Engine: Dual MLP Running in Flash (PROGMEM)"));
    Serial.println(F("=================================================="));
}

// -------------------------------------------------------------
// MAIN INFERENCE & ACTUATION LOOP
// -------------------------------------------------------------
void loop() {
    // 1. DATA ACQUISITION & PHYSICAL CALIBRATION
    float raw_soil = analogRead(PIN_SOIL);
    float soil_pct = (1.0f - (raw_soil / 1023.0f)) * 100.0f;
    float temp_c   = (analogRead(PIN_TEMP) * (5.0f / 1023.0f)) * 100.0f;
    float hum_pct  = (analogRead(PIN_HUM) / 1023.0f) * 100.0f;
    float light    = (analogRead(PIN_LDR) / 1023.0f) * 100.0f;

    float cow_weight = map(analogRead(PIN_COW_WEIGHT), 0, 1023, 350, 700);
    float milk_yield = map(analogRead(PIN_COW_MILK), 0, 1023, 5, 35);

    // Dynamic nutritional quality estimates (tied to environmental baseline)
    float forage_ndf = 46.5f;
    float forage_cp  = 14.2f;

    // 2. MODEL 1: CROP STRESS NEURAL NETWORK INFERENCE
    float crop_inputs[4] = {soil_pct, temp_c, hum_pct, light};
    int stress_class = predict_crop_stress(crop_inputs);

    // 3. MODEL 2: FODDER DMI NEURAL NETWORK INFERENCE
    float fodder_inputs[4] = {cow_weight, milk_yield, forage_ndf, forage_cp};
    float predicted_dmi = predict_fodder_dmi(fodder_inputs);

    // Feed ration distribution
    float concentrate_kg = predicted_dmi * 0.40f;
    float green_fodder_kg = (predicted_dmi * 0.60f) * 0.70f;
    float dry_fodder_kg   = (predicted_dmi * 0.60f) * 0.30f;

    // 4. PRINT TELEMETRY TO SERIAL DASHBOARD
    Serial.println(F("\n--- [FIELD SENSOR TELEMETRY] ---"));
    Serial.print(F("Soil: ")); Serial.print(soil_pct, 1);
    Serial.print(F("% | Air Temp: ")); Serial.print(temp_c, 1);
    Serial.print(F("C | Humidity: ")); Serial.print(hum_pct, 1);
    Serial.print(F("% | Light Index: ")); Serial.println(light, 1);

    Serial.println(F("--- [AI DIAGNOSTIC & ACTUATION] ---"));
    switch(stress_class) {
        case 0:
            Serial.println(F("STATUS: HEALTHY -> Normal Conditions. Actuators Idle."));
            digitalWrite(RELAY_IRRIGATION, HIGH);
            digitalWrite(RELAY_MISTING, HIGH);
            digitalWrite(RELAY_ALARM, HIGH);
            break;
        case 1:
            Serial.println(F("STATUS: DROUGHT STRESS -> Actuating Drip Irrigation Valve [D8]"));
            digitalWrite(RELAY_IRRIGATION, LOW);  // Valve ON
            digitalWrite(RELAY_MISTING, HIGH);
            digitalWrite(RELAY_ALARM, HIGH);
            break;
        case 2:
            Serial.println(F("STATUS: HEAT/RAD STRESS -> Actuating Misting Sprinklers [D9]"));
            digitalWrite(RELAY_IRRIGATION, HIGH);
            digitalWrite(RELAY_MISTING, LOW);     // Pump ON
            digitalWrite(RELAY_ALARM, HIGH);
            break;
        case 3:
            Serial.println(F("STATUS: OVERWATERED / ANOXIA -> Triggering System Alarm [D10]"));
            digitalWrite(RELAY_IRRIGATION, HIGH); // Force Valve OFF
            digitalWrite(RELAY_MISTING, HIGH);
            digitalWrite(RELAY_ALARM, LOW);       // Alarm ON
            break;
    }

    Serial.println(F("--- [FODDER RATION FORMULATION] ---"));
    Serial.print(F("Target Cow: ")); Serial.print(cow_weight, 0);
    Serial.print(F(" kg | Yield: ")); Serial.print(milk_yield, 1); Serial.println(F(" L/day"));
    Serial.print(F("Predicted DMI Demand: ")); Serial.print(predicted_dmi, 2); Serial.println(F(" kg DM/day"));
    Serial.print(F("  - Concentrate Mix:   ")); Serial.print(concentrate_kg, 2); Serial.println(F(" kg"));
    Serial.print(F("  - Fresh Green Chop:  ")); Serial.print(green_fodder_kg, 2); Serial.println(F(" kg"));
    Serial.print(F("  - Dry Straw / Stover:")); Serial.print(dry_fodder_kg, 2); Serial.println(F(" kg"));

    // 4-second inference loop
    delay(4000);
}