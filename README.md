# SFIC-ai-fodder-optimizer-and-crop-stress-detection-biomodel-based-project-
# EdgeAgri-AI: TinyML Crop Stress Detection & Fodder Optimizer

[![Platform: Arduino Uno](https://img.shields.io/badge/Platform-Arduino%20Uno%20(ATmega328P)-blue.svg)](https://www.arduino.cc/)
[![ML: Pure C / PROGMEM](https://img.shields.io/badge/Inference-Zero--SRAM%20Bare--Metal%20C-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

EdgeAgri-AI is a fully offline, zero-dependency TinyML system providing real-time crop stress classification and livestock fodder intake optimization on an 8-bit ATmega328P microcontroller (Arduino Uno).

---

## Key Features
* **Zero External Dependencies**: Operates without TensorFlow Lite runtime overhead; all inference calculations execute in plain C using static arrays stored in Flash (`PROGMEM`).
* **Microscopic Footprint**: Consumes only **4,620 Bytes of Flash (14%)** and **268 Bytes of SRAM (13%)**, preventing stack-heap collisions.
* **Biophysically Grounded**: Trained on continuous agro-hydrological data derived from a Soil-Plant-Atmosphere Continuum (SPAC) mechanistic model.
* **Direct Hardware Actuation**: Translates real-time predictions directly into physical actions via a 3-channel relay module for automated irrigation, misting, and drainage alarms.

---

## Hardware Pinout

| Pin | Type | Connection | Function |
| :--- | :--- | :--- | :--- |
| **A0** | Analog In | Capacitive Soil Moisture Sensor v1.2 | Soil Water Availability Index |
| **A1** | Analog In | LM35 Precision Temperature Sensor | Ambient Microclimate Temp |
| **A2** | Analog In | Analog Humidity Sensor / Potentiometer | Relative Humidity % |
| **A3** | Analog In | LDR + 10 kΩ Resistor Divider | Solar Radiation Index |
| **A4** | Analog In | 10 kΩ Linear Potentiometer | Livestock Body Weight (350–700 kg) |
| **A5** | Analog In | 10 kΩ Linear Potentiometer | Milk Yield (5–35 L/day) |
| **D8** | Digital Out | Relay Channel 1 | Drip Irrigation Solenoid Valve |
| **D9** | Digital Out | Relay Channel 2 | Overhead Canopy Misting Pump |
| **D10**| Digital Out | Relay Channel 3 | Emergency Drainage / Overwatering Alarm |

---

## Directory Structure
```text
EdgeAgri-AI/
│
├── training/
│   └── EdgeAgri_Training_Pipeline.ipynb   # Complete Google Colab Biomodel & Training
│
├── firmware/
│   ├── EdgeAgriUno.ino                   # Main Arduino bare-metal loop & actuation
│   └── agri_models.h                     # Generated C-header with PROGMEM neural network weights
│
├── docs/
│   └── technical_specifications.md       # Detailed biophysical formulation
│
└── README.md