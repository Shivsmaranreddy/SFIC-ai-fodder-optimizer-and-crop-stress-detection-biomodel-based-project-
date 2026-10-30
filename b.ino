#include "esp_camera.h"
#include "Arduino.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "model_data.h" // Your exported hexadecimal array header

// --- Hardware Pin Configurations ---
const int TRIG_PIN = 12; 
const int ECHO_PIN = 13;

// --- Camera Pinout (AI-Thinker ESP32-CAM configuration) ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// --- Ultrasonic Logic Parameters ---
float distanceHistory[5] = {200.0, 200.0, 200.0, 200.0, 200.0};
int historyIndex = 0;
unsigned long lastAnnouncementTime = 0;
const unsigned long ANNOUNCEMENT_COOLDOWN = 4000; 

// --- Labels (Must match Python class_names order perfectly) ---
const char* LABELS[] = {"chair", "table", "door"}; 

// --- TFLite Global Variables ---
const int tensor_arena_size = 60 * 1024; // Allocate 60KB internal memory arena
uint8_t tensor_arena[tensor_arena_size];
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
  Serial.begin(115200);
  
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  // Initialize ESP32-CAM Camera Settings
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_GRAYSCALE; 
  config.frame_size = FRAMESIZE_96X96; // 96x96 matching Python setup
  config.jpeg_quality = 12;
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera Init Failed!");
    while(1);
  }

  // Load the model from model_data.h array
  model = tflite::GetModel(assistive_device_model_tflite);
  
  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX, tflite::ops::micro::Register_SOFTMAX());

  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, tensor_arena_size);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

float getFilteredDistance() {
  digitalWrite(TRIG_PIN, LOW); delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  
  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return 400.0;
  
  float rawDistance = (duration * 0.0343) / 2.0;
  distanceHistory[historyIndex] = rawDistance;
  historyIndex = (historyIndex + 1) % 5;
  
  float sum = 0;
  for(int i=0; i<5; i++) sum += distanceHistory[i];
  return sum / 5.0;
}

void loop() {
  float distance = getFilteredDistance();
  unsigned long currentTime = millis();

  if (distance < 50.0) {
    Serial.println("ALERT:STOP"); // Immediate buzzer override
    delay(200);
    return;
  }

  if (distance >= 50.0 && distance <= 200.0) {
    if (currentTime - lastAnnouncementTime > ANNOUNCEMENT_COOLDOWN) {
      
      camera_fb_t * fb = esp_camera_fb_get();
      if (!fb) return;

      // Feed pixels into the CNN input tensor block
      for (int i = 0; i < input->bytes; i++) {
        // Convert grayscale pixel byte (0-255) to INT8 precision representation (-128 to 127)
        input->data.int8[i] = (int8_t)((int)fb->buf[i] - 128);
      }
      esp_camera_fb_return(fb);

      // Run Inference
      if (interpreter->Invoke() == kTfLiteOk) {
        int best_idx = 0;
        int8_t max_score = -128;
        
        for (int i = 0; i < output->dims->data[1]; i++) {
          if (output->data.int8[i] > max_score) {
            max_score = output->data.int8[i];
            best_idx = i;
          }
        }

        // Output command over Serial to audio driver shield 
        if (max_score > 0) { // Check confidence threshold
          Serial.print("SPEAK:");
          Serial.print(LABELS[best_idx]);
          Serial.print(" ahead at ");
          Serial.print(distance / 100.0, 1);
          Serial.println(" meters.");
          lastAnnouncementTime = currentTime;
        }
      }
    }
  }
  delay(100);
}
