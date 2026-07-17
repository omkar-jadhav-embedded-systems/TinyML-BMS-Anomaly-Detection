// ====================================================================
//  FINAL VERSION: TINYML WITH PRIMED DSP FILTER
// ====================================================================

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "anomaly_model.h"

// --- Hardware Pin Definition ---
const int PIN_LED = 13;

// --- DSP FILTER CONFIGURATION ---
const int FILTER_WINDOW_SIZE = 5;
float voltageHistory[FILTER_WINDOW_SIZE];
float currentHistory[FILTER_WINDOW_SIZE];
float tempHistory[FILTER_WINDOW_SIZE];
int filterIndex = 0;
bool filterIsPrimed = false;

// --- Global variables for the model and interpreter ---
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}

// --- On-Device Scaling & Anomaly Parameters ---
const float VOLTAGE_MIN = 11.5034;
const float VOLTAGE_MAX = 13.4998;
const float CURRENT_MIN = 1.0003;
const float CURRENT_MAX = 7.4999;
const float TEMP_MIN = 20.0101;
const float TEMP_MAX = 84.9987;
const float ANOMALY_THRESHOLD = 0.0418;

// --- Function Prototypes ---
float update_filter(float newValue, float history[]);
void run_inference(float raw_data[3]);


// =====================================================================
//                             SETUP
// =====================================================================
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000);

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);

  // --- Initialize TensorFlow Lite ---
  model = tflite::GetModel(anomaly_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) { MicroPrintf("FATAL: Model schema version is not supported!"); return; }
  static tflite::MicroMutableOpResolver<4> op_resolver;
  op_resolver.AddFullyConnected(); op_resolver.AddRelu(); op_resolver.AddQuantize(); op_resolver.AddDequantize();
  static tflite::MicroInterpreter static_interpreter(model, op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) { MicroPrintf("FATAL: AllocateTensors() failed."); return; }
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("\n=============================================");
  Serial.println(" TinyML Anomaly System with DSP Noise Filter ");
  Serial.println("=============================================");

  // --- PRIME THE DSP FILTER ---
  // Before starting, we must fill the filter's history with normal data
  // to establish a stable baseline and avoid initial false alarms.
  Serial.println("\n>>> Priming DSP filter with initial normal readings...");
  float priming_data[3] = {12.1, 3.5, 45.2}; // A typical normal reading
  for (int i = 0; i < FILTER_WINDOW_SIZE; i++) {
    update_filter(priming_data[0], voltageHistory);
    update_filter(priming_data[1], currentHistory);
    update_filter(priming_data[2], tempHistory);
    filterIndex = (filterIndex + 1) % FILTER_WINDOW_SIZE; // Manually advance index
    delay(10); // Small delay
  }
  filterIsPrimed = true;
  Serial.println(">>> Filter is primed. Starting main loop.\n");
}


// =====================================================================
//                           MAIN LOOP
// =====================================================================
void loop() {
  // TEST 1: System is running normally with slight jitter
  float normal_data[3] = {12.12, 3.48, 45.15};
  run_inference(normal_data);
  delay(2000);

  // TEST 2: A single, large but temporary current spike hits the system
  float noisy_spike[3] = {12.10, 9.50, 45.20};
  run_inference(noisy_spike);
  delay(2000);

  // TEST 3: The system returns to normal after the spike
  run_inference(normal_data);
  delay(2000);

  // TEST 4: A sustained overcurrent anomaly begins
  Serial.println("\n>>> [Simulating Sustained Hardware Overcurrent (True Anomaly)] <<<");
  float sustained_anomaly[3] = {12.20, 9.50, 48.00};
  for (int i = 0; i < FILTER_WINDOW_SIZE; i++) {
    run_inference(sustained_anomaly);
    delay(2000);
  }
}


// =====================================================================
//                       Helper Functions
// =====================================================================

// --- DSP: Sliding Window Moving Average Filter ---
float update_filter(float newValue, float history[]) {
  history[filterIndex] = newValue; // Insert new value
  float sum = 0;
  for (int i = 0; i < FILTER_WINDOW_SIZE; i++) { sum += history[i]; }
  return sum / FILTER_WINDOW_SIZE; // Return the new average
}


// --- Combined DSP + ML Inference Runner ---
void run_inference(float raw_data[3]) {
  Serial.printf("\n--- New Reading ---\n");
  
  // 1. Update filter and get smoothed values
  float filtered_V = update_filter(raw_data[0], voltageHistory);
  float filtered_A = update_filter(raw_data[1], currentHistory);
  float filtered_T = update_filter(raw_data[2], tempHistory);
  
  // Advance the shared index for the next run
  filterIndex = (filterIndex + 1) % FILTER_WINDOW_SIZE;

  Serial.printf("Raw Inputs       : [V: %.2f, A: %.2f, C: %.2f]\n", raw_data[0], raw_data[1], raw_data[2]);
  Serial.printf("DSP Filter Output: [V: %.2f, A: %.2f, C: %.2f] (Smoothed)\n", filtered_V, filtered_A, filtered_T);

  // 2. Scale the filtered (smoothed) sensor inputs
  float scaled_input[3];
  scaled_input[0] = (filtered_V - VOLTAGE_MIN) / (VOLTAGE_MAX - VOLTAGE_MIN);
  scaled_input[1] = (filtered_A - CURRENT_MIN) / (CURRENT_MAX - CURRENT_MIN);
  scaled_input[2] = (filtered_T - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);

  // 3. Load into the neural network, run inference, calculate MAE
  input->data.f[0] = scaled_input[0];
  input->data.f[1] = scaled_input[1];
  input->data.f[2] = scaled_input[2];
  if (interpreter->Invoke() != kTfLiteOk) { MicroPrintf("Invoke() failed."); return; }
  float* reconstructed_output = output->data.f;
  float mae = 0.0;
  for (int i = 0; i < 3; i++) { mae += abs(reconstructed_output[i] - scaled_input[i]); }
  mae /= 3.0;
  Serial.printf("Model MAE Error  : %.4f\n", mae);

  // 4. Make Decision
  if (mae > ANOMALY_THRESHOLD) {
    Serial.println("🚨 -> VERDICT: ANOMALY DETECTED!");
    digitalWrite(PIN_LED, HIGH);
  } else {
    Serial.println("✅ -> VERDICT: System is Normal");
    digitalWrite(PIN_LED, LOW);
  }
}
