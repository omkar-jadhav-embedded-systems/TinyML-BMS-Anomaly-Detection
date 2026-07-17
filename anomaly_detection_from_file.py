import os
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Load your Excel Sheets ---
try:
    train_df = pd.read_excel('Training_data.xlsx')
    test_df = pd.read_excel('Testing_data.xlsx')
    anomaly_df = pd.read_excel('Anomalies_data.xlsx')
    print("Successfully loaded datasets.")
except FileNotFoundError:
    print("Error: Please make sure your Excel files are in this folder.")
    exit()

# --- 2. Extract and Normalize Data ---
features = ['Voltage_V', 'Current_A', 'Temperature_C']
x_train_raw = train_df[features].values
x_test_raw = test_df[features].values
x_anomaly_raw = anomaly_df[features].values

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
x_test_normal = scaler.transform(x_test_raw).astype(np.float32)
x_test_anomalous = scaler.transform(x_anomaly_raw).astype(np.float32)

print("\n--- Scaling Bounds for C++ deployment ---")
for col, min_val, max_val in zip(features, scaler.data_min_, scaler.data_max_):
    print(f"{col} -> Min: {min_val:.4f}, Max: {max_val:.4f}")

# --- 3. Build and Train the QAT Autoencoder ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='linear')
])

q_aware_model = tfmot.quantization.keras.quantize_model(model)
q_aware_model.compile(optimizer='adam', loss='mae')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

print("\nStarting Autoencoder training...")
q_aware_model.fit(
    x_train, x_train,
    epochs=1000, batch_size=16, verbose='auto',
    validation_split=0.2, callbacks=[early_stopping]
)

# --- 4. Define the Anomaly Threshold ---
train_predictions = q_aware_model.predict(x_train, verbose=0)
train_mae_loss = np.mean(np.abs(train_predictions - x_train), axis=1)

# Set threshold at the 99th percentile of normal training errors
anomaly_threshold = np.percentile(train_mae_loss, 99.58)
print(f"\n[ALERT] Tuned Anomaly Threshold calculated at: {anomaly_threshold:.4f}")

# --- 5. Export and Convert to TFLite (INT8) ---
model_path = "Anomaly_model"
tf.saved_model.save(q_aware_model, model_path)


def representative_dataset_gen():
    for input_value in x_train[::100]:
        yield [np.array(input_value, dtype=np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

tflite_model_file = pathlib.Path('lite/anomaly_model.tflite')
tflite_model_file.parent.mkdir(parents=True, exist_ok=True)
size_opti_model_bytes = tflite_model_file.write_bytes(tflite_model)

# --- Memory Footprint Comparison ---
original_size_bytes = sum(f.stat().st_size for f in pathlib.Path(model_path).rglob('*') if f.is_file())
original_size_kb = original_size_bytes / 1024.0
quantized_size_kb = size_opti_model_bytes / 1024.0
compression_ratio = ((original_size_kb - quantized_size_kb) / original_size_kb) * 100

print("\n" + "=" * 50)
print("             MEMORY FOOTPRINT COMPARISON         ")
print("=" * 50)
print(f"Original SavedModel Size  : {original_size_kb:.2f} KB")
print(f"Quantized TFLite Size     : {quantized_size_kb:.2f} KB")
print(f"Model Compression Ratio   : {compression_ratio:.2f}% Slashed! ⚡")
print("=" * 50)


# --- 6. Test the TFLite Model on Normal vs Anomalous Data ---
# OPTIMIZATION: Resizing tensor to batch-process arrays instead of iterating single samples
def tflite_predict_batch(interpreter, x_data):
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']

    interpreter.resize_tensor_input(input_idx, x_data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_idx, x_data)
    interpreter.invoke()

    return interpreter.get_tensor(output_idx)


interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

# Run predictions
pred_normal = tflite_predict_batch(interpreter, x_test_normal)
mae_normal = np.mean(np.abs(pred_normal - x_test_normal), axis=1)

pred_anomaly = tflite_predict_batch(interpreter, x_test_anomalous)
mae_anomaly = np.mean(np.abs(pred_anomaly - x_test_anomalous), axis=1)

# Calculate Evaluation Metrics
TP = np.sum(mae_anomaly > anomaly_threshold)
FN = np.sum(mae_anomaly <= anomaly_threshold)
FP_indices = np.where(mae_normal > anomaly_threshold)[0]
FP = len(FP_indices)
TN = np.sum(mae_normal <= anomaly_threshold)

accuracy = (TP + TN) / (TP + TN + FP + FN) * 100 if (TP + TN + FP + FN) > 0 else 0
precision = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
recall = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n================ TFLite Industrial-Grade Evaluation ================")
print(f"Threshold limit: {anomaly_threshold:.4f}")
print(f"Total Test Samples Evaluated: {len(x_test_normal) + len(x_test_anomalous)}")
print("-" * 68)
print(f"True Positives (Anomalies Caught)    : {TP:<3} / {len(x_test_anomalous)}")
print(f"False Negatives (Anomalies Missed)   : {FN:<3} / {len(x_test_anomalous)}")
print(f"False Positives (False Alarms)       : {FP:<3} / {len(x_test_normal)}")
print(f"True Negatives (Normal Left Alone)   : {TN:<3} / {len(x_test_normal)}")
print("-" * 68)
print(f"Overall Classification Accuracy      : {accuracy:.2f}%")
print(f"Precision (When flagged, how accurate): {precision:.2f}%")
print(f"Recall/Sensitivity (Failures caught) : {recall:.2f}%")
print(f"F1-Score (Balanced metric)           : {f1_score:.2f}%")
print("====================================================================\n")

# Print Detailed False Alarm Diagnosis
if FP > 0:
    print("=================== FALSE ALARM DIAGNOSIS ===================")
    print("These normal data points crossed the safety threshold.")
    print(f"{'Test Row #':<10} | {'Voltage (V)':<12} | {'Current (A)':<12} | {'Temp (°C)':<12} | {'Model Error':<12}")
    print("-" * 68)
    for idx in FP_indices:
        v, a, t = x_test_raw[idx]
        print(f"{idx:<10} | {v:<12.4f} | {a:<12.4f} | {t:<12.4f} | {mae_normal[idx]:<12.4f}")
    print("=============================================================\n")

h_file_path = "anomaly_model.h"
# --- 7. Generate C++ Header File ---
with open(h_file_path, "w") as f:
    f.write('// Auto-generated QAT TensorFlow Lite Anomaly Autoencoder\n\n')
    f.write('#ifndef ANOMALY_MODEL_H\n#define ANOMALY_MODEL_H\n\n')

    hex_lines = [", ".join([f"0x{b:02x}" for b in tflite_model[i:i + 12]]) for i in range(0, len(tflite_model), 12)]
    c_array = ",\n  ".join(hex_lines)

    f.write('const unsigned char anomaly_model[] = {\n  ' + c_array + '\n};\n\n')
    f.write(f'const unsigned int anomaly_model_len = {len(tflite_model)};\n\n')
    f.write('#endif // ANOMALY_MODEL_H\n')

print("C++ anomaly_model.h file successfully generated!")


# --- 8. Generate raw Intel HEX (.hex) File ---
def write_intel_hex(data, filename, start_address=0x0000):
    with open(filename, "w") as f:
        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            byte_count = len(chunk)
            address = start_address + i
            record_type = 0x00  # Data Record

            # Build checksum
            checksum = byte_count + (address >> 8) + (address & 0xFF) + record_type
            checksum += sum(chunk)
            checksum = (-checksum) & 0xFF

            # Format as: :[ByteCount][Address][RecordType][Data][Checksum]
            hex_data = "".join([f"{b:02x}" for b in chunk])
            f.write(f":{byte_count:02x}{address:04x}{record_type:02x}{hex_data}{checksum:02x}\n".upper())

        # Write End of File (EOF) Record
        f.write(":00000001FF\n")

hex_file_path = "anomaly_model.hex"
# Write the model as a .hex file starting at flash offset 0x0000
write_intel_hex(tflite_model, hex_file_path)
print("Intel HEX anomaly_model.hex file successfully generated!")

# --- 9. File Size Metrics Collection & Performance Printout ---
h_file_size_kb = os.path.getsize(h_file_path) / 1024.0
hex_file_size_kb = os.path.getsize(hex_file_path) / 1024.0

print("\n" + "=" * 50)
print("             EXPORTED EMBEDDED FILE SIZES         ")
print("=" * 50)
print(f"Raw Quantized TFLite Model : {quantized_size_kb:.2f} KB (Binary payload)")
print(f"C++ Header File (.h)       : {h_file_size_kb:.2f} KB (Text-encoded source code)")
print(f"Intel HEX File (.hex)      : {hex_file_size_kb:.2f} KB (Hardware flasher payload)")
print("=" * 50)
print("Note: The text-encoded sizes (.h and .hex) are larger on disk because")
print("they store data as human-readable hex characters. However, when compiled,")
print(f"only the raw binary payload {quantized_size_kb:.2f} KB is burned into your MCU's Flash RAM.\n")
