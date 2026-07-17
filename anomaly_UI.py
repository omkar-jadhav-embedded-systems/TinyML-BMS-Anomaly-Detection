import os
import time
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Suppress TF warnings for a clean industrial dashboard
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# --- UI CONFIG ---
st.set_page_config(page_title="TinyML BMS Live Showcase", layout="wide", page_icon="🔋")


# --- CUSTOM KERAS CALLBACK FOR REAL-TIME STREAMLIT UPDATES ---
class StreamlitTrainingUpdate(tf.keras.callbacks.Callback):
    def __init__(self, chart_placeholder, text_placeholder, progress_bar, epochs):
        self.chart_placeholder = chart_placeholder
        self.text_placeholder = text_placeholder
        self.progress_bar = progress_bar
        self.epochs = epochs
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs.get('val_loss', logs['loss']))

        self.text_placeholder.markdown(
            f"**Epoch:** `{epoch + 1}/{self.epochs}` | **Train Loss:** `{logs['loss']:.4f}` | **Val Loss:** `{logs.get('val_loss', 0):.4f}`")
        self.progress_bar.progress((epoch + 1) / self.epochs)

        df = pd.DataFrame({'Train Loss': self.train_loss, 'Val Loss': self.val_loss})
        self.chart_placeholder.line_chart(df, height=300, use_container_width=True)


# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data():
    train_df = pd.read_excel('Training_data.xlsx')
    test_df = pd.read_excel('Testing_data.xlsx')
    anomaly_df = pd.read_excel('Anomalies_data.xlsx')

    features = ['Voltage_V', 'Current_A', 'Temperature_C']
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_df[features].values).astype(np.float32)
    x_test = scaler.transform(test_df[features].values).astype(np.float32)
    x_anomaly = scaler.transform(anomaly_df[features].values).astype(np.float32)

    return train_df, x_train, x_test, x_anomaly, scaler


def get_dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def tflite_predict_batch(interpreter, x_data):
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    interpreter.resize_tensor_input(input_idx, x_data.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_idx, x_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_idx)


def write_intel_hex(data, filename, start_address=0x0000):
    with open(filename, "w") as f:
        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            byte_count = len(chunk)
            address = start_address + i
            record_type = 0x00  # Data Record
            checksum = byte_count + (address >> 8) + (address & 0xFF) + record_type
            checksum += sum(chunk)
            checksum = (-checksum) & 0xFF
            hex_data = "".join([f"{b:02x}" for b in chunk])
            f.write(f":{byte_count:02x}{address:04x}{record_type:02x}{hex_data}{checksum:02x}\n".upper())
        f.write(":00000001FF\n")


# --- HEADER ---
st.title("🔋 Battery Management System: TinyML Live Demo")
st.markdown("Watch the model train live, compress itself for microcontrollers, and evaluate telemetry in real-time.")
st.divider()

# Load Data
try:
    train_df, x_train, x_test_normal, x_test_anomalous, scaler = load_data()
except Exception as e:
    st.error("Make sure your Excel files are in the same folder!")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(
    ["🚀 Phase 1: Live Training & Export", "📊 Phase 2: Industrial Evaluation", "📡 Phase 3: Live Sensor Stream"])

# ==========================================
# TAB 1: LIVE TRAINING & COMPRESSION
# ==========================================
with tab1:
    st.header("1. Live Quantization-Aware Training (QAT)")

    col1, col2 = st.columns([1, 3])
    with col1:
        epochs_input = st.number_input("Max Epochs", min_value=10, max_value=1000, value=100, step=10)
        start_training = st.button("▶️ Start Live Training", use_container_width=True, type="primary")

    with col2:
        st_text = st.empty()
        st_progress = st.empty()
        st_chart = st.empty()

    st.divider()
    metrics_placeholder = st.empty()

    if start_training:
        st.session_state['trained'] = False

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(3, activation='linear')
        ])

        q_aware_model = tfmot.quantization.keras.quantize_model(model)
        q_aware_model.compile(optimizer='adam', loss='mae')

        live_updater = StreamlitTrainingUpdate(st_chart, st_text, st_progress, epochs_input)
        # Monitored to "loss" and patience=5 to prevent running to 100 epochs unnecessarily
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        st.toast("Live Training Initiated...", icon="🚀")

        q_aware_model.fit(
            x_train, x_train,
            epochs=epochs_input, batch_size=16, verbose=0,
            validation_split=0.2, callbacks=[live_updater, early_stopping]
        )

        # Threshold
        train_preds = q_aware_model.predict(x_train, verbose=0)
        train_mae = np.mean(np.abs(train_preds - x_train), axis=1)
        threshold = np.percentile(train_mae, 99.58)  # Using your exact 99.58 percentile

        with st.spinner("Compiling Embedded Firmware Files..."):
            model_path = "Anomaly_model_tmp"
            tf.saved_model.save(q_aware_model, model_path)
            orig_size_kb = get_dir_size(model_path) / 1024.0


            def rep_dataset():
                for i in x_train[::100]: yield [np.array(i, dtype=np.float32)]


            converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            tflite_model = converter.convert()

            quant_size_kb = len(tflite_model) / 1024.0
            comp_ratio = ((orig_size_kb - quant_size_kb) / orig_size_kb) * 100

            # Write TFLite
            with open("anomaly_model.tflite", "wb") as f:
                f.write(tflite_model)

            # Generate .h file
            h_file_path = "anomaly_model.h"
            with open(h_file_path, "w") as f:
                f.write('// Auto-generated QAT TensorFlow Lite Anomaly Autoencoder\n\n')
                f.write('#ifndef ANOMALY_MODEL_H\n#define ANOMALY_MODEL_H\n\n')
                hex_lines = [", ".join([f"0x{b:02x}" for b in tflite_model[i:i + 12]]) for i in
                             range(0, len(tflite_model), 12)]
                c_array = ",\n  ".join(hex_lines)
                f.write('const unsigned char anomaly_model[] = {\n  ' + c_array + '\n};\n\n')
                f.write(f'const unsigned int anomaly_model_len = {len(tflite_model)};\n\n')
                f.write('#endif // ANOMALY_MODEL_H\n')
            h_file_size_kb = os.path.getsize(h_file_path) / 1024.0

            # Generate .hex file
            hex_file_path = "anomaly_model.hex"
            write_intel_hex(tflite_model, hex_file_path)
            hex_file_size_kb = os.path.getsize(hex_file_path) / 1024.0

        st.session_state['tflite_model'] = tflite_model
        st.session_state['threshold'] = threshold
        st.session_state['trained'] = True

        with metrics_placeholder.container():
            st.success("✅ Training & Embedded Compilation Complete!")

            # Memory Compression Metrics
            st.subheader("💾 TFLite Compression Footprint")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original TF32 Size", f"{orig_size_kb:.2f} KB")
            c2.metric("Quantized TFLite (INT8)", f"{quant_size_kb:.2f} KB", "Fits in MCU Cache!")
            c3.metric("Total Model Compression", f"{comp_ratio:.2f} %", "Size Slashed ⚡")

            st.divider()

            # File Export Sizes & Downloads
            st.subheader("📦 Exported Embedded Files")
            st.markdown(
                f"**Note:** The text-encoded sizes (`.h` and `.hex`) are larger on disk because they store data as human-readable hex characters. However, when compiled, only the raw binary payload (`{quant_size_kb:.2f} KB`) is burned into your MCU's Flash RAM.")

            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.metric("Raw .tflite Binary", f"{quant_size_kb:.2f} KB")
                with open("anomaly_model.tflite", "rb") as f:
                    st.download_button("📥 Download .tflite", f, file_name="anomaly_model.tflite",
                                       use_container_width=True)
            with ec2:
                st.metric("C++ Header (.h)", f"{h_file_size_kb:.2f} KB")
                with open(h_file_path, "rb") as f:
                    st.download_button("📥 Download .h", f, file_name="anomaly_model.h", use_container_width=True)
            with ec3:
                st.metric("Intel HEX (.hex)", f"{hex_file_size_kb:.2f} KB")
                with open(hex_file_path, "rb") as f:
                    st.download_button("📥 Download .hex", f, file_name="anomaly_model.hex", use_container_width=True)

# ==========================================
# TAB 2: INDUSTRIAL EVALUATION & ERROR MAP
# ==========================================
with tab2:
    st.header("2. Industrial-Grade Performance Evaluation")
    if 'trained' not in st.session_state:
        st.warning("⚠️ Please train the model in Phase 1 first!")
    else:
        threshold = st.session_state['threshold']
        interpreter = tf.lite.Interpreter(model_content=st.session_state['tflite_model'])
        interpreter.allocate_tensors()

        # Run TFLite Inference
        pred_norm = tflite_predict_batch(interpreter, x_test_normal)
        mae_norm = np.mean(np.abs(pred_norm - x_test_normal), axis=1)

        pred_anom = tflite_predict_batch(interpreter, x_test_anomalous)
        mae_anom = np.mean(np.abs(pred_anom - x_test_anomalous), axis=1)

        # Calculate Metrics
        TP = np.sum(mae_anom > threshold)
        FN = np.sum(mae_anom <= threshold)
        FP_indices = np.where(mae_norm > threshold)[0]
        FP = len(FP_indices)
        TN = np.sum(mae_norm <= threshold)

        total_samples = len(x_test_normal) + len(x_test_anomalous)
        acc = (TP + TN) / (TP + TN + FP + FN) * 100
        prec = TP / (TP + FP) * 100 if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) * 100 if (TP + FN) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        eval_col1, eval_col2 = st.columns([2, 3])

        with eval_col1:
            st.markdown("### 📊 Performance Report")
            st.markdown(f"""
            | Evaluation Metric | Score / Counts |
            | :--- | :--- |
            | **Threshold Limit** | `{threshold:.4f}` |
            | **Total Test Samples** | `{total_samples}` |
            | **True Positives (TP)** | `{TP} / {len(x_test_anomalous)}` |
            | **False Negatives (FN)** | `{FN} / {len(x_test_anomalous)}` |
            | **False Positives (FP)** | `{FP} / {len(x_test_normal)}` |
            | **True Negatives (TN)** | `{TN} / {len(x_test_normal)}` |
            | **Overall Accuracy** | **`{acc:.2f}%`** |
            | **Precision** | **`{prec:.2f}%`** |
            | **Recall (Sensitivity)** | **`{rec:.2f}%`** |
            | **F1-Score** | **`{f1:.2f}%`** |
            """)

        with eval_col2:
            st.markdown("### 📈 Visual Error Map")
            fig_scatter = go.Figure()

            idx_norm = np.arange(len(mae_norm))
            mask_tn = mae_norm <= threshold
            mask_fp = mae_norm > threshold

            fig_scatter.add_trace(
                go.Scatter(x=idx_norm[mask_tn], y=mae_norm[mask_tn], mode='markers', name='Normal (Safe)',
                           marker=dict(color='green', size=6)))
            fig_scatter.add_trace(
                go.Scatter(x=idx_norm[mask_fp], y=mae_norm[mask_fp], mode='markers', name='False Positives',
                           marker=dict(color='orange', size=12, symbol='x')))

            idx_anom = np.arange(len(mae_norm), len(mae_norm) + len(mae_anom))
            mask_tp = mae_anom > threshold
            mask_fn = mae_anom <= threshold

            fig_scatter.add_trace(
                go.Scatter(x=idx_anom[mask_tp], y=mae_anom[mask_tp], mode='markers', name='Anomalies Caught',
                           marker=dict(color='red', size=6)))
            fig_scatter.add_trace(
                go.Scatter(x=idx_anom[mask_fn], y=mae_anom[mask_fn], mode='markers', name='Anomalies Missed',
                           marker=dict(color='purple', size=12, symbol='x')))

            fig_scatter.add_hline(y=threshold, line_dash="dash", line_color="black",
                                  annotation_text=f"Safety Threshold ({threshold:.4f})")
            fig_scatter.update_layout(xaxis_title="Test Sample Index", yaxis_title="Model Error (MAE)", height=450,
                                      margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        # False Alarm Diagnostics
        if FP > 0:
            st.divider()
            st.subheader("⚠️ False Alarm Diagnosis")
            st.write("These normal data points crossed the safety threshold. (Usually extreme edge-cases).")

            raw_test = scaler.inverse_transform(x_test_normal)
            fa_data = []
            for idx in FP_indices:
                v, a, t = raw_test[idx]
                fa_data.append(
                    {"Row #": idx, "Voltage (V)": round(v, 4), "Current (A)": round(a, 4), "Temp (°C)": round(t, 4),
                     "Model Error": round(mae_norm[idx], 4)})

            st.table(pd.DataFrame(fa_data))

# ==========================================
# TAB 3: LIVE SENSOR STREAM
# ==========================================
with tab3:
    st.header("3. Real-Time IoT Telemetry Stream")

    if 'trained' not in st.session_state:
        st.warning("⚠️ Please train the model in Phase 1 first!")
    else:
        stream_col1, stream_col2, stream_col3 = st.columns([1, 2, 2])

        with stream_col1:
            stream_type = st.radio("Select Data Stream", ["Normal Operation", "Hardware Failures"])
            start_stream = st.button("▶️ Start Live Stream", type="primary", use_container_width=True)
            stop_stream = st.button("⏹️ Stop Stream", use_container_width=True)

        with stream_col2:
            metrics_ui = st.empty()
            gauge_ui = st.empty()

        with stream_col3:
            st.markdown("### Real-Time Error Trace")
            trace_ui = st.empty()

        if start_stream:
            data_to_stream = x_test_normal if stream_type == "Normal Operation" else x_test_anomalous
            raw_data = scaler.inverse_transform(data_to_stream)
            threshold = st.session_state['threshold']

            interpreter = tf.lite.Interpreter(model_content=st.session_state['tflite_model'])
            interpreter.allocate_tensors()

            error_history = []

            for i in range(len(data_to_stream)):
                current_input = data_to_stream[i:i + 1]
                raw_v, raw_a, raw_t = raw_data[i]

                pred = tflite_predict_batch(interpreter, current_input)
                error = float(np.mean(np.abs(pred - current_input)))
                is_anomaly = error > threshold
                error_history.append(error)

                status_color = "red" if is_anomaly else "green"
                status_text = "CRITICAL ANOMALY" if is_anomaly else "SYSTEM NORMAL"

                metrics_ui.markdown(f"""
                **Voltage:** `{raw_v:.2f} V` | **Current:** `{raw_a:.2f} A` | **Temp:** `{raw_t:.2f} °C`  
                Status: <span style='color:{status_color}; font-weight:bold; font-size:18px;'>{status_text}</span>
                """, unsafe_allow_html=True)

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=error,
                    gauge={
                        'axis': {'range': [0, threshold * 3]},
                        'bar': {'color': "darkred" if is_anomaly else "darkgreen"},
                        'steps': [{'range': [0, threshold], 'color': "lightgreen"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold}
                    }
                ))
                fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
                gauge_ui.plotly_chart(fig_gauge, use_container_width=True)

                fig_trace = go.Figure()
                plot_data = error_history[-50:]
                fig_trace.add_trace(
                    go.Scatter(y=plot_data, mode='lines+markers', line=dict(color='blue'), marker=dict(size=4)))
                fig_trace.add_hline(y=threshold, line_dash="dash", line_color="red")
                fig_trace.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10),
                                        yaxis_range=[0, max(threshold * 3, max(plot_data) * 1.2)])
                trace_ui.plotly_chart(fig_trace, use_container_width=True)

                time.sleep(0.3)
