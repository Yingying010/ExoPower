import serial
import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys

# ==== Config ====
SERIAL_PORT = '/dev/tty.usbserial-110'
BAUD_RATE = 115200
FS = 500  # Hz
WINDOW_SIZE = int(0.4 * FS)
STEP_SIZE = int(0.2 * FS)

# ==== Model & Actions ====
model = joblib.load("AnalyzeData/data/trained_emg_model.pkl")
action_to_motor_cmd = {'Lifting': 'MOTOR_FORWARD', 'Resting': 'MOTOR_STOP'}
action_colors = {'Lifting': (176, 224, 230), 'Resting': (211, 211, 211)}

# ==== Filters ====
def bandpass_filter(signal, low=20, high=100, fs=500):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_avg(signal, window_size=50):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    return [mav, rms, wl, zc]

# ==== Initialize Buffers ====
buffer_emg = deque(maxlen=WINDOW_SIZE)
buffer_time = deque(maxlen=WINDOW_SIZE)
last_prediction = None
step_counter = 0

# ==== Serial Connection ====
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Serial port connected.")
except serial.SerialException:
    print("❌ Failed to connect to serial port.")
    sys.exit(1)

# ==== PyQtGraph Setup ====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-Time EMG")
plot = win.addPlot(title="EMG Signal")
plot.setYRange(-350, 350)
curve_raw = plot.plot(pen='w')
curve_smooth = plot.plot(pen='y')
text_label = pg.TextItem(anchor=(0,1))
plot.addItem(text_label)
win.show()

# ==== Real-time Update Function ====
def update():
    global step_counter, last_prediction

    while ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            if ',' not in line:
                continue
            timestamp_str, emg_str = line.split(',')
            timestamp = float(timestamp_str)
            emg_val = float(emg_str)
        except Exception:
            continue

        buffer_time.append(timestamp)
        buffer_emg.append(emg_val)

        # Prediction every STEP_SIZE
        if len(buffer_emg) >= WINDOW_SIZE and step_counter >= STEP_SIZE:
            emg_array = np.array(buffer_emg)
            emg_centered = emg_array - np.mean(emg_array)
            emg_filtered = bandpass_filter(emg_centered)
            emg_rectified = np.abs(emg_filtered)
            emg_smoothed = moving_avg(emg_rectified)

            feats = extract_features(emg_smoothed)
            pred_action = model.predict([feats])[0]

            if pred_action != last_prediction:
                last_prediction = pred_action
                cmd = action_to_motor_cmd.get(pred_action, 'MOTOR_STOP')
                ser.write((cmd + '\n').encode('utf-8'))
                print(f"[{timestamp:.2f}s] Prediction: {pred_action} | Sent CMD: {cmd}")
            step_counter = 0
        else:
            step_counter += 1

        # Plot update
        t_axis = np.linspace(0, WINDOW_SIZE / FS, WINDOW_SIZE)
        raw_data = np.array(buffer_emg)
        if len(raw_data) < WINDOW_SIZE:
            raw_data = np.pad(raw_data, (0, WINDOW_SIZE - len(raw_data)))
        curve_raw.setData(t_axis, raw_data)

        if 'emg_smoothed' in locals():
            curve_smooth.setData(t_axis, emg_smoothed[:WINDOW_SIZE])
        else:
            curve_smooth.clear()

        text_label.setText(f"Prediction: {last_prediction or '---'}", color='w')

# ==== Timer ====
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

print("🎯 Real-time EMG recognition started...")
sys.exit(app.exec_())
