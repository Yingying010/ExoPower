import sys
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

# === Generate Simulated EMG Data ===
fs = 1000
segments = [
    ('Resting', 5, (-10, 10)),
    ('Lifting Ramp', 1, 'ramp_up'),
    ('Lifting Hold', 1, (-250, 250)),
    ('Lowering Ramp', 1, 'ramp_down'),
    ('Resting', 5, (-10, 10))
]

emg_data, labels, timestamps = [], [], []
time_pointer = 0
for label, duration, amp in segments:
    samples = int(duration * fs)
    if isinstance(amp, tuple):
        signal = np.random.uniform(amp[0], amp[1], samples)
    elif amp == 'ramp_up':
        ramp = np.linspace(30, 250, samples)
        noise = np.random.normal(0, 1, samples)
        signal = ramp * np.random.uniform(-1, 1, samples) + noise
    elif amp == 'ramp_down':
        ramp = np.linspace(250, 30, samples)
        noise = np.random.normal(0, 1, samples)
        signal = ramp * np.random.uniform(-1, 1, samples) + noise
    emg_data.extend(signal)
    labels.extend([label] * samples)
    timestamps.extend(np.linspace(time_pointer, time_pointer + duration, samples, endpoint=False))
    time_pointer += duration

# === Parameters ===
FS = 1000
WINDOW_SIZE = int(0.2 * FS)
STEP_SIZE = int(0.1 * FS)

# === Model and Color Mapping ===
model = joblib.load("AnalyzeData/data/trained_emg_model.pkl")
action_to_motor_cmd = {'Lifting': 'MOTOR_FORWARD', 'Resting': 'MOTOR_STOP'}
action_colors = {'Lifting': (176, 224, 230), 'Resting': (211, 211, 211), 'unknown': (255, 255, 255)}

# === Signal Processing ===
def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_avg(signal, window_size=100):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    return [mav, rms, wl, zc]

# === Buffers ===
buffer_emg = deque(maxlen=WINDOW_SIZE)
buffer_time = deque(maxlen=WINDOW_SIZE)
last_prediction = None
step_counter = 0
state_buffer = deque(maxlen=5)  # ✅ 投票机制缓冲区

# === PyQtGraph Setup ===
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Simulated Real-Time EMG")
plot = win.addPlot(title="EMG Signal")
plot.setYRange(-350, 350)
curve_raw = plot.plot(pen='w')
curve_smooth = plot.plot(pen='y')
text_label = pg.TextItem(anchor=(0,1))
plot.addItem(text_label)
bg_rect = pg.LinearRegionItem([0, WINDOW_SIZE / FS], movable=False,
                              brush=pg.mkBrush(*action_colors['unknown'], 80))
plot.addItem(bg_rect)
win.show()

# === Simulated Data Source ===
times = np.array(timestamps)
emgs = np.array(emg_data)
step = 0

# === Real-Time Update ===
def update():
    global step, step_counter, last_prediction

    if step >= len(times):
        timer.stop()
        return

    timestamp = times[step]
    emg_val = emgs[step]
    step += 1

    buffer_time.append(timestamp)
    buffer_emg.append(emg_val)

    if len(buffer_emg) >= WINDOW_SIZE and step_counter >= STEP_SIZE:
        emg_array = np.array(buffer_emg)
        emg_centered = emg_array - np.mean(emg_array)
        emg_filtered = bandpass_filter(emg_centered)
        emg_rectified = np.abs(emg_filtered)
        emg_smoothed = moving_avg(emg_rectified)

        feats = extract_features(emg_smoothed)
        pred_action = model.predict([feats])[0]

        # ✅ 状态预测缓冲判断
        state_buffer.append(pred_action)
        if state_buffer.count(pred_action) >= 4 and pred_action != last_prediction:
            last_prediction = pred_action
            cmd = action_to_motor_cmd.get(pred_action, 'MOTOR_STOP')
            print(f"[{timestamp:.2f}s] Prediction: {pred_action} | Sim CMD: {cmd}")

        step_counter = 0
    else:
        step_counter += 1

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
    bg_rect.setBrush(pg.mkBrush(*action_colors.get(last_prediction, (255,255,255)), 80))

# === Start Timer ===
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

print("✅ Simulated real-time EMG prediction with voting started.")
sys.exit(app.exec_())
