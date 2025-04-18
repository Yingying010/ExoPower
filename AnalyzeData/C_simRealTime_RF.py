import sys
import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from collections import deque
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt

# ==== Simulated EMG Data ====
fs = 500
segments = [
    ('Resting', 3, (-10, 10)),
    ('Lifting Ramp', 1, 'ramp_up'),
    ('Lifting Hold', 2, (-100, 100)),
    ('Lowering Ramp', 3, 'ramp_down'),
    ('Resting', 3, (-10, 10))
]
emg_data, timestamps = [], []
time_pointer = 0
for label, duration, amp in segments:
    samples = int(duration * fs)
    if isinstance(amp, tuple):
        signal = np.random.uniform(amp[0], amp[1], samples)
    elif amp == 'ramp_up':
        ramp = np.linspace(30, 100, samples)
        noise = np.random.normal(0, 1, samples)
        signal = ramp * np.random.uniform(-1, 1, samples) + noise
    elif amp == 'ramp_down':
        ramp = np.linspace(100, 30, samples)
        noise = np.random.normal(0, 1, samples)
        signal = ramp * np.random.uniform(-1, 1, samples) + noise
    emg_data.extend(signal)
    timestamps.extend(np.linspace(time_pointer, time_pointer + duration, samples, endpoint=False))
    time_pointer += duration

# ==== Parameters ====
FS = 500
WINDOW_SIZE = int(0.4 * FS)
STEP_SIZE = int(0.2 * FS)

model = joblib.load("rfModel.pkl")

LABEL_MAP = {0: 'Idle', 1: 'Lifting'}
action_to_motor_cmd = {'Idle': 'MOTOR_STOP', 'Lifting': 'MOTOR_FORWARD'}
action_colors = {'Idle': (180, 180, 180), 'Lifting': (135, 206, 250)}

buffer_emg = deque(maxlen=WINDOW_SIZE)
buffer_time = deque(maxlen=WINDOW_SIZE)
state_buffer = deque(maxlen=5)
last_prediction = None
step_counter = 0

predict_queue = queue.Queue(maxsize=1)
prediction_result = {'label': 'Idle', 'emg_val': 0.0}

# 记录所有数据用于最终可视化
timestamps_all = []
emg_all = []
labels_all = []

# ==== 特征提取函数 ====
def extract_features(signal):
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    sk = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**3) / np.std(signal)**3)
    ku = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**4) / np.std(signal)**4)
    return [mav, rms, wl, zc, ssc, sk, ku]

def bandpass_filter(signal, low=20, high=200, fs=500):
    nyq = 0.5 * fs
    high = min(high, nyq * 0.99)
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

# ==== 推理线程 ====
def predictor():
    global prediction_result, last_prediction
    while True:
        try:
            window, timestamp, emg_val = predict_queue.get()
            signal = np.array(window) - np.mean(window)

            mav = np.mean(np.abs(signal))
            if mav < 8:
                label = 'Idle'
            else:
                feats = extract_features(signal)
                pred_num = model.predict([feats])[0]
                label = LABEL_MAP[pred_num]

            state_buffer.append(label)
            if state_buffer.count(label) >= 4 and label != last_prediction:
                last_prediction = label
                cmd = action_to_motor_cmd.get(label, 'MOTOR_STOP')
                print(f"[{timestamp:.2f}s] Prediction: {label} | Sim CMD: {cmd}")
            prediction_result['label'] = label
        except Exception as e:
            print("❌ Predictor error:", e)

threading.Thread(target=predictor, daemon=True).start()

# ==== GUI ====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Simulated EMG Real-Time RF")
plot = win.addPlot(title="EMG Signal")
plot.setYRange(-200, 200)
curve_raw = plot.plot(pen='w')
text_label = pg.TextItem(anchor=(0,1))
plot.addItem(text_label)
bg_rect = pg.LinearRegionItem([0, WINDOW_SIZE / FS], movable=False,
                              brush=pg.mkBrush(*action_colors['Idle'], 80))
plot.addItem(bg_rect)
plot.setLabel("bottom", "Time", units="s")
plot.setLabel("left", "EMG")

win.show()

# ==== 更新函数 ====
step = 0
def update():
    global step_counter, step
    if step >= len(emg_data):
        timer.stop()
        show_emg_plot()
        return

    emg_val = emg_data[step]
    timestamp = timestamps[step]
    step += 1

    buffer_emg.append(emg_val)
    buffer_time.append(timestamp)
    timestamps_all.append(timestamp)
    emg_all.append(emg_val)
    labels_all.append(prediction_result['label'])

    step_counter += 1
    if len(buffer_emg) >= WINDOW_SIZE and step_counter >= STEP_SIZE:
        if not predict_queue.full():
            predict_queue.put((np.array(buffer_emg).copy(), timestamp, emg_val))
        step_counter = 0

    # 显示原始信号
    raw_data = np.array(buffer_emg)
    t_axis = np.linspace(0, len(raw_data)/FS, len(raw_data))
    curve_raw.setData(t_axis, raw_data)

    # UI更新
    label = prediction_result['label']
    text_label.setText(f"Prediction: {label}\nEMG: {emg_val:.2f}", color='w')
    bg_rect.setBrush(pg.mkBrush(*action_colors.get(label, (255,255,255)), 80))

# ==== 绘图函数 ====
def show_emg_plot():
    if timestamps_all:
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps_all, emg_all, label='EMG', color='black', linewidth=0.8)

        current_label = labels_all[0]
        start_idx = 0
        for i in range(1, len(labels_all)):
            if labels_all[i] != current_label or i == len(labels_all) - 1:
                end_time = timestamps_all[i]
                plt.axvspan(timestamps_all[start_idx], end_time,
                            color=np.array(action_colors.get(current_label, (200,200,200)))/255.0, alpha=0.2,
                            label=current_label if i == 1 else None)
                current_label = labels_all[i]
                start_idx = i

        plt.xlabel("Time (s)")
        plt.ylabel("EMG Value")
        plt.title("Simulated EMG Signal with Predicted Labels")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# ==== 启动 ====
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

print("🎯 Simulated EMG RF Real-Time Prediction Started...")
sys.exit(app.exec_())
