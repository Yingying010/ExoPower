import serial
import time
import numpy as np
from collections import deque
import threading
import queue
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import matplotlib.pyplot as plt
import sys
import atexit
import joblib

# ==== 加载 XGBoost 模型 ====
model = joblib.load("xgbModel.pkl")

# ==== 配置参数 ====
SERIAL_PORT = '/dev/tty.usbserial-1110'
BAUD_RATE = 115200
FS = 500
WINDOW_SIZE = 100
STEP_SIZE = 50

LABEL_MAP = {0: 'Idle', 1: 'Lifting'}
action_to_motor_cmd = {'Idle': 'MOTOR_STOP', 'Lifting': 'MOTOR_FORWARD'}
action_colors = {'Idle': (180, 180, 180), 'Lifting': (135, 206, 250)}

buffer_emg = deque(maxlen=WINDOW_SIZE)
buffer_time = deque(maxlen=WINDOW_SIZE)
predict_queue = queue.Queue(maxsize=1)
prediction_result = {'label': 'Idle', 'emg_val': 0.0}
state_buffer = deque(maxlen=5)
last_prediction = None
step_counter = 0

# ✅ 保存全部数据用于最终可视化
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

# ==== 串口连接 ====
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Serial port connected.")
except serial.SerialException:
    print("❌ Failed to connect to serial port.")
    sys.exit(1)

# ==== PyQtGraph 实时图设置 ====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-Time EMG (XGBoost)")
plot = win.addPlot(title="EMG Signal")
plot.setYRange(-100, 100)
curve_raw = plot.plot(pen='w')
text_label = pg.TextItem(anchor=(0,1))
plot.addItem(text_label)
bg_rect = pg.LinearRegionItem([0, WINDOW_SIZE / FS], movable=False,
                              brush=pg.mkBrush(*action_colors['Idle'], 80))
plot.addItem(bg_rect)
win.show()

# ==== 推理线程 ====
def predictor():
    global prediction_result, last_prediction
    while True:
        try:
            window, timestamp = predict_queue.get()
            signal = np.array(window) - np.mean(window)  # 中心化
            features = extract_features(signal)
            pred_num = model.predict([features])[0]
            label = LABEL_MAP[pred_num]

            state_buffer.append(label)
            if state_buffer.count(label) >= 4 and label != last_prediction:
                last_prediction = label
                cmd = action_to_motor_cmd.get(label, 'MOTOR_STOP')
                ser.write((cmd + '\n').encode('utf-8'))
                print(f"[{timestamp:.2f}s] Prediction: {label} | Sent CMD: {cmd}")
            prediction_result['label'] = label
        except Exception as e:
            print(f"❌ Predictor error: {e}")

threading.Thread(target=predictor, daemon=True).start()

# ==== 实时更新函数 ====
def update():
    global step_counter
    try:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8').strip()
            if ',' not in line:
                return
            timestamp_str, emg_str = line.split(',')
            timestamp = float(timestamp_str)
            emg_val = float(emg_str)

            buffer_time.append(timestamp)
            buffer_emg.append(emg_val)

            timestamps_all.append(timestamp)
            emg_all.append(emg_val)
            labels_all.append(prediction_result['label'])

            raw_data = np.array(buffer_emg)
            t_axis = np.linspace(0, len(raw_data)/FS, len(raw_data))
            curve_raw.setData(t_axis, raw_data)

            step_counter += 1
            if len(buffer_emg) >= WINDOW_SIZE and step_counter >= STEP_SIZE:
                if not predict_queue.full():
                    predict_queue.put((np.array(buffer_emg).copy(), timestamp))
                step_counter = 0

            pred_label = prediction_result['label']
            text_label.setText(f"Prediction: {pred_label}\nEMG: {emg_val:.2f}", color='w')
            bg_rect.setBrush(pg.mkBrush(*action_colors.get(pred_label, (255,255,255)), 80))

    except Exception as e:
        print(f"❌ Update error: {e}")

# ==== 程序结束后绘图（颜色分段） ====
@atexit.register
def show_emg_plot():
    if timestamps_all:
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps_all, emg_all, label='EMG Signal', color='black', linewidth=0.8)

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

        plt.xlabel('Time (ms)')
        plt.ylabel('EMG Value')
        plt.title('Full EMG Trace with Predicted Labels (XGBoost)')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# ==== 启动定时器 ====
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1)

print("🎯 Real-time EMG recognition started (XGBoost)...")
sys.exit(app.exec_())