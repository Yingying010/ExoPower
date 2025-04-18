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
from dynamixel_sdk import *

# ==== RF 模型加载 ====
model = joblib.load("rfModel.pkl")
LABELS = ['Idle', 'Lifting']
action_colors = {'Lifting': (176, 224, 230), 'Idle': (211, 211, 211), 'unknown': (255, 255, 255)}

# ==== 串口 EMG 配置 ====
SERIAL_PORT = '/dev/tty.usbserial-1110'
BAUD_RATE = 115200
FS = 500
WINDOW_SIZE = 100
STEP_SIZE = 50

# ==== Dynamixel 参数 ====
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
PROTOCOL_VERSION = 2.0
DXL_ID = 1
DEVICENAME = '/dev/tty.usbserial-FT9HDB5F'
BAUDRATE = 57600
DXL_RESOLUTION = 4095 / 360.0
CENTER_POS = 2048
POS_REST = CENTER_POS
POS_CW_180 = CENTER_POS - int(60 * DXL_RESOLUTION)
POS_CCW_180 = CENTER_POS + int(60 * DXL_RESOLUTION)

# ==== MotorThread 控制类 ====
class MotorThread(QtCore.QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.lock = QtCore.QMutex()
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.portHandler.openPort()
        self.portHandler.setBaudRate(BAUDRATE)

        # 启用扭矩后归位
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_GOAL_POSITION, 2048)  # 0°
        QtCore.QThread.msleep(200)  # 等待电机回到中心位置


        # === 初始化为位置模式 ===
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_OPERATING_MODE, 3)
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, 108, 10)
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, 112, 50)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_GOAL_POSITION, CENTER_POS)
        QtCore.QThread.msleep(1000)

        self.command_queue = deque()

    def rotate_ccw(self):
        print("🔄 Lifting: rotate CCW 180°")
        self.command_queue.append(POS_CCW_180)

    def rotate_cw(self):
        print("🔁 Resting: rotate CW 180°")
        self.command_queue.append(POS_CW_180)

    def run(self):
        while self.running:
            if self.command_queue:
                goal_pos = self.command_queue.popleft()
                self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_GOAL_POSITION, goal_pos)
                QtCore.QThread.msleep(150)
            else:
                self.msleep(20)

    def stop(self):
        self.running = False
        self.wait()
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        self.portHandler.closePort()

# ==== 特征提取 ====
def extract_features(signal):
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    sk = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**3) / np.std(signal)**3)
    ku = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**4) / np.std(signal)**4)
    return [mav, rms, wl, zc, ssc, sk, ku]

# ==== 全局缓存 ====
buffer_emg = deque(maxlen=WINDOW_SIZE)
prediction_result = {'label': 'Idle', 'emg_val': 0.0}
state_buffer = deque(maxlen=7)
last_prediction = None
step_counter = 0
timestamps_all = []
emg_all = []
labels_all = []
motor_thread = MotorThread()
motor_thread.start()

# ==== 串口连接 ====
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print("✅ Serial port connected.")
except serial.SerialException:
    print("❌ Failed to connect to serial port.")
    sys.exit(1)

# ==== 实时图形界面 ====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-Time EMG (Random Forest)")
plot = win.addPlot(title="EMG Signal")
plot.setYRange(-100, 100)
curve_raw = plot.plot(pen='w')
text_label = pg.TextItem(anchor=(0,1))
plot.addItem(text_label)
bg_rect = pg.LinearRegionItem([0, WINDOW_SIZE / FS], movable=False,
                              brush=pg.mkBrush(*action_colors['Idle'], 80))
plot.addItem(bg_rect)
win.show()

# ==== 串口读取线程（Signal方式） ====
Signal = QtCore.Signal
class SerialReaderThread(QtCore.QThread):
    data_received = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.running = True  # ✅ 添加控制标志

    def run(self):
        while self.running:
            try:
                if ser and ser.in_waiting:
                    line = ser.readline().decode('utf-8').strip()
                    if ',' not in line:
                        continue
                    timestamp_str, emg_str = line.split(',')
                    timestamp = float(timestamp_str)
                    emg_val = float(emg_str)
                    self.data_received.emit(timestamp, emg_val)
            except Exception as e:
                print(f"❌ Serial read error: {e}")
                QtCore.QThread.msleep(50)  # 避免刷屏错误

    def stop(self):
        self.running = False
        self.wait()  # ✅ 等待线程完全退出


# ==== 串口数据处理函数 ====
def on_data_received(timestamp, emg_val):
    global step_counter, last_prediction
    buffer_emg.append(emg_val)
    timestamps_all.append(timestamp)
    emg_all.append(emg_val)
    labels_all.append(prediction_result['label'])

    raw_data = np.array(buffer_emg)
    t_axis = np.linspace(0, len(raw_data)/FS, len(raw_data))
    curve_raw.setData(t_axis, raw_data)

    step_counter += 1
    if len(buffer_emg) >= WINDOW_SIZE and step_counter >= STEP_SIZE:
        signal = np.array(buffer_emg)
        signal = signal - np.mean(signal)

        mav = np.mean(np.abs(signal))
        if mav < 3:
            label = 'Idle'
        else:
            features = extract_features(signal)
            pred_num = model.predict([features])[0]
            label = LABELS[pred_num]

        state_buffer.append(label)
        if state_buffer.count(label) >= 5 and label != last_prediction:
            last_prediction = label
            print(f"[{timestamp:.2f}s] Prediction: {label}")
            if label == 'Lifting':
                motor_thread.rotate_ccw()
            elif label == "Idle":
                motor_thread.rotate_cw()

        prediction_result['label'] = label
        step_counter = 0

    pred_label = prediction_result['label']
    text_label.setText(f"Prediction: {pred_label}\nEMG: {emg_val:.2f}", color='w')
    bg_rect.setBrush(pg.mkBrush(*action_colors.get(pred_label, (255,255,255)), 80))

# ==== 启动串口读取线程 ====
serial_thread = SerialReaderThread()
serial_thread.data_received.connect(on_data_received)
serial_thread.start()

# ==== 程序退出绘图 + 释放资源 ====
@atexit.register
def shutdown():
    serial_thread.stop()
    ser.close()
    motor_thread.stop()
    print("🛑 Closed serial & motor port.")
    if timestamps_all:
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps_all, emg_all, label='EMG Signal', color='black')
        current_label = labels_all[0]
        start_idx = 0
        for i in range(1, len(labels_all)):
            if labels_all[i] != current_label or i == len(labels_all)-1:
                end_time = timestamps_all[i]
                color = np.array(action_colors.get(current_label, (200,200,200))) / 255.0
                plt.axvspan(timestamps_all[start_idx], end_time, color=color, alpha=0.2,
                            label=current_label if start_idx == 0 else None)
                current_label = labels_all[i]
                start_idx = i
        plt.xlabel('Time (ms)')
        plt.ylabel('EMG')
        plt.title('EMG Signal with Motor-Controlled Prediction')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

print("🚀 Real-time EMG + MotorThread control (Multithreaded) started...")
sys.exit(app.exec_())