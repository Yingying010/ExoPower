import sys
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from collections import deque
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from dynamixel_sdk import *

# === Generate Simulated EMG Data ===
fs = 1000
segments = [
    ('Resting', 1, (-10, 10)),
    ('Lifting Ramp', 0.2, 'ramp_up'),
    ('Lifting Hold', 0.5, (-250, 250)),
    ('Lowering Ramp', 0.2, 'ramp_down'),
    ('Resting', 1, (-10, 10))
]
emg_data, timestamps = [], []
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
    timestamps.extend(np.linspace(time_pointer, time_pointer + duration, samples, endpoint=False))
    time_pointer += duration

# === Parameters ===
FS = 1000
WINDOW_SIZE = int(0.2 * FS)
STEP_SIZE = int(0.1 * FS)

# === Dynamixel Setup Constants ===
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
PROTOCOL_VERSION = 2.0
DXL_ID = 1
DEVICENAME = '/dev/tty.usbserial-FT9HDB5F'
BAUDRATE = 57600
DXL_RESOLUTION = 4095 / 360.0
CENTER_POS = 2048
POS_CW_180 = CENTER_POS - int(60 * DXL_RESOLUTION)
POS_CCW_180 = CENTER_POS + int(60 * DXL_RESOLUTION)

# === Motor Worker Thread (Position Mode) ===
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

# === Signal Processing ===
def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_avg(signal, window_size=100):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def extract_features(signal):
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    sk = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**3) / np.std(signal)**3)
    ku = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**4) / np.std(signal)**4)
    return [mav, rms, wl, zc, ssc, sk, ku]

# === Model and Color Mapping ===
model = joblib.load("model/rfModel.pkl")
action_colors = {'Lifting': (176, 224, 230), 'Resting': (211, 211, 211), 'unknown': (255, 255, 255)}

# === PyQtGraph Setup ===
class EMGWindow(pg.GraphicsLayoutWidget):
    def closeEvent(self, event):
        print("❗窗口关闭，停止电机线程")
        motor_thread.stop()
        event.accept()

app = QtWidgets.QApplication([])
win = EMGWindow(title="EMG Real-Time with QThread Motor")
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

# === Buffers and State ===
buffer_emg = deque(maxlen=WINDOW_SIZE)
step_counter = 0
state_buffer = deque(maxlen=5)
last_prediction = None
motor_thread = MotorThread()
motor_thread.start()
step = 0
emgs = np.array(emg_data)
times = np.array(timestamps)

# === Real-Time Update ===
def update():
    global step, step_counter, last_prediction

    if step >= len(times):
        timer.stop()
        motor_thread.stop()
        return

    timestamp = times[step]
    emg_val = emgs[step]
    step += 1
    buffer_emg.append(emg_val)

    if len(buffer_emg) >= WINDOW_SIZE and step_counter >= STEP_SIZE:
        emg_array = np.array(buffer_emg)
        emg_centered = emg_array - np.mean(emg_array)
        emg_filtered = bandpass_filter(emg_centered)
        emg_rectified = np.abs(emg_filtered)
        emg_smoothed = moving_avg(emg_rectified)
        feats = extract_features(emg_smoothed)
        label_map = {0: 'Resting', 1: 'Lifting'}
        pred_action = label_map[model.predict([feats])[0]]

        state_buffer.append(pred_action)

        if state_buffer.count(pred_action) >= 3 and pred_action != last_prediction:
            last_prediction = pred_action
            print(f"[{timestamp:.2f}s] Prediction: {pred_action}")
            if pred_action == "Lifting":
                motor_thread.rotate_ccw()
            elif pred_action == "Resting":
                motor_thread.rotate_cw()

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
print("✅ QThread-based motor control started.")
sys.exit(app.exec_())
