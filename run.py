# emg_mqtt_aws_control.py

import json
import time
import sys
import atexit
import joblib
import numpy as np
import threading
import matplotlib.pyplot as plt
from collections import deque
from awscrt import mqtt as aws_mqtt
from awsiot import mqtt_connection_builder
import paho.mqtt.client as mqtt
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from dynamixel_sdk import *

# ==== 模型加载 ====
model = joblib.load("model/rfModel.pkl")
LABELS = ['Idle', 'Lifting']
action_colors = {'Lifting': (176, 224, 230), 'Idle': (211, 211, 211), 'unknown': (255, 255, 255)}

# ==== EMG参数 ====
FS = 500
WINDOW_SIZE = 100
STEP_SIZE = 50

# ==== Dynamixel 参数 ====
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_POSITION = 116
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

# ==== AWS IoT 配置 ====
AWS_ENDPOINT = "a1zwy6veu609mq-ats.iot.eu-west-2.amazonaws.com"
AWS_CLIENT_ID = "emg-predictor"
PATH_TO_CERT = "upload/emg02.cert.pem"
PATH_TO_KEY = "upload/emg02.private.key"
PATH_TO_ROOT = "upload/root-CA.crt"
PUBLISH_TOPIC = "emg/prediction"

mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=AWS_ENDPOINT,
    cert_filepath=PATH_TO_CERT,
    pri_key_filepath=PATH_TO_KEY,
    ca_filepath=PATH_TO_ROOT,
    client_id=AWS_CLIENT_ID,
    clean_session=False,
    keep_alive_secs=30
)

print("\ud83d\udd0c Connecting to AWS IoT...")
mqtt_connection.connect().result()
print("\u2705 AWS Connected!")

# ==== 控制电机 ====
class MotorThread(QtCore.QThread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.command_queue = deque()
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)
        self.portHandler.openPort()
        self.portHandler.setBaudRate(BAUDRATE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID, ADDR_OPERATING_MODE, 3)
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, 112, 50)
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_GOAL_POSITION, CENTER_POS)
        QtCore.QThread.msleep(1000)

    def rotate_ccw(self):
        self.command_queue.append(POS_CCW_180)

    def rotate_cw(self):
        self.command_queue.append(POS_CW_180)

    def run(self):
        while self.running:
            if self.command_queue:
                pos = self.command_queue.popleft()
                self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID, ADDR_GOAL_POSITION, pos)
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
    signal = np.array(signal)
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    sk = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**3) / np.std(signal)**3)
    ku = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**4) / np.std(signal)**4)
    return [mav, rms, wl, zc, ssc, sk, ku]

# ==== 图形界面 ====
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Real-Time EMG (MQTT + AWS)")
plot = win.addPlot(title="EMG Signal")
plot.setYRange(-100, 100)
curve_raw = plot.plot(pen='w')
text_label = pg.TextItem(anchor=(0, 1))
plot.addItem(text_label)
bg_rect = pg.LinearRegionItem([0, WINDOW_SIZE / FS], movable=False,
                              brush=pg.mkBrush(*action_colors['Idle'], 80))
plot.addItem(bg_rect)
win.show()

# ==== 全局缓存 ====
buffer_emg = deque(maxlen=WINDOW_SIZE)
state_buffer = deque(maxlen=7)
step_counter = 0
last_prediction = None
prediction_result = {'label': 'Idle', 'emg_val': 0.0}
timestamps_all, emg_all, labels_all = [], [], []

motor_thread = MotorThread()
motor_thread.start()

# ==== 数据处理 ====
def on_data_received(timestamp, emg_val):
    global step_counter, last_prediction
    buffer_emg.append(emg_val)
    timestamps_all.append(timestamp)
    emg_all.append(emg_val)
    labels_all.append(prediction_result['label'])

    curve_raw.setData(np.linspace(0, len(buffer_emg) / FS, len(buffer_emg)), np.array(buffer_emg))

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

            aws_payload = {
                "timestamp": int(time.time() * 1000),
                "emg_value": float(emg_val),
                "prediction": label.lower()
            }
            mqtt_connection.publish(
                topic=PUBLISH_TOPIC,
                payload=json.dumps(aws_payload),
                qos=aws_mqtt.QoS.AT_LEAST_ONCE
            )
            print(f"\ud83d\udce4 Prediction published to AWS: {aws_payload}")

        prediction_result['label'] = label
        step_counter = 0

    pred_label = prediction_result['label']
    text_label.setText(f"Prediction: {pred_label}\nEMG: {emg_val:.2f}", color='w')
    bg_rect.setBrush(pg.mkBrush(*action_colors.get(pred_label, (255, 255, 255)), 80))

# ==== 本地 MQTT 接收 ====
def on_local_message(client, userdata, msg):
    try:
        message = json.loads(msg.payload.decode("utf-8"))
        timestamp = float(message.get("timestamp", 0))
        emg_val = float(message.get("emg_value", 0))
        on_data_received(timestamp, emg_val)
    except Exception as e:
        print(f"\u274c MQTT message error: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_local_message
mqtt_client.connect("192.168.0.183", 1883, 60)
mqtt_client.subscribe("emg/data")
threading.Thread(target=mqtt_client.loop_forever, daemon=True).start()
print("\ud83d\udfe2 Subscribed to emg/data on local MQTT")

# ==== 程序退出释放资源 ====
@atexit.register
def shutdown():
    motor_thread.stop()
    mqtt_connection.disconnect().result()
    print("\ud83d\uded1 Closed motor & AWS MQTT.")
    if timestamps_all:
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps_all, emg_all, label='EMG Signal', color='black')
        current_label = labels_all[0]
        start_idx = 0
        for i in range(1, len(labels_all)):
            if labels_all[i] != current_label or i == len(labels_all) - 1:
                end_time = timestamps_all[i]
                color = np.array(action_colors.get(current_label, (200, 200, 200))) / 255.0
                plt.axvspan(timestamps_all[start_idx], end_time, color=color, alpha=0.2,
                            label=current_label if start_idx == 0 else None)
                current_label = labels_all[i]
                start_idx = i
        plt.xlabel('Time (ms)')
        plt.ylabel('EMG')
        plt.title('EMG Signal with Predictions')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

print("\ud83d\ude80 EMG Realtime + Motor + AWS Publishing started...")
sys.exit(app.exec_())
