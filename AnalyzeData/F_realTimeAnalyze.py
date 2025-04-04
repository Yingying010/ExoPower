import serial
import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from collections import deque
import matplotlib.pyplot as plt

# ==== Serial Configuration ====
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600

# ==== Sliding Window Parameters ====
FS = 1000  # Sampling rate (Hz)
WINDOW_SIZE = int(0.2 * FS)  # 200ms window
STEP_SIZE = int(0.1 * FS)    # Update every 100ms

# ==== Action Mapping ====
code_to_action = {
    0: 'Idle',
    1: 'Lifting',
    2: 'Holding',
    3: 'Lowering'
}
action_to_motor_cmd = {
    'Lifting': 'MOTOR_FORWARD',
    'Lowering': 'MOTOR_BACKWARD',
    'Idle': 'MOTOR_STOP',
    'Holding': 'MOTOR_STOP'
}
action_colors = {
    'Idle': '#D3D3D3',
    'Lifting': '#B0E0E6',
    'Holding': '#98FB98',
    'Lowering': '#FFB6C1'
}

# ==== Load Trained Model ====
model = joblib.load("AnalyzeData/data/trained_emg_model.pkl")

# ==== Signal Processing Functions ====
def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_avg(signal, window_size=100):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    return [mav, rms, wl, zc]

# ==== Initialize Serial & Buffers ====
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Allow time for serial port to initialize

buffer_emg = deque(maxlen=WINDOW_SIZE)
buffer_time = deque(maxlen=WINDOW_SIZE)
last_prediction = None

# ==== Initialize Real-Time Plot ====
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], label='EMG Smoothed')
background = ax.axvspan(0, WINDOW_SIZE / FS, color='white', alpha=0.3)
ax.set_xlim(0, WINDOW_SIZE / FS)
ax.set_ylim(-0.1, 0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('EMG')
ax.set_title('Real-Time EMG Signal')
ax.grid(True)
text_label = ax.text(0.02, 0.9, '', transform=ax.transAxes)
plt.legend()
plt.tight_layout()

print("✅ Real-time gesture recognition started...")

try:
    while True:
        line_raw = ser.readline().decode('utf-8').strip()
        if ',' not in line_raw:
            continue

        try:
            timestamp_str, emg_str = line_raw.split(',')
            timestamp = float(timestamp_str)
            emg_val = float(emg_str)
        except ValueError:
            continue

        buffer_time.append(timestamp)
        buffer_emg.append(emg_val)

        if len(buffer_emg) == WINDOW_SIZE:
            # Signal processing: filter + rectify + smooth
            emg_array = np.array(buffer_emg)
            emg_centered = emg_array - np.mean(emg_array)
            emg_filtered = bandpass_filter(emg_centered)
            emg_rectified = np.abs(emg_filtered)
            emg_smoothed = moving_avg(emg_rectified)

            # Feature extraction + prediction
            feats = extract_features(emg_smoothed)
            pred_code = model.predict([feats])[0]
            action = code_to_action.get(pred_code, 'Unknown')

            # Motor control logic (avoid repeated commands)
            if action != last_prediction:
                last_prediction = action
                cmd = action_to_motor_cmd.get(action, 'MOTOR_STOP')
                ser.write((cmd + '\n').encode('utf-8'))
                print(f"[{timestamp:.2f}s] Action: {action} → Sent Command: {cmd}")

            # Update real-time plot
            t_axis = np.linspace(0, WINDOW_SIZE / FS, WINDOW_SIZE)
            line.set_xdata(t_axis)
            line.set_ydata(emg_smoothed[:WINDOW_SIZE])
            text_label.set_text(f"Prediction: {action}")

            # Update background color based on action
            background.remove()
            background = ax.axvspan(0, WINDOW_SIZE / FS, color=action_colors.get(action, 'white'), alpha=0.2)

            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(STEP_SIZE / FS)

except KeyboardInterrupt:
    print("\n🛑 Real-time recognition stopped.")
    ser.close()
