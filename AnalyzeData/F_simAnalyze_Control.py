"""
The plot background color changes according to the predicted action:
Idle 🩶
Lifting 🟦
Holding 🟩
Lowering 🩷
"""

import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

# ==== Load simulated EMG data from CSV ====
df_raw = pd.read_csv('AnalyzeData/data/test/testData.csv')
times = df_raw['Time (s)'].values
emgs = df_raw['EMG'].values

# ==== Sliding Window Parameters ====
FS = 1000  # Sampling rate
WINDOW_SIZE = int(0.2 * FS)  # 200ms window
STEP_SIZE = int(0.1 * FS)    # Update every 100ms

# ==== Label Mappings ====
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
    'Idle': '#D3D3D3',      # Gray
    'Lifting': '#B0E0E6',   # Light blue
    'Holding': '#98FB98',   # Light green
    'Lowering': '#FFB6C1'   # Light pink
}

# ==== Load the trained model ====
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

# ==== Initialize buffers ====
buffer_emg = deque(maxlen=WINDOW_SIZE)
buffer_time = deque(maxlen=WINDOW_SIZE)
last_prediction = None

# ==== Real-Time Plot Initialization ====
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line_smoothed, = ax.plot([], [], label='EMG Smoothed')
line_raw, = ax.plot([], [], label='Raw EMG', alpha=0.4)
background = ax.axvspan(0, WINDOW_SIZE / FS, color='white', alpha=0.3)
ax.set_xlim(0, WINDOW_SIZE / FS)
ax.set_ylim(-0.1, 0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('EMG')
ax.set_title('Simulated Real-time EMG Signal')
ax.grid(True)
text_label = ax.text(0.02, 0.9, '', transform=ax.transAxes)
plt.legend()
plt.tight_layout()
plt.show()

print("✅ Simulated real-time gesture recognition started...")

try:
    dt = 0.01  # Simulation step size
    step = 0
    while step < len(times):
        timestamp = float(times[step])
        emg_val = float(emgs[step])
        buffer_time.append(timestamp)
        buffer_emg.append(emg_val)

        if len(buffer_emg) >= 30:
            emg_array = np.array(buffer_emg)
            emg_centered = emg_array - np.mean(emg_array)
            emg_filtered = bandpass_filter(emg_centered)
            emg_rectified = np.abs(emg_filtered)
            emg_smoothed = moving_avg(emg_rectified)
            if len(emg_smoothed) > WINDOW_SIZE:
                emg_smoothed = emg_smoothed[:WINDOW_SIZE]
            elif len(emg_smoothed) < WINDOW_SIZE:
                emg_smoothed = np.pad(emg_smoothed, (0, WINDOW_SIZE - len(emg_smoothed)), mode='edge')

            feats = extract_features(emg_smoothed)
            pred_code = model.predict([feats])[0]
            action = code_to_action.get(pred_code, 'Unknown')

            if action != last_prediction:
                last_prediction = action
                cmd = action_to_motor_cmd.get(action, 'MOTOR_STOP')
                print(f"[{timestamp:.2f}s] Predict: {action} | Sim CMD: {cmd}")

            t_axis = np.linspace(0, WINDOW_SIZE / FS, WINDOW_SIZE)
            y_data = emg_smoothed[:len(t_axis)]
            y_raw = np.array(buffer_emg)[-WINDOW_SIZE:]
            if len(y_raw) < len(t_axis):
                y_raw = np.pad(y_raw, (0, len(t_axis) - len(y_raw)), mode='edge')

            line_smoothed.set_xdata(t_axis)
            line_smoothed.set_ydata(y_data)
            line_raw.set_xdata(t_axis)
            line_raw.set_ydata(y_raw)
            text_label.set_text(f"Prediction: {action}")

            background.remove()
            background = ax.axvspan(0, WINDOW_SIZE / FS, color=action_colors.get(action, 'white'), alpha=0.2)
            ax.set_title(f"Simulated Real-time EMG Signal [{action}]")
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

        step += 1

    plt.ioff()
    plt.show()

except KeyboardInterrupt:
    print("\n🛑 Simulated real-time recognition stopped.")
