'''
高频采样、低频预测
'''
import time
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

# ==== 参数设定 ====
fs = 1000  # 采样率 Hz
segments = [
    ('Resting', 1, (-10, 10)),             # 0–2s 低幅值噪声
    ('Lifting Ramp', 1, 'ramp_up'),        # 2–3s 逐渐增强
    ('Lifting Hold', 1, (-250, 250)),      # 3–5s 保持高幅值
    ('Lowering Ramp', 1, 'ramp_down'),     # 5–6s 逐渐减弱
    ('Resting', 1, (-10, 10))              # 6–7s 再次回到休息状态
]

emg_data, labels, timestamps = [], [], []
time_pointer = 0

# ==== 构建信号 ====
for label, duration, amp in segments:
    samples = int(duration * fs)
    t = np.linspace(0, duration, samples, endpoint=False)

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
    else:
        signal = np.zeros(samples)

    emg_data.extend(signal)
    labels.extend([label] * samples)
    timestamps.extend(np.linspace(time_pointer, time_pointer + duration, samples, endpoint=False))
    time_pointer += duration

# ==== 转为 DataFrame ====
df_sim = pd.DataFrame({'Time (s)': timestamps, 'EMG': emg_data, 'Label': labels})
times = df_sim['Time (s)'].values
emgs = df_sim['EMG'].values

# ==== 可视化 ====
plt.figure(figsize=(14, 4))
plt.plot(df_sim['Time (s)'], df_sim['EMG'], label='Simulated EMG', linewidth=0.8)

# 添加背景色标注
color_map = {
    'Resting': '#D3D3D3',
    'Lifting Ramp': '#ADD8E6',
    'Lifting Hold': '#87CEFA',
    'Lowering Ramp': '#B0C4DE'
}
start = 0
for label, duration, _ in segments:
    end = start + duration
    plt.axvspan(start, end, color=color_map.get(label, 'white'), alpha=0.2)
    plt.text((start + end)/2, max(emg_data)*0.8, label, ha='center', va='top')
    start = end

plt.xlabel('Time (s)')
plt.ylabel('EMG Amplitude')
plt.title('Simulated EMG Signal: Rest → Lifting → Rest')
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== Parameters ====
FS = 1000
WINDOW_SIZE = int(0.2 * FS)
STEP_SIZE = int(0.1 * FS)

# ==== Mappings ====
action_to_motor_cmd = {'Lifting': 'MOTOR_FORWARD', 'Resting': 'MOTOR_STOP'}
action_colors = {'Lifting': '#B0E0E6', 'Resting': '#D3D3D3'}

# ==== Signal Processing ====
def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_avg(signal, window_size=100):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    return [mav, rms, wl, zc]

# ==== Load Model ====
model = joblib.load("AnalyzeData/data/trained_emg_model.pkl")

# ==== Buffers ====
buffer_emg = deque(maxlen=WINDOW_SIZE)
buffer_time = deque(maxlen=WINDOW_SIZE)
last_prediction = None
step_size_counter = 0

# ==== Plotting Setup ====
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))
line_smoothed, = ax.plot([], [], label='EMG Smoothed')
line_raw, = ax.plot([], [], label='Raw EMG', alpha=0.4)
background = ax.axvspan(0, WINDOW_SIZE / FS, color='white', alpha=0.3)
ax.set_xlim(0, WINDOW_SIZE / FS)
ax.set_ylim(-350, 350)
ax.set_xlabel('Time (s)')
ax.set_ylabel('EMG')
ax.set_title('Simulated Real-time EMG Signal')
ax.grid(True)
text_label = ax.text(0.02, 0.9, '', transform=ax.transAxes)
text_emg = ax.text(0.98, 0.9, '', transform=ax.transAxes, ha='right')  # 右上角：EMG值
text_time = ax.text(0.98, 0.85, '', transform=ax.transAxes, ha='right')  # 显示当前时间
plt.legend()
plt.tight_layout()
plt.show()

print("✅ Simulated real-time lifting/resting recognition started...")

# ==== Real-time Simulation ====
try:
    dt = 0.001
    step = 0
    action = "unknown"
    while step < len(times):
        timestamp = float(times[step])
        emg_val = float(emgs[step])
        buffer_time.append(timestamp)
        buffer_emg.append(emg_val)

        # 每100ms执行一次预测
        if len(buffer_emg) >= WINDOW_SIZE and step_size_counter >= STEP_SIZE:
            emg_array = np.array(buffer_emg)
            emg_centered = emg_array - np.mean(emg_array)
            emg_filtered = bandpass_filter(emg_centered)
            emg_rectified = np.abs(emg_filtered)
            emg_smoothed = moving_avg(emg_rectified)

            emg_smoothed = emg_smoothed[:WINDOW_SIZE] if len(emg_smoothed) > WINDOW_SIZE else np.pad(emg_smoothed, (0, WINDOW_SIZE - len(emg_smoothed)), mode='edge')

            feats = extract_features(emg_smoothed)
            pred_code = model.predict([feats])[0]
            action = pred_code

            # Print command on state change
            if action != last_prediction:
                last_prediction = action
                cmd = action_to_motor_cmd.get(action, 'MOTOR_STOP')
                print(f"[{timestamp:.2f}s] Predict: {action} | Sim CMD: {cmd}")
            step_size_counter = 0  # 重置预测间隔

        # 每1ms更新图像
        t_axis = np.linspace(0, WINDOW_SIZE / FS, WINDOW_SIZE)
        if 'emg_smoothed' in locals():
            line_smoothed.set_xdata(t_axis)
            line_smoothed.set_ydata(emg_smoothed)

        y_raw = np.array(buffer_emg)
        y_raw = y_raw[-WINDOW_SIZE:] if len(y_raw) >= WINDOW_SIZE else np.pad(y_raw, (0, WINDOW_SIZE - len(y_raw)), mode='edge')
        line_raw.set_xdata(t_axis)
        line_raw.set_ydata(y_raw)

        text_label.set_text(f"Prediction: {action}")
        text_emg.set_text(f"EMG: {emg_val:.1f}")
        text_time.set_text(f"Time: {timestamp:.2f}s")

        background.remove()
        background = ax.axvspan(0, WINDOW_SIZE / FS, color=action_colors.get(action, 'white'), alpha=0.2)
        ax.set_title(f"Simulated Real-time EMG Signal [{action}]")

        fig.canvas.draw()
        fig.canvas.flush_events()

        step += 1
        step_size_counter += 1
        time.sleep(dt)

    plt.ioff()
    plt.show()
except KeyboardInterrupt:
    print("\n🛑 Simulated real-time recognition stopped.")
