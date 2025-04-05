'''
✅ 第一段代码（离线批处理方式）
这一段是你上面提到的标准预测逻辑，适用于 一次性处理整段信号数据并分析预测结果。

✅ 预测流程如下：
将整个信号预处理（滤波 & 平滑）
按照时间窗（200ms）+ 滑窗（100ms）提取每一段特征
将所有特征输入模型做一次性预测（model.predict(X)）
和 ground truth 标签作对比
可视化（整段）

📦 适合场景：
训练后的模型评估
分析EMG曲线与动作阶段对应关系
输出标签趋势图
'''
import pandas as pd
import joblib
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# ==== 模拟参数设定 ====
fs = 1000  # 采样率 Hz
segments = [
    ('Resting', 2, (-10, 10)),             # 更安静的休息状态
    ('Lifting Ramp', 1, 'ramp_up'),        # 上升过渡
    ('Lifting Hold', 2, (-250, 250)),      # 高强度保持
    ('Lowering Ramp', 1, 'ramp_down'),     # 下降过渡
    ('Resting', 1, (-10, 10))              # 返回休息
]

emg_data, labels, timestamps = [], [], []
time_pointer = 0

# ==== 构建模拟信号 ====
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

# ==== 预处理函数 ====
def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def moving_avg(signal, window_size=100):
    return np.convolve(np.abs(signal), np.ones(window_size)/window_size, mode='same')

# ==== 信号预处理 ====
emg_filtered = bandpass_filter(emgs)
emg_smoothed = moving_avg(emg_filtered)
time_array = times

# ==== 可视化模拟信号 ====
plt.figure(figsize=(14, 4))
plt.plot(df_sim['Time (s)'], df_sim['EMG'], label='Simulated EMG', linewidth=0.8)

# 添加阶段背景色
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

# ==== 特征提取 ====
window_size = int(0.2 * fs)
step_size = int(0.1 * fs)

def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    return [mav, rms, wl, zc]

features = []
timestamps = []

for start in range(0, len(emg_smoothed) - window_size, step_size):
    end = start + window_size
    window = emg_smoothed[start:end]
    time_mid = time_array[start + window_size // 2]
    
    feats = extract_features(window)
    features.append(feats)
    timestamps.append(time_mid)

# ==== 构建特征表格 ====
df_features_pred = pd.DataFrame({
    'Time (s)': timestamps,
    'MAV': [f[0] for f in features],
    'RMS': [f[1] for f in features],
    'WL': [f[2] for f in features],
    'ZC': [f[3] for f in features]
})

# ==== 真实标签分段（2–6s为Lifting，其余Resting）====
df_features_pred['True_Label'] = [
    'Resting' if t < 2 or t > 6 else 'Lifting'
    for t in df_features_pred['Time (s)']
]

# ==== 加载模型并预测 ====
model = joblib.load("AnalyzeData/data/trained_emg_model.pkl")
X_new = df_features_pred[['MAV', 'RMS', 'WL', 'ZC']].values
y_pred = model.predict(X_new)
df_features_pred['Predicted_Label'] = y_pred

# ==== 输出混淆矩阵 ====
print("\n[True vs Predicted]")
print(df_features_pred.groupby(['True_Label', 'Predicted_Label']).size())

# ==== 可视化预测 vs 真实 ====
plt.figure(figsize=(14, 4))
plt.plot(df_features_pred['Time (s)'], df_features_pred['Predicted_Label'], drawstyle='steps-post', label='Predicted')
plt.plot(df_features_pred['Time (s)'], df_features_pred['True_Label'], drawstyle='steps-post', label='True', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Label')
plt.title('Predicted vs True Action Label')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==== 特征可视化（MAV & RMS） ====
plt.figure(figsize=(12, 4))
plt.plot(df_features_pred['Time (s)'], df_features_pred['MAV'], label='MAV')
plt.plot(df_features_pred['Time (s)'], df_features_pred['RMS'], label='RMS')
plt.title("Feature Curves Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Feature Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==== 保存结果 ====
df_features_pred.to_csv("AnalyzeData/data/test/emg_features_with_prediction.csv", index=False)
print("✅ 动作预测完成并保存为 CSV")
