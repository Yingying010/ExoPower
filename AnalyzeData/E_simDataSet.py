import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==== 参数设定 ====
fs = 1000  # 采样率（Hz）
rest_duration = 3  # s
lift_duration = 4  # s

samples_rest = rest_duration * fs
samples_lift = lift_duration * fs
total_samples = samples_rest * 2 + samples_lift

# ==== 生成模拟信号 ====

# Resting: 在 -20 ~ +20 区间内波动
rest_signal_1 = np.random.normal(0, 5, samples_rest)
rest_signal_1 = np.clip(rest_signal_1, -20, 20)

# Lifting: 增大波动范围，带周期抖动（肌肉活动）
t_lift = np.linspace(0, lift_duration, samples_lift)
lift_signal = (
    np.sin(2 * np.pi * 10 * t_lift) * 30 +  # 正弦成分
    np.random.normal(0, 30, samples_lift)   # 高幅噪声
)
lift_signal = np.clip(lift_signal, -100, 100)

# 第二段 Resting
rest_signal_2 = np.random.normal(0, 5, samples_rest)
rest_signal_2 = np.clip(rest_signal_2, -20, 20)

# 合成整段数据
signal = np.concatenate([rest_signal_1, lift_signal, rest_signal_2])
timestamps = np.arange(total_samples) / fs
labels = (
    ['Resting'] * samples_rest +
    ['Lifting'] * samples_lift +
    ['Resting'] * samples_rest
)

# 构造 DataFrame
df_simulated = pd.DataFrame({
    'Time (s)': timestamps,
    'EMG': signal,
    'Label': labels
})

# 保存
df_simulated.to_csv('AnalyzeData/data/test/testData.csv', index=False)
print("✅ Simulated EMG saved as 'testData.csv'")

# ==== 可视化 ====
plt.figure(figsize=(14, 4))
plt.plot(timestamps, signal, label='Simulated EMG Signal', linewidth=1)
plt.fill_between(timestamps, -110, 110, where=(df_simulated['Label'] == 'Lifting'), color='orange', alpha=0.2, label='Lifting')
plt.xlabel('Time (s)')
plt.ylabel('EMG Amplitude')
plt.title('Simulated EMG Signal: Resting → Lifting → Resting')
plt.ylim(-120, 120)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


