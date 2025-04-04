import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
import pandas as pd

df_raw = pd.read_csv('AnalyzeData/data/simulated_emg_data.csv')

# Step 1: DC offset removal
emg_centered = df_raw['EMG'] - np.mean(df_raw['EMG'])

# Step 2: Bandpass filter (20-450 Hz)
def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

emg_filtered = bandpass_filter(emg_centered)

# Step 3: Absolute value rectification
emg_rectified = np.abs(emg_filtered)

# Step 4: Smoothing (sliding average, window size 100 points ≈ 100ms)
def moving_avg(signal, window_size=100):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

emg_smoothed = moving_avg(emg_rectified)


# df_raw['Filtered'] = emg_filtered
df_raw['Smoothed'] = emg_smoothed
time_array = df_raw['Time (s)'].values
labels = df_raw['Label'].values

df_processed = pd.DataFrame({
    'Time (s)': time_array,
    'EMG_Smoothed': emg_smoothed,
    'Labels': labels
})

df_processed.to_csv('AnalyzeData/data/preprocessed_emg.csv', index=False)

# Plotting the original vs preprocessed signal
plt.figure(figsize=(15, 5))
plt.plot(df_raw['Time (s)'], df_raw['EMG'], label='Raw EMG', alpha=0.4)
plt.plot(df_raw['Time (s)'], df_raw['Smoothed'], label='Preprocessed EMG', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('EMG Signal: Raw vs Preprocessed')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()