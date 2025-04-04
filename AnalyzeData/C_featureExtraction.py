import pandas as pd
import numpy as np

# Read preprocessed data (including labels)
df = pd.read_csv('AnalyzeData/data/preprocessed_emg.csv')
emg_smooth = df['EMG_Smoothed'].values
time_array = df['Time (s)'].values
pointwise_labels = df['Labels'].values 

# Sliding window parameters
fs = 1000
window_size = int(0.2 * fs)
step_size = int(0.1 * fs)

# Characteristic function
def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window ** 2))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    return [mav, rms, wl, zc]

# Sliding Window Processing
features = []
timestamps = []
window_labels = []

for start in range(0, len(emg_smooth) - window_size, step_size):
    end = start + window_size
    window = emg_smooth[start:end]
    time_mid = time_array[start + window_size // 2]
    
    feats = extract_features(window)
    features.append(feats)
    timestamps.append(time_mid)

    # Take the label that appears most frequently in the window as the window label
    window_label = pd.Series(pointwise_labels[start:end]).mode()[0]
    window_labels.append(window_label)


df_features = pd.DataFrame({
    'Time (s)': timestamps,
    'MAV': [f[0] for f in features],
    'RMS': [f[1] for f in features],
    'WL': [f[2] for f in features],
    'ZC': [f[3] for f in features],
    'Label': window_labels
})

# save
df_features.to_csv('AnalyzeData/data/emg_features.csv', index=False)
