import pandas as pd
import joblib 
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Read original data
df_raw = pd.read_csv('AnalyzeData/data/test/testData.csv')  

# preprocessing
emg_centered = df_raw['EMG'] - np.mean(df_raw['EMG'])

def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal)

emg_filtered = bandpass_filter(emg_centered)
emg_rectified = np.abs(emg_filtered)

def moving_avg(signal, window_size=100):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

emg_smoothed = moving_avg(emg_rectified)
time_array = df_raw['Time (s)'].values

# feature extraction
fs = 1000
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

# Constructing the feature
df_features_pred = pd.DataFrame({
    'Time (s)': timestamps,
    'MAV': [f[0] for f in features],
    'RMS': [f[1] for f in features],
    'WL': [f[2] for f in features],
    'ZC': [f[3] for f in features]
})

# Load the model and predict
model = joblib.load("AnalyzeData/data/trained_emg_model.pkl")
X_new = df_features_pred[['MAV', 'RMS', 'WL', 'ZC']].values
y_pred = model.predict(X_new)
df_features_pred['Predicted_Label'] = y_pred

# save
df_features_pred.to_csv("AnalyzeData/data/test/emg_features_with_prediction.csv", index=False)
print("✅ 动作预测完成！")
print(df_features_pred[['Time (s)', 'Predicted_Label']].head(10))

# visualisation
plt.figure(figsize=(14, 4))
plt.plot(df_features_pred['Time (s)'], df_features_pred['Predicted_Label'], drawstyle='steps-post')
plt.xlabel('Time (s)')
plt.ylabel('Predicted Label')
plt.title('Predicted Action Label Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
