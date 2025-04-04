
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt
import pandas as pd


###
### 模拟数据转信号
###
df_raw = pd.read_csv('AnalyzeData/data/simulated_emg_data.csv')  

plt.figure(figsize=(15, 4))
plt.plot(df_raw['Time (s)'], df_raw['EMG'], label='Raw EMG Signal', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('EMG Amplitude')
plt.title('Raw EMG Signal Visualization')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

###
### 数据处理
###
df_raw = pd.read_csv('AnalyzeData/data/simulated_emg_data.csv')
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

df_raw['Smoothed'] = emg_smoothed
time_array = df_raw['Time (s)'].values
labels = df_raw['Label'].values

# 自定义映射（你设定的顺序）
custom_mapping = {
    'Idle': 0,
    'Lifting': 1,
    'Holding': 2,
    'Lowering': 3
}

df_processed = pd.DataFrame({
    'Time (s)': time_array,
    'EMG_Smoothed': emg_smoothed,
    'Labels': labels,
    'Code':pd.Series(labels).map(custom_mapping)  # 应用自定义编码
})

df_processed.to_csv('AnalyzeData/data/preprocessed.csv', index=False)

# 画图对比原始 vs 预处理后的信号
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

###
### Feature Extraction
###
df = pd.read_csv('AnalyzeData/data/preprocessed.csv')
emg_smooth = df['EMG_Smoothed'].values
time_array = df['Time (s)'].values
labels = df['Code'].values 

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
window_labels = []

for start in range(0, len(emg_smooth) - window_size, step_size):
    end = start + window_size
    window = emg_smooth[start:end]
    time_mid = time_array[start + window_size // 2]
    
    feats = extract_features(window)
    features.append(feats)
    timestamps.append(time_mid)

    window_label = pd.Series(labels[start:end]).mode()[0]
    window_labels.append(window_label)


df_features = pd.DataFrame({
    'Time (s)': timestamps,
    'MAV': [f[0] for f in features],
    'RMS': [f[1] for f in features],
    'WL': [f[2] for f in features],
    'ZC': [f[3] for f in features],
    'Label': window_labels
})

df_features.to_csv('AnalyzeData/data/emg_features.csv', index=False)



###
### Model training
###
# 1. Read Feature
df = pd.read_csv("AnalyzeData/data/emg_features.csv")
print(f"label distribution:{df['Label'].value_counts()}")

# 2. Feature & Label
X = df[['MAV', 'RMS', 'WL', 'ZC']].values
y = df['Label'].values

# 3. encoding
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print(f"Mapping of LabelEncoder:{encoder.classes_}")


# 4. Training Set Spliting
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# 6. predict & evaluation
y_pred = clf.predict(X_test)

# 7. Reporting
target_names = ['Idle', 'Lifting', 'Holding', 'Lowering']
print(classification_report(y_test, y_pred, target_names=target_names))

# 8.Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save Model
joblib.dump(clf, "AnalyzeData/data/trained_emg_model.pkl")
