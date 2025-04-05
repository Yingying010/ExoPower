import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==== File paths ====
lifting_files = [
    'AnalyzeData/data/dataset/lifting/emg-1591098061.csv',
    'AnalyzeData/data/dataset/lifting/emg-1591119699.csv',
    'AnalyzeData/data/dataset/lifting/emg-1591119947.csv',
    'AnalyzeData/data/dataset/lifting/emg-1591120123.csv'
]
resting_files = [
    'AnalyzeData/data/dataset/reset/emg-1591011691.csv',
    'AnalyzeData/data/dataset/reset/emg-1591011824.csv',
    'AnalyzeData/data/dataset/reset/emg-1591012010.csv',
    'AnalyzeData/data/dataset/reset/emg-1591012128.csv'
]

# ==== Signal processing ====
def bandpass_filter(signal, low=20, high=450, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signal):
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    return [mav, rms, wl, zc]

# ==== Single-channel training logic ====
def process_as_independent_channels(files, label):
    features = []
    labels = []

    for file in files:
        df = pd.read_csv(file)
        emg_channels = [col for col in df.columns if col.startswith("emg")]

        for ch in emg_channels:
            signal = df[ch].values
            filtered = bandpass_filter(signal)
            rectified = np.abs(filtered)
            smoothed = np.convolve(rectified, np.ones(100)/100, mode='same')

            # Slide window
            window_size = 200
            step_size = 100
            for start in range(0, len(smoothed) - window_size, step_size):
                window = smoothed[start:start + window_size]
                feats = extract_features(window)
                features.append(feats)
                labels.append(label)

    return features, labels



# ==== Collect dataset ====
X_lift, y_lift = process_as_independent_channels(lifting_files, 'Lifting')
X_rest, y_rest = process_as_independent_channels(resting_files, 'Resting')

X = np.array(X_lift + X_rest)
y = np.array(y_lift + y_rest)

print(f"Total samples: {len(X_lift)}+{len(X_rest)}")

# ==== Train/Test split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Train model ====
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ==== Evaluation ====
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ==== Save model ====
joblib.dump(clf, 'AnalyzeData/data/trained_emg_model.pkl')
print("✅ Model saved as trained_emg_model.pkl")
