import os
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import Counter
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier

# ==== 配置 ====
data_dir = 'signal_saved'
window_size = 100
step_size = 50
fs = 500

# ==== 滤波器函数 ====
# def bandpass_filter(signal, low=20, high=200, fs=500):
#     nyq = 0.5 * fs
#     high = min(high, nyq * 0.99)
#     b, a = butter(4, [low / nyq, high / nyq], btype='band')
#     return filtfilt(b, a, signal)

# ==== 特征提取函数 ====
def extract_features(signal):
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    sk = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**3) / np.std(signal)**3)
    ku = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**4) / np.std(signal)**4)
    return [mav, rms, wl, zc, ssc, sk, ku]

# ==== 处理单个文件 ====
def process_file(filepath):
    df = pd.read_csv(filepath)
    signal = df['EMG Value'].values
    labels = df['Label'].values
    features, window_labels = [], []

    for start in range(0, len(signal) - window_size, step_size):
        win_signal = signal[start:start + window_size]
        win_label = labels[start + int(0.1*window_size):start + int(0.9*window_size)]
        feats = extract_features(win_signal)
        label = Counter(win_label).most_common(1)[0][0]
        features.append(feats)
        window_labels.append(label)

    return features, window_labels

# ==== 读取数据 ====
all_features, all_labels = [], []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        f_path = os.path.join(data_dir, file)
        print(f"📂 正在处理：{f_path}")
        feats, labels = process_file(f_path)
        all_features.extend(feats)
        all_labels.extend(labels)

X = np.array(all_features)
y_str = np.array(all_labels)
print(f"✅ 读取完毕：共 {len(X)} 个窗口样本")

# ==== 标签编码 ====
le = LabelEncoder()
y = le.fit_transform(y_str)
joblib.dump(le, 'label_encoder.pkl')

# ==== 训练 RandomForest 模型 ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
clf.fit(X_train, y_train)

# ==== 模型评估 ====
y_pred = clf.predict(X_test)
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

cm = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred), labels=le.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.show()

# ==== 保存模型 ====
joblib.dump(clf, 'rfModel.pkl')
print("✅ 模型已保存为 rfModel.pkl")
