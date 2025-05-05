import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data_dir = 'signal_saved'
window_size = 100
step_size = 50
fs = 500

label_map = {'idle': 0, 'lifting': 1}

def extract_features(signal):
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    sk = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**3) / np.std(signal)**3)
    ku = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**4) / np.std(signal)**4)
    return [mav, rms, wl, zc, ssc, sk, ku]

def load_windows(path, for_lstm=False):
    df = pd.read_csv(path)
    signal = df['EMG Value'].values
    labels = df['Label'].values
    X, y = [], []
    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start+window_size]
        label_win = labels[start:start+window_size]
        major = Counter(label_win).most_common(1)[0][0].lower()
        if major in label_map:
            if for_lstm:
                X.append(window)
            else:
                X.append(extract_features(window))
            y.append(label_map[major])
    return X, y

X_rf, y_rf = [], []
X_lstm, y_lstm = [], []
for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        path = os.path.join(data_dir, file)
        x1, y1 = load_windows(path, for_lstm=False)
        x2, y2 = load_windows(path, for_lstm=True)
        X_rf.extend(x1)
        y_rf.extend(y1)
        X_lstm.extend(x2)
        y_lstm.extend(y2)

Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
Xlstm_train, Xlstm_test, ylstm_train, ylstm_test = train_test_split(np.array(X_lstm), np.array(y_lstm), test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(Xrf_train, yrf_train)
rf_pred = rf.predict(Xrf_test)
rf_acc = accuracy_score(yrf_test, rf_pred)
print("\n✅ RF Accuracy:", rf_acc)
print(classification_report(yrf_test, rf_pred))

xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
xgb.fit(Xrf_train, yrf_train)
xgb_pred = xgb.predict(Xrf_test)
xgb_acc = accuracy_score(yrf_test, xgb_pred)
print("\n✅ XGBoost Accuracy:", xgb_acc)
print(classification_report(yrf_test, xgb_pred))

class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_ds = EMGDataset(Xlstm_train, ylstm_train)
test_ds = EMGDataset(Xlstm_test, ylstm_test)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=64)

class EMGLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = EMGLSTM().to('cpu')
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train_accs, test_accs = [], []
for epoch in range(20):
    model.train()
    correct = 0
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        correct += (pred.argmax(1) == yb).sum().item()
    train_accs.append(correct / len(train_ds))

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = model(xb)
            correct += (pred.argmax(1) == yb).sum().item()
    test_accs.append(correct / len(test_ds))
    print(f"[Epoch {epoch+1}] Train Acc: {train_accs[-1]:.3f} | Test Acc: {test_accs[-1]:.3f}")

plt.figure(figsize=(10, 6))
plt.plot(train_accs, label='LSTM Train Acc')
plt.plot(test_accs, label='LSTM Test Acc')
plt.axhline(rf_acc, color='green', linestyle='--', label='RF Test Acc')
plt.axhline(xgb_acc, color='orange', linestyle='--', label='XGB Test Acc')
plt.title("Model Comparison: LSTM vs RF vs XGBoost")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
