import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from collections import Counter
import matplotlib.pyplot as plt

# ==== 参数设置 ====
DATA_DIR = 'signal_saved'
WINDOW_SIZE = 100
STEP_SIZE = 50
BATCH_SIZE = 64
EPOCHS = 20
FS = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== 标签映射 ====
LABEL_MAP = {'idle': 0, 'lifting': 1}

# ==== 滤波函数 ====
# def bandpass_filter(signal, low=20, high=450, fs=1000):
#     nyq = 0.5 * fs
#     b, a = butter(4, [low/nyq, high/nyq], btype='band')
#     return filtfilt(b, a, signal)

# ==== 数据加载 ====
def load_windows(file_path):
    df = pd.read_csv(file_path)
    signal = df['EMG Value'].values
    labels = df['Label'].values

    X, y = [], []
    for start in range(0, len(signal) - WINDOW_SIZE, STEP_SIZE):
        window = signal[start:start + WINDOW_SIZE]
        label_window = labels[start:start + WINDOW_SIZE]
        major_label = Counter(label_window).most_common(1)[0][0]
        if major_label in LABEL_MAP:
            X.append(window)
            y.append(LABEL_MAP[major_label])
    return X, y

all_X, all_y = [], []
for file in os.listdir(DATA_DIR):
    if file.endswith('.csv'):
        X_win, y_win = load_windows(os.path.join(DATA_DIR, file))
        all_X.extend(X_win)
        all_y.extend(y_win)

X_np = np.array(all_X)  # shape: (N, 200)
y_np = np.array(all_y)

print(f"✅ Loaded {len(X_np)} samples")

# ==== 划分数据集 ====
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# ==== PyTorch Dataset ====
class EMGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape: (N, 200, 1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = EMGDataset(X_train, y_train)
test_ds = EMGDataset(X_test, y_test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ==== LSTM 模型 ====
class EMGLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一帧输出
        return self.fc(out)

model = EMGLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== 训练循环 ====
train_accs, test_accs = [], []

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (preds.argmax(1) == yb).sum().item()
        total += len(yb)

    train_acc = correct / total
    train_accs.append(train_acc)

    # 验证集
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            correct += (preds.argmax(1) == yb).sum().item()
            total += len(yb)
    test_acc = correct / total
    test_accs.append(test_acc)

    print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

# ==== 结果可视化 ====
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM EMG Classification")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ==== 保存模型 ====
torch.save(model.state_dict(), "lstm_emg_model.pt")
print("✅ 模型已保存为 lstm_emg_model.pt")
