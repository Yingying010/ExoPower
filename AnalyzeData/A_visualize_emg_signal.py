import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load your EMG file ===
file_path = 'AnalyzeData/data/test/testData.csv'  # <- change if needed
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# === Load CSV ===
df = pd.read_csv(file_path)

# === Extract EMG channels ===
emg_channels = [col for col in df.columns if col.startswith("emg")]

# === Create Plot ===
plt.figure(figsize=(15, 10))

for i, ch in enumerate(emg_channels):
    plt.subplot(len(emg_channels), 1, i + 1)
    plt.plot(df[ch].values[:5000], label=ch)  # optional: visualize only first 5s
    plt.title(f"Raw Signal - {ch}")
    plt.ylabel("Amplitude")
    plt.grid(True)
    if i == len(emg_channels) - 1:
        plt.xlabel("Sample Index")

plt.tight_layout()
plt.suptitle("Raw EMG Signals from All Channels", fontsize=16, y=1.02)
plt.show()
