
from matplotlib import pyplot as plt

import pandas as pd

df_raw = pd.read_csv('AnalyzeData/data/simulated_emg_data.csv')  

# Plot the raw EMG signal from the original file (without labels)
plt.figure(figsize=(15, 4))
plt.plot(df_raw['Time (s)'], df_raw['EMG'], label='Raw EMG Signal', linewidth=0.8)
plt.xlabel('Time (s)')
plt.ylabel('EMG Amplitude')
plt.title('Raw EMG Signal Visualization')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
