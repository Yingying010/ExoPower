import serial
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt
from pynput import keyboard
import os


PORT = '/dev/tty.usbserial-1110'
BAUD = 115200
SAVE_FOLDER = os.path.join(os.getcwd(), 'signal_saved')

labels = ['idle', 'lifting']
label_index = 0
recording = True

def on_press(key):
    global label_index, recording
    try:
        if key == keyboard.Key.space:
            label_index = (label_index + 1) % len(labels)
            print(f"⚡ Change Labels: {labels[label_index]}")
        elif key.char == 'q':
            recording = False
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

ser = serial.Serial(PORT, BAUD)
time.sleep(2)

os.makedirs(SAVE_FOLDER, exist_ok=True)
file_count = len([f for f in os.listdir(SAVE_FOLDER) if f.endswith('.csv')]) + 200
csv_path = os.path.join(SAVE_FOLDER, f'signal_{file_count}.csv')
img_path = os.path.join(SAVE_FOLDER, f'signal_{file_count}.png')

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time (ms)', 'EMG Value', 'Label'])
    start_time = time.time()

    print("Start collecting: press space to switch tabs, press q to exit collecting")
    try:
        while recording:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            elapsed_ms = int((time.time() - start_time) * 1000)

            if line and elapsed_ms >= 2000:
                try:
                    value = float(line)
                    writer.writerow([elapsed_ms, value, labels[label_index]])
                    print(f"{elapsed_ms}ms, {value}, label={labels[label_index]}")
                except ValueError:
                    pass
    finally:
        ser.close()
        listener.stop()
        print("============= Finish Collection, Close Serial Port ============")


df = pd.read_csv(csv_path)
plt.figure(figsize=(10, 5))
for label in labels:
    subset = df[df['Label'] == label]
    plt.plot(subset['Time (ms)'], subset['EMG Value'], label=label)

plt.legend()
plt.title("EMG singal")
plt.xlabel("time (ms)")
plt.ylabel("EMG value")
plt.grid(True)
plt.tight_layout()
plt.savefig(img_path)
plt.show()

print(f"📁 Saved CSV: {csv_path}")
print(f"🖼️ Saved PNG: {img_path}")
