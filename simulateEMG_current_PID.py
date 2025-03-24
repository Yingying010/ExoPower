import os
import time
import signal
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import numpy as np

# Dynamixel 配置
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_POSITION = 132

CURRENT_CONTROL_MODE = 0  # 0 = Current Mode
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
DXL_ID = 1
DEVICENAME = '/dev/tty.usbserial-FT9HDB5F'  # 设备端口

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
MAX_CURRENT = 6
POS_MIN = 180
POS_MAX = 300
DXL_RESOLUTION = 4095 / 360.0
TOLERANCE = 1 
BRAKE_CURRENT = 2 

# Angle Mapping
EMG_MIN = 10
EMG_MAX = 800

# PID
Kp = 0.5
Ki = 0.01
Kd = 0.1

error_sum = 0
previous_error = 0

#Init port
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Port opening failed")
    quit()
if not portHandler.setBaudRate(BAUDRATE):
    print("Baud rate setting failed")
    quit()

# Listen for `CTRL + C` to ensure motors stop on exit
def stop_motor(signal, frame):
    print("\nTermination signal received, stop the motor!")
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()
    print("Port closed, motor stopped")
    quit()

signal.signal(signal.SIGINT, stop_motor)

packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# Generate simulated EMG data (signal 10 → 600)
emgData = []
for cycle in range(1):
    emgData.extend(list(np.random.randint(10, 50, size=10)))
    emgData.extend(list(np.random.randint(50, 300, size=20)))
    emgData.extend(list(np.random.randint(300, 600, size=30)))
    emgData.extend(list(np.random.randint(600, 900, size=50))) 
    emgData.extend(list(np.random.randint(300, 600, size=30)))
    emgData.extend(list(np.random.randint(50, 300, size=20)))
    emgData.extend(list(np.random.randint(10, 50, size=10)))

smoothed_emg = 0
alpha = 0.2
previous_emg = 0

while True:
    for emg_value in emgData:
        # 信号滤波
        smoothed_emg = alpha * emg_value + (1 - alpha) * smoothed_emg

        # 映射 EMG 信号值到目标角度
        target_angle = POS_MIN + ((smoothed_emg - EMG_MIN) / (EMG_MAX - EMG_MIN)) * (POS_MAX - POS_MIN)
        target_angle = max(POS_MIN, min(POS_MAX, target_angle))  # 限制范围

        # 获取当前角度
        present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        if present_position < 0 or present_position > 4095:
            print(f"⚠️ The read `present_position` value is abnormal: {present_position}")
            continue

        current_angle = float(present_position) / DXL_RESOLUTION

        # PID
        error = target_angle - current_angle
        error_sum += error  # Integral Item
        error_diff = error - previous_error  # Differential term
        previous_error = error  # update error

        # Calculate current for PID control
        motor_current = int(Kp * error + Ki * error_sum + Kd * error_diff)
        motor_current = max(min(motor_current, MAX_CURRENT), -MAX_CURRENT)

        # Apply brake current to prevent over-limit
        if current_angle >= POS_MAX:
            print(f"⚠️ When the maximum angle {POS_MAX}° is exceeded, the brake current is applied")
            motor_current = -BRAKE_CURRENT
        elif current_angle <= POS_MIN:
            print(f"⚠️ Below the minimum angle {POS_MIN}°, brake current is applied")
            motor_current = BRAKE_CURRENT

        # Send current to control torque
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, motor_current)

        print(f"📊 EMG: {smoothed_emg:.2f} | 🎯 goal angle: {target_angle:.2f}° | ⚡ goal current: {motor_current} mA | 🏎️ current goal: {current_angle:.2f}°")

        time.sleep(0.1)
