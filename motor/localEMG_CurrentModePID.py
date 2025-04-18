import os
import time
import signal
from dynamixel_sdk import *  # Dynamixel SDK
import numpy as np
import serial

# ========== Dynamixel Constants ==========
ADDR_TORQUE_ENABLE = 64
ADDR_OPERATING_MODE = 11
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_POSITION = 132

CURRENT_CONTROL_MODE = 0  # 0 = Current Mode
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
DXL_ID = 1
DEVICENAME = '/dev/tty.usbserial-FT9HDB5F'  # Dynamixel port

CURRENT_CONTROL_MODE = 0  # 0 = Current Mode
POSITION_CONTROL_MODE = 3  # 3 = Position Mode
ADDR_GOAL_POSITION = 116  # goal position

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
MAX_CURRENT = 6  # Maximum current (mA)
POS_MIN = 0    # Minimum angle (°)
POS_MAX = 300    # Maximum angle (°)
DXL_RESOLUTION = 4095 / 360.0  # Dynamixel angle resolution
POS_180 = int(180 * DXL_RESOLUTION)
BRAKE_CURRENT = 2  # Brake current when going out of range
TOLERANCE = 1

# ========== EMG Constants ==========
SERIAL_EMG_PORT = '/dev/tty.usbmodem1102'
EMG_BAUDRATE = 9600
EMG_MIN = 0     # Minimum EMG value (adjust for your device)
EMG_MAX = 1000  # Maximum EMG value (adjust for your device)

# ========== PID Parameters ==========
Kp = 1.0
Ki = 0.02
Kd = 0.2

error_sum = 0.0
previous_error = 0.0

# ========== Initialize Dynamixel Port ==========
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Failed to open the port.")
    quit()
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set the baudrate.")
    quit()

# Listen for CTRL + C, and ensure the motor stops on exit
def stop_motor(signal, frame):
    print("\nReceived termination signal, stopping the motor!")
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, 0)  # Set current to 0
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)  # Disable torque
    portHandler.closePort()  # Close port
    print("Port closed, motor stopped.")
    quit()

signal.signal(signal.SIGINT, stop_motor)

# init 180 degree
print("Set the motor initial position to 0°.")
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, POS_180)

while True:
    present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
    current_angle = present_position / DXL_RESOLUTION

    print(f"🎯 当前位置: {current_angle:.2f}°")
    
    if abs(current_angle - 0) <= TOLERANCE:
        print("✅ 电机已到达 0°，切换到 Current Mode 进行 EMG 控制")
        break

    time.sleep(0.1)

# ========== Switch to Current Mode ==========
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# ========== Initialize and Connect to EMG Serial Port ==========
try:
    ser_emg = serial.Serial(SERIAL_EMG_PORT, EMG_BAUDRATE, timeout=0.1)
    print(f"Connected to EMG serial port: {SERIAL_EMG_PORT}")
    print()
except Exception as e:
    print(f"Failed to connect to EMG serial port: {e}")
    stop_motor(None, None)

# ========== Exponential Moving Average for EMG ==========
smoothed_emg = 0
alpha = 0.2
previous_emg = 0

# -----------------------------------------
# Main loop: Read EMG -> Filter -> Control Motor
# -----------------------------------------
print("Starting control loop...")
time.sleep(1.0)  # Short delay before starting
while True:
    try:
        # Assume one EMG value per line, e.g., "123\n"
        line = ser_emg.readline().decode('utf-8').strip()
        print(line)
        if not line:
            # If no data is read, skip or fill with default, etc.
            continue

        # Convert the input string to a numeric value
        emg_raw_value = float(line)  # or int(line), depending on your data format
        
        # Signal Filtering (EMA)
        smoothed_emg = alpha * emg_raw_value + (1 - alpha) * smoothed_emg

        # Map EMG value to target angle
        target_angle = POS_MIN + ((smoothed_emg - EMG_MIN) / (EMG_MAX - EMG_MIN)) * (POS_MAX - POS_MIN)
        # Keep the angle within [POS_MIN, POS_MAX]
        target_angle = max(POS_MIN, min(POS_MAX, target_angle))

        # Read current angle
        present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        if present_position < 0 or present_position > 4095:
            print(f"⚠️ Abnormal 'present_position' value read: {present_position}")
            continue

        current_angle = float(present_position) / DXL_RESOLUTION

        # Calculate error
        error = target_angle - current_angle
        error_sum += error      # Integral term
        error_diff = error - previous_error  # Derivative term
        previous_error = error

        # Calculate PID-based current
        motor_current = int(Kp * error + Ki * error_sum + Kd * error_diff)
        motor_current = max(min(motor_current, MAX_CURRENT), -MAX_CURRENT)

        # If out of range, apply brake current
        if current_angle >= POS_MAX:
            print(f"⚠️ Exceeding maximum angle {POS_MAX}°, applying brake current")
            motor_current = -BRAKE_CURRENT
        elif current_angle <= POS_MIN:
            print(f"⚠️ Below minimum angle {POS_MIN}°, applying brake current")
            motor_current = BRAKE_CURRENT

        # Send current to Dynamixel
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, motor_current)

        # Print debug info
        print(f"📊 EMG: {smoothed_emg:.2f} | 🎯 Target Angle: {target_angle:.2f}° "
              f"| ⚡ Target Current: {motor_current} mA | 🏎️ Current Angle: {current_angle:.2f}°")

        time.sleep(0.1)

    except KeyboardInterrupt:
        stop_motor(None, None)
    except Exception as e:
        print(f"Error reading/processing EMG data: {e}")
        # You can choose whether to stop the motor on error or just continue
        # stop_motor(None, None)
        pass
