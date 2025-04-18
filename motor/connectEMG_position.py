import time
import paho.mqtt.client as mqtt
from dynamixel_sdk import *

SERIAL_PORT = "/dev/tty.usbserial-FT9HDB5F"
BAUDRATE = 115200
DYNAMIXEL_BAUDRATE = 57600

ADDR_TORQUE_ENABLE = 64 
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

PROTOCOL_VERSION = 2.0
DXL_ID = 1
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

DXL_MIN_POSITION = 0 
DXL_MAX_POSITION = 4095
TURN_ANGLE = 4095 

# init Dynamixel
portHandler = PortHandler(SERIAL_PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Failed to open the serial port！")
    quit()

if not portHandler.setBaudRate(DYNAMIXEL_BAUDRATE):
    print("Failed to set the baud rate！")
    quit()

# Activate torque
dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("Torque Enable Error:", packetHandler.getTxRxResult(dxl_comm_result))
    quit()
elif dxl_error != 0:
    print("Torque Enable Packet Error:", packetHandler.getRxPacketError(dxl_error))
    quit()
else:
    print("Dynamixel torque is now enabled!")


MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 8883
MQTT_TOPIC = "emgDate"

# record time
emg_value = None
low_emg_start_time = None
high_emg_start_time = None
last_movement_time = None

LOW_EMG_THRESHOLD = 300
HIGH_EMG_THRESHOLD = 300
REQUIRED_DURATION = 3
MOVEMENT_COOLDOWN = 2

# get the current position of motor **
dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
PREV_POSITION = dxl_present_position 
print(f"Current motor position: {PREV_POSITION}")

# Resetting motor position
if PREV_POSITION != DXL_MIN_POSITION:
    print("Resetting motor position to 0°...")
    PREV_POSITION = DXL_MIN_POSITION
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, DXL_MIN_POSITION)

    # Waiting for the motor to reach the target position
    while True:
        dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        if abs(dxl_present_position - DXL_MIN_POSITION) < 20:
            break
        time.sleep(0.1)

print("Motor initialization complete, waiting for EMG signal...")

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker!")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    global emg_value
    try:
        emg_value = float(msg.payload.decode().strip())
        print(f"EMG: {emg_value}")
    except ValueError:
        print("Invalid MQTT message received.")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

while True:
    if emg_value is not None:
        current_time = time.time()

        # Clockwise rotation (EMG < 300 for 3 seconds)
        if emg_value < LOW_EMG_THRESHOLD:
            if low_emg_start_time is None:
                low_emg_start_time = current_time
            if high_emg_start_time is not None:
                high_emg_start_time = None

            if current_time - low_emg_start_time >= REQUIRED_DURATION:
                if last_movement_time is None or current_time - last_movement_time > MOVEMENT_COOLDOWN:
                    print("EMG < 300 for 3 seconds, rotating clockwise 360°!")
                    target_position = (PREV_POSITION - TURN_ANGLE) % 4096

                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, target_position)

                    if dxl_comm_result == COMM_SUCCESS:
                        print(f"Rotating clockwise to {target_position} (Angle: {target_position * 360 / 4095:.2f}°)")
                        PREV_POSITION = target_position
                        last_movement_time = current_time
                    else:
                       print("Clockwise rotation failed!")

                low_emg_start_time = None 

        # Counterclockwise rotation (EMG > 300 for 3 seconds)
        elif emg_value > HIGH_EMG_THRESHOLD:
            if high_emg_start_time is None:
                high_emg_start_time = current_time
            if low_emg_start_time is not None:
                low_emg_start_time = None

            if current_time - high_emg_start_time >= REQUIRED_DURATION:
                if last_movement_time is None or current_time - last_movement_time > MOVEMENT_COOLDOWN:
                    print("💡 EMG above 300 for 3 seconds, rotating counterclockwise 360°!")
                    target_position = (PREV_POSITION + TURN_ANGLE) % 4096

                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, target_position)

                    if dxl_comm_result == COMM_SUCCESS:
                        print(f"Rotating counterclockwise to {target_position} (Angle: {target_position * 360 / 4095:.2f}°)")
                        PREV_POSITION = target_position
                        last_movement_time = current_time
                    else:
                        print("Counterclockwise rotation failed!")

                high_emg_start_time = None

    time.sleep(0.1)
