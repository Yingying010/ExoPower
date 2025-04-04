import time
import signal
from dynamixel_sdk import *  # Dynamixel SDK

# ========== Dynamixel Constants ==========
ADDR_TORQUE_ENABLE      = 64
ADDR_OPERATING_MODE     = 11
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132

TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0
POSITION_CONTROL_MODE   = 3
PROTOCOL_VERSION        = 2.0
DXL_ID                  = 1
BAUDRATE                = 57600
DEVICENAME              = '/dev/tty.usbserial-FT9HDB5F'  # ← Change to your actual port

DXL_RESOLUTION          = 4095 / 360.0  # ticks per degree
TOLERANCE               = 1  # ±1° tolerance

# ========== Initialize Port ==========
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("❌ Failed to open the port.")
    quit()
if not portHandler.setBaudRate(BAUDRATE):
    print("❌ Failed to set the baudrate.")
    quit()

# ========== Safe Exit Function ==========
def stop_motor(signal, frame):
    print("\n🛑 Termination signal received, disabling motor torque...")
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    portHandler.closePort()
    print("✅ Port closed. Program exited.")
    quit()

signal.signal(signal.SIGINT, stop_motor)

# ========== Set to Position Control Mode and Move to 0° ==========
print("⚙️ Setting motor to Position Control Mode. Moving to 0°...")
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

pos_0 = int(0 * DXL_RESOLUTION)
packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, pos_0)

# ========== Wait Until Motor Reaches 0° ==========
while True:
    present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
    current_angle = present_position / DXL_RESOLUTION
    print(f"🎯 Current angle: {current_angle:.2f}°")

    if abs(current_angle - 0) <= TOLERANCE:
        print("✅ Motor successfully reset to 0°")
        break

    time.sleep(0.1)

# ========== Disable Torque & Close Port ==========
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()
print("✅ Motor torque disabled. Port closed. Done.")
