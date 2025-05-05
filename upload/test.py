import time
from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return "SocketIO server running"

@socketio.on("connect")
def test_connect():
    print("🟢 前端已连接 socket")
    # 发送一条测试消息
    socketio.emit("emg_data", {"timestamp": 123, "signal_value": 0.5, "prediction": "idle"})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5050)
