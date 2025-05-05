from flask import Flask
from flask_socketio import SocketIO
from awscrt import io, mqtt
from awsiot import mqtt_connection_builder
import json

ENDPOINT = "a1zwy6veu609mq-ats.iot.eu-west-2.amazonaws.com"
CLIENT_ID = "emg02"
PATH_TO_CERT = "upload/emg02.cert.pem"
PATH_TO_KEY = "upload/emg02.private.key"
PATH_TO_ROOT = "upload/root-CA.crt"
TOPIC = "emg/data"
PORT = 8883

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

latest_message = None

@app.route("/")
def index():
    return "SocketIO server running"

@socketio.on("connect")
def test_connect():
    print("🟢 front-eng has already connected socket")
    if latest_message:
        socketio.emit("emg_data", latest_message) 


mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=ENDPOINT,
    port=PORT,
    cert_filepath=PATH_TO_CERT,
    pri_key_filepath=PATH_TO_KEY,
    client_bootstrap=io.ClientBootstrap.get_or_create_static_default(),
    ca_filepath=PATH_TO_ROOT,
    client_id=CLIENT_ID,
    clean_session=False,
    keep_alive_secs=6,
)

print("🔌 connecting AWS IoT...")
connect_future = mqtt_connection.connect()
connect_future.result()
print("✅ AWS IoT connected")

data_buffer = []

def on_message_received(topic, payload, **kwargs):
    global data_buffer
    message = json.loads(payload)
    print("📥 received:", message)
    data_buffer.append(message)

    if len(data_buffer) >= 30:
        print("📦 sending：", len(data_buffer))
        socketio.emit("emg_data_batch", data_buffer)
        data_buffer = []

mqtt_connection.subscribe(
    topic=TOPIC,
    qos=mqtt.QoS.AT_LEAST_ONCE,
    callback=on_message_received,
)


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5050)
