import json
import time
import joblib
import numpy as np
from awscrt import mqtt
from awsiot import mqtt_connection_builder
import paho.mqtt.client as paho_local

LOCAL_BROKER_IP = "192.168.0.183"
LOCAL_TOPIC = "emg/data"

AWS_ENDPOINT = "a1zwy6veu609mq-ats.iot.eu-west-2.amazonaws.com"
AWS_CLIENT_ID = "emg-predictor"
PATH_TO_CERT = "upload/emg02.cert.pem"
PATH_TO_KEY = "upload/emg02.private.key"
PATH_TO_ROOT = "upload/root-CA.crt"
PUBLISH_TOPIC = "emg/prediction"

WINDOW_SIZE = 100
emg_window = []
model = joblib.load("model/rfModel.pkl")

def extract_features(signal):
    signal = np.array(signal)
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.sign(signal)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    sk = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**3) / np.std(signal)**3)
    ku = 0 if np.all(signal == 0) else float(np.mean((signal - np.mean(signal))**4) / np.std(signal)**4)
    return [mav, rms, wl, zc, ssc, sk, ku]

mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=AWS_ENDPOINT,
    cert_filepath=PATH_TO_CERT,
    pri_key_filepath=PATH_TO_KEY,
    ca_filepath=PATH_TO_ROOT,
    client_id=AWS_CLIENT_ID,
    clean_session=False,
    keep_alive_secs=30
)

print("🔌 Connecting to AWS IoT...")
mqtt_connection.connect().result()
print("✅ AWS Connected!")


def on_local_message(client, userdata, msg):
    global emg_window
    try:
        message = json.loads(msg.payload.decode("utf-8"))
        emg_value = float(message.get("emg_value", 0))
        timestamp = int(message.get("timestamp", 0))
        print(f"📥 Local EMG received: {message}")

        emg_window.append(emg_value)
        if len(emg_window) > WINDOW_SIZE:
            emg_window.pop(0)

        if len(emg_window) == WINDOW_SIZE:
            features = extract_features(emg_window)
            prediction = int(model.predict([features])[0])
            label = "idle" if prediction == 0 else "lifting"

            aws_payload = {
                "timestamp": int(time.time() * 1000),
                "emg_value": emg_value,
                "prediction": label
            }

            mqtt_connection.publish(
                topic=PUBLISH_TOPIC,
                payload=json.dumps(aws_payload),
                qos=mqtt.QoS.AT_LEAST_ONCE
            )
            print(f"📤 Prediction published to AWS: {aws_payload}")

    except Exception as e:
        print(f"❌ Error in local handler: {e}")

local_client = paho_local.Client()
local_client.on_message = on_local_message
local_client.connect(LOCAL_BROKER_IP, 1883, 60)
local_client.subscribe(LOCAL_TOPIC)
print(f"🟢 Subscribed to {LOCAL_TOPIC} on {LOCAL_BROKER_IP}")

try:
    local_client.loop_forever()
except KeyboardInterrupt:
    mqtt_connection.disconnect().result()
    print("🛑 Stopped.")