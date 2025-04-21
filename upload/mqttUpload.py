import time
import json
from awscrt import mqtt
from awsiot import mqtt_connection_builder

# 配置信息
ENDPOINT = "a1zwy6veu609mq-ats.iot.eu-west-2.amazonaws.com"
CLIENT_ID = "emg01"
PATH_TO_CERT = "/Users/heartfillialucy/UCL/Designing Sensor Systems/Assignment/ExoPower/upload/emg02.cert.pem"
PATH_TO_KEY = "/Users/heartfillialucy/UCL/Designing Sensor Systems/Assignment/ExoPower/upload/emg02.private.key"
PATH_TO_ROOT = "/Users/heartfillialucy/UCL/Designing Sensor Systems/Assignment/ExoPower/upload/root-CA.crt"
TOPIC = "emg/data"  # 你可以自定义主题
PORT = 8883

# 构建 MQTT 连接
mqtt_connection = mqtt_connection_builder.mtls_from_path(
    endpoint=ENDPOINT,
    cert_filepath=PATH_TO_CERT,
    pri_key_filepath=PATH_TO_KEY,
    client_bootstrap=None,
    ca_filepath=PATH_TO_ROOT,
    client_id=CLIENT_ID,
    clean_session=False,
    keep_alive_secs=30
)

# 建立连接
print("🔌 正在连接 AWS IoT...")
connect_future = mqtt_connection.connect()
connect_future.result()
print("✅ 连接成功!")

# 模拟实时 EMG 数据上传
try:
    while True:
        emg_value = round(0.5 + 0.5 * time.time() % 1, 2)  # 模拟一个值
        message = {
            "timestamp": int(time.time()),
            "emg": emg_value
        }
        mqtt_connection.publish(
            topic=TOPIC,
            payload=json.dumps(message),
            qos=mqtt.QoS.AT_LEAST_ONCE
        )
        print(f"📤 已发布: {message}")
        time.sleep(1)
except KeyboardInterrupt:
    print("❌ 停止上传")
    mqtt_connection.disconnect()
