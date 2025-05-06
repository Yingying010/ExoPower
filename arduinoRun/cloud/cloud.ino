#include <WiFiNINA.h>
#include <ArduinoMqttClient.h>

const char* ssid = "MSc_IoT";
const char* password = "MSc_IoT@UCL";
const char* mqttServer = "192.168.0.180"; 
const int mqttPort = 1883;
const char* mqttTopic = "emg/data";

WiFiClient net;
MqttClient mqttClient(net);
const int emgPin = A0;

void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  unsigned long startAttemptTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttemptTime < 10000) {
    Serial.print(".");
    delay(500);
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n❌ WiFi connection failed. Will retry...");
  }
}

void connectMQTT() {
  Serial.print("Connecting to MQTT... ");
  mqttClient.setId("emg02");

  if (!mqttClient.connect(mqttServer, mqttPort)) {
    Serial.print("❌ MQTT connect failed. Error = ");
    Serial.println(mqttClient.connectError());
  } else {
    Serial.println("✅ MQTT connected!");
  }
}

void setup() {
  Serial.begin(115200);
  delay(2000);

  connectWiFi();
  connectMQTT();
}

// === Butterworth Bandpass Filter ===
float Filter(float input) {
  float output = input;

  // Section 1
  static float z1_1, z2_1;
  float x1 = output - (-0.73945727 * z1_1) - (0.59923508 * z2_1);
  output = 0.00223489 * x1 + 0.00446978 * z1_1 + 0.00223489 * z2_1;
  z2_1 = z1_1;
  z1_1 = x1;

  // Section 2
  static float z1_2, z2_2;
  float x2 = output - (-1.03789224 * z1_2) - (0.64082390 * z2_2);
  output = x2 + 2.0 * z1_2 + z2_2;
  z2_2 = z1_2;
  z1_2 = x2;

  // Section 3
  static float z1_3, z2_3;
  float x3 = output - (-0.59186255 * z1_3) - (0.80647974 * z2_3);
  output = x3 - 2.0 * z1_3 + z2_3;
  z2_3 = z1_3;
  z1_3 = x3;

  // Section 4
  static float z1_4, z2_4;
  float x4 = output - (-1.33318587 * z1_4) - (0.85392964 * z2_4);
  output = x4 - 2.0 * z1_4 + z2_4;
  z2_4 = z1_4;
  z1_4 = x4;

  return output;
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️ WiFi disconnected, trying to reconnect...");
    connectWiFi();
  }

  if (!mqttClient.connected()) {
    Serial.println("⚠️ MQTT disconnected, trying to reconnect...");
    connectMQTT();
  }

  int rawValue = analogRead(emgPin);
  float filteredValue = Filter((float)rawValue);

  String payload = "{\"timestamp\":" + String(millis()) + ",\"emg_value\":" + String(filteredValue, 2) + "}";

  if (mqttClient.beginMessage(mqttTopic)) {
    mqttClient.print(payload);
    if (mqttClient.endMessage()) {
      Serial.println("✅ Published: " + payload);
    } else {
      Serial.println("❌ endMessage() failed");
    }
  } else {
    Serial.println("❌ beginMessage() failed");
  }

  delay(10);
}
