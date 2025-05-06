
require("dotenv").config();
const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const mqtt = require("mqtt");
const fs = require("fs");
const path = require("path");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173",
    methods: ["GET", "POST"],
  },
});

const mqttOptions = {
  host: process.env.AWS_IOT_ENDPOINT,
  port: 8883,
  protocol: "mqtts",
  key: fs.readFileSync(path.join(__dirname, process.env.AWS_IOT_PRIVATE_KEY)),
  cert: fs.readFileSync(path.join(__dirname, process.env.AWS_IOT_CERTIFICATE)),
  ca: fs.readFileSync(path.join(__dirname, process.env.AWS_IOT_CA)),
  clientId: "emg-dashboard-client-" + Math.floor(Math.random() * 1000),
  
};
console.log("🔧 MQTT 配置：", mqttOptions);
console.log("🔍 证书文件存在？",
  fs.existsSync(path.join(__dirname, process.env.AWS_IOT_PRIVATE_KEY)),
  fs.existsSync(path.join(__dirname, process.env.AWS_IOT_CERTIFICATE)),
  fs.existsSync(path.join(__dirname, process.env.AWS_IOT_CA))
);

const mqttClient = mqtt.connect(mqttOptions);

mqttClient.on("connect", () => {
  console.log("✅ connected to AWS IoT Core");

  mqttClient.subscribe("emg/data", { qos: 0 }, (err) => {
    if (err) {
      console.error("❌ failed to subscribe:", err);
    } else {
      console.log("📡 subscribe to emg/data success!（QoS 0）");
    }
  });
  
});

mqttClient.on("error", (error) => {
  console.error("❌ MQTT error connectiomn:", error);
});

mqttClient.on("close", () => {
  console.log("🔌 MQTT connection close");
});

mqttClient.on("reconnect", () => {
  console.log("🔁 reconnecting to MQTT...");
});


// mqttClient.on("message", (topic, message) => {
//   console.log(`📥 收到 MQTT 消息 -> 主题: ${topic}, 内容: ${message.toString()}`);
//   const value = parseFloat(message.toString());
//   if (!isNaN(value)) {
//     console.log("📥 MQTT 数据：", value);

//     // ✅ 这一行可能现在缺失了
//     io.emit("emg", value); // ⬅️ 添加这一行，广播给前端

//     // ✅ 建议加一行调试输出，确认已广播
//     console.log("📤 向前端发送：", value);
//   }
// });

mqttClient.on("message", (topic, message) => {
  try {
    const payload = JSON.parse(message.toString()); // ✅ 解析 JSON 字符串
    const value = parseFloat(payload.signal_value); // ✅ 取出 signal_value

    console.log("📥  MQTT Message Received -> topic:", topic, ", content:", payload);
    
    if (!isNaN(value)) {
      console.log("📥 MQTT DATA：", value);
      io.emit("emg", value); // ✅ 广播给前端
    } else {
      console.warn("⚠️ 收到无效 signal_value:", payload.signal_value);
    }
  } catch (err) {
    console.error("❌ JSON 解析失败:", err.message);
  }
});


io.on("connection", (socket) => {
  console.log("🌐 前端连接成功");
});

server.listen(3001, () => {
  console.log("🚀 Server running at http://localhost:3001");
});
