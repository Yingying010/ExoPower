import React, { useEffect, useRef, useState } from "react";
import { Line } from "react-chartjs-2";
import { io } from "socket.io-client";
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale,
} from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";

// 注册 Chart.js 模块和插件
ChartJS.register(
  LineElement,
  PointElement,
  LinearScale,
  Title,
  Tooltip,
  Legend,
  CategoryScale,
  ChartDataLabels
);

// 初始化 socket.io 客户端
const socket = io("http://localhost:3001");

socket.on("connect", () => {
  console.log("✅ Socket 已连接，ID:", socket.id);
});

export default function EMGChart({ isRunning, emgValues, latestEMGRef }) {
  const [dataPoints, setDataPoints] = useState([]);
  const [latestValue, setLatestValue] = useState(null);
  const counterRef = useRef(0); // 使用 ref 做采样点编号

  useEffect(() => {
    const handleEmg = (value) => {
      const num = parseFloat(value);
      if (isNaN(num)) return;

      const newPoint = { x: counterRef.current, y: num };
      setDataPoints((prev) => [...prev.slice(-49), newPoint]);
      setLatestValue(num);
      counterRef.current += 1;

      if (latestEMGRef) {
        latestEMGRef.current = num;
        console.log("📡 Update latestEMGRef.current =", latestEMGRef.current);
      }

      console.log("📡 收到 EMG 数据:", num);
      console.log("📡 添加点:", newPoint);
      console.log("📡 最新 EMG：", num);
      console.log("📡 更新 latestEMGRef.current =", latestEMGRef.current);

    };

    socket.on("emg", handleEmg);

    return () => {
      socket.off("emg", handleEmg);
    };
  }, []);

  const data = {
    datasets: [
      {
        label: "EMG Signal",
        data: dataPoints,
        parsing: false,
        fill: false,
        borderColor: "rgb(75, 192, 192)",
        tension: 0.3,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: {
        display: true,
        text: "EMG Realtime Signal (Scrolling Window)",
      },
      datalabels: {
        display: false,
        align: "end",
        anchor: "end",
        color: "blue",
        font: {
          size: 10,
        },
        formatter: (value) => value.y.toFixed(2),
      },
    },
    scales: {
      x: {
        type: "linear",
        title: {
          display: true,
          text: "timestamp",
        },
        ticks: {
          stepSize: 1,
        },
      },
      y: {
        min: -300,
        max: 300,
        title: {
          display: true,
          text: "EMG Value",
        },
      },
    },
  };

  return (
    <div className="card mt-4">
      <div className="card-header">
        <strong>Status：</strong> Lifting
        <span style={{ marginLeft: "20px", color: "blue" }}>
          Current EMG Value：{latestValue !== null ? latestValue.toFixed(2) : "--"}
        </span>
      </div>
      <div className="card-body" style={{ height: "400px", width: "100%" }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
}

//--------------------------------------------ver2
