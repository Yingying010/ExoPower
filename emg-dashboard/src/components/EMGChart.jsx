import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import io from "socket.io-client";

// 注册 Chart.js 所需组件
import {
  Chart as ChartJS,
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(
  LineElement,
  PointElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend
);

// 连接 socket.io 服务器
const socket = io("http://localhost:5050");

const EMGChart = () => {
  const [emgData, setEmgData] = useState([]);
  const [chartKey, setChartKey] = useState(0);

  useEffect(() => {
    socket.on("emg_data", (message) => {
      console.log("📥 接收到数据：", message);
      console.log("🔍 signal_value:", message.signal_value);

      setEmgData((prevData) => {
        const updated = [...prevData, message].slice(-30); // 仅保留最近30个
        return updated;
      });
    });

    return () => {
      socket.off("emg_data");
    };
  }, []);

  useEffect(() => {
    setChartKey((prev) => prev + 1);
  }, [emgData]);

  const chartData = {
    labels: emgData.map((point) =>
      new Date(point.timestamp * 1000).toLocaleTimeString()
    ),
    datasets: [
      {
        label: "EMG Signal",
        data: emgData.map((point) => point.signal_value), // ✅ 使用正确字段名
        borderColor: "rgba(75,192,192,1)",
        backgroundColor: "rgba(75,192,192,0.2)",
        tension: 0.3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: "Time",
        },
      },
      y: {
        type: "linear",
        display: true,
        title: {
          display: true,
          text: "EMG Value",
        },
      },
    },
  };

  return (
    <div style={{ width: "100%", maxWidth: "800px", margin: "0 auto" }}>
      <h2>Real-Time EMG Data</h2>
      <Line key={chartKey} data={chartData} options={chartOptions} />
    </div>
  );
};

export default EMGChart;
