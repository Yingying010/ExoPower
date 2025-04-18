import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  LineElement,
  TimeScale,
  LinearScale,
  PointElement,
  Tooltip,
  Legend,
} from "chart.js";
import "chartjs-adapter-date-fns";

ChartJS.register(LineElement, TimeScale, LinearScale, PointElement, Tooltip, Legend);

// 模拟数据生成函数
function generateMockEMGData() {
  return {
    timestamp: Date.now(),
    emg_value: (Math.sin(Date.now() / 500) + 1) / 2 + Math.random() * 0.1,
  };
}

export default function EMGDashboard() {
  const [emgData, setEmgData] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      const data = generateMockEMGData();
      setEmgData((prev) => [...prev.slice(-99), data]);
    }, 200);

    return () => clearInterval(interval);
  }, []);

  const chartData = {
    labels: emgData.map((d) => new Date(d.timestamp)),
    datasets: [
      {
        label: "EMG Value",
        data: emgData.map((d) => d.emg_value),
        borderColor: "#36a2eb",
        fill: false,
        tension: 0.1,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    scales: {
      x: {
        type: "time",
        time: {
          unit: "second",
        },
        title: {
          display: true,
          text: "时间",
        },
      },
      y: {
        title: {
          display: true,
          text: "EMG 值",
        },
        min: 0,
        max: 1.5,
      },
    },
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">ExoPower 实时 EMG 数据监控（本地模拟）</h1>
      <div className="bg-white shadow rounded-lg p-4">
        <Line data={chartData} options={chartOptions} />
      </div>
    </div>
  );
}
