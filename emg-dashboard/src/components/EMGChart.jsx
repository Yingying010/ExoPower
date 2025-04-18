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

function generateMockEMGData() {
  return {
    timestamp: Date.now(),
    emg_value: (Math.sin(Date.now() / 500) + 1) / 2 + Math.random() * 0.1,
  };
}

export default function EMGChart() {
  const [emgData, setEmgData] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      const data = generateMockEMGData();
      setEmgData((prev) => [...prev.slice(-49), data]);
    }, 200);
    return () => clearInterval(interval);
  }, []);

  const chartData = {
    labels: emgData.map((d) => new Date(d.timestamp)),
    datasets: [
      {
        label: "EMG",
        data: emgData.map((d) => d.emg_value),
        borderColor: "#6366f1",
        fill: false,
        tension: 0.2,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    scales: {
      x: {
        type: "time",
        time: { unit: "second" },
        title: { display: true, text: "时间" },
      },
      y: {
        min: 0,
        max: 1.5,
        title: { display: true, text: "EMG 值" },
      },
    },
  };

  return (
    <div className="bg-white shadow rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-2">实时肌电信号</h3>
      <Line data={chartData} options={chartOptions} />
    </div>
  );
}
