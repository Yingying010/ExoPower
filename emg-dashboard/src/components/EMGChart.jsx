import React, { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";

export default function EMGChart() {
  const [emgData, setEmgData] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5000/api/emg")
      .then((res) => res.json())
      .then((data) => {
        setEmgData(data);
      });
  }, []);

  const chartData = {
    labels: emgData.map((point) => point.timestamp),
    datasets: [
      {
        label: "EMG Signal",
        data: emgData.map((point) => point.emg_value),
        borderColor: "rgb(75, 192, 192)",
        tension: 0.3,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { position: "top" },
      title: { display: true, text: "EMG Signal (from S3)" },
    },
    scales: {
      y: { min: 0, max: 1 },
    },
  };

  return (
    <div className="card">
      <div className="card-body">
        <Line data={chartData} options={chartOptions} />
      </div>
    </div>
  );
}
