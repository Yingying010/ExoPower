import React from "react";

export default function MotorGauge({ angle }) {
  return (
    <div className="text-center" style={{ width: "180px" }}>
      <svg viewBox="0 0 200 100" width="100%">
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#ccc" strokeWidth="10" />
        <line
          x1="100"
          y1="100"
          x2={100 + 80 * Math.cos((Math.PI * (angle - 90)) / 180)}
          y2={100 + 80 * Math.sin((Math.PI * (angle - 90)) / 180)}
          stroke="#007bff"
          strokeWidth="6"
          strokeLinecap="round"
        />
      </svg>
      <div className="mt-2">
        <strong>{Math.round(angle)}°</strong>
      </div>
    </div>
  );
}
