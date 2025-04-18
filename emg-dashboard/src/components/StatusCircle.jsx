import React from "react";

export default function StatusCircle({ value, label }) {
  return (
    <div className="bg-white shadow rounded-lg p-4 flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full" viewBox="0 0 36 36">
          <path
            d="M18 2.0845
               a 15.9155 15.9155 0 0 1 0 31.831
               a 15.9155 15.9155 0 0 1 0 -31.831"
            fill="none"
            stroke="#e5e7eb"
            strokeWidth="2"
          />
          <path
            d="M18 2.0845
               a 15.9155 15.9155 0 0 1 ${(value / 100) * 100 * 0.628} 0"
            fill="none"
            stroke="#6366f1"
            strokeWidth="2"
            strokeDasharray={`${value}, 100`}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center text-xl font-semibold">
          {value}%
        </div>
      </div>
      <p className="mt-2 text-sm text-gray-500">{label}</p>
    </div>
  );
}
