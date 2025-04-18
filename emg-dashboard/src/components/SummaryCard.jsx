import React from "react";

export default function SummaryCard({ title, value }) {
  return (
    <div className="bg-white shadow rounded-lg p-4 text-center">
      <p className="text-sm text-gray-500">{title}</p>
      <p className="text-xl font-bold">{value}</p>
    </div>
  );
}
