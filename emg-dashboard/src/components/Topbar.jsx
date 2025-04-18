import React from "react";

export default function Topbar({ user, status }) {
  return (
    <div className="flex justify-between items-center">
      <div>
        <h2 className="text-2xl font-semibold">Dashboard</h2>
        <p className="text-sm text-gray-500">当前状态: {status}</p>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-gray-700">👤 {user}</span>
      </div>
    </div>
  );
}
