import React from "react";

export default function Sidebar() {
  return (
    <aside className="bg-indigo-700 text-white p-4 space-y-4 min-h-screen">
      <h1 className="text-xl font-bold mb-6">ExoPower</h1>
      <nav className="space-y-2">
        <a href="#" className="block hover:text-indigo-300">Dashboard</a>
        <a href="#" className="block hover:text-indigo-300">History</a>
        <a href="#" className="block hover:text-indigo-300">Settings</a>
        <a href="#" className="block hover:text-indigo-300">Help</a>
      </nav>
    </aside>
  );
}
