import React from "react";

const menuItems = [
  { label: "History" },
  { label: "Settings" }, // 名字仍为 Settings，但功能为 Introduction
  { label: "Help" },
];

export default function Sidebar({ onMenuClick }) {
  return (
    <div className="bg-primary text-white vh-100 p-3" style={{ width: "240px" }}>
      <h4 className="mb-4">ExoPower</h4>
      <ul className="nav flex-column">
        {menuItems.map(({ label }) => (
          <li className="nav-item" key={label}>
            <button
              onClick={() => onMenuClick(label)}
              className="nav-link text-white px-3 py-2 rounded w-100 text-start border-0 bg-transparent hover:bg-white hover:text-primary transition"
            >
              {label === "Settings" ? "Introduction" : label}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
