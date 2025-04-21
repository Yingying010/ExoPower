import React from "react";

export default function Modal({ title, children, onClose }) {
  return (
    <div
      className="position-fixed top-0 start-0 w-100 h-100 d-flex justify-content-center align-items-center"
      style={{ backgroundColor: "rgba(0, 0, 0, 0.5)", zIndex: 1050 }}
    >
      <div className="bg-white rounded p-4 shadow" style={{ minWidth: "400px", maxWidth: "90%" }}>
        <div className="d-flex justify-content-between align-items-center mb-3">
          <h5 className="m-0">{title}</h5>
          <button onClick={onClose} className="btn btn-sm btn-outline-secondary">✕</button>
        </div>
        <div>{children}</div>
      </div>
    </div>
  );
}
