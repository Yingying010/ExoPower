// import React from "react";
// import Sidebar from "./components/Sidebar";
// import EMGChart from "./components/EMGChart";

// export default function App() {
//   return (
//     <div className="d-flex">
//       <Sidebar />
//       <div className="p-4 flex-grow-1">
//         <h1 className="mb-4">EMG Dashboard</h1>
//         <div className="card">
//           <div className="card-body">
//             <EMGChart />
//             <div className="bg-white text-primary border border-primary rounded px-4 py-3 mt-3 w-100 shadow-sm">
//               <strong>Status:</strong> <span className="ms-2">💪 Lifting</span>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }

import React, { useEffect, useState, useRef } from "react";
import Sidebar from "./components/Sidebar";
import EMGChart from "./components/EMGChart";
import MotorGauge from "./components/MotorGauge";
import Modal from "./components/Modal";


export default function App() {
  const [angle, setAngle] = useState(0);
  const [targetAngle, setTargetAngle] = useState(0);
  const [status, setStatus] = useState("🛌 Resting");
  const [isRunning, setIsRunning] = useState(false);
  const [modalContent, setModalContent] = useState(null);
  const [lastAvgEMG, setLastAvgEMG] = useState(null);

  const latestEMGRef = useRef(0);
  const emgValues = useRef([]); // 用于存储EMG数据

  // 模拟 lifting 状态切换与 EMG 值更新
  useEffect(() => {
    if (!isRunning) return;


    const interval = setInterval(() => {
    const emgValue = latestEMGRef.current; // ✅ 现在拿到最新的 EMG 值
    const isLifting = emgValue > 5;
  
    setStatus(isLifting ? "💪 Lifting" : "🛌 Resting");
    setTargetAngle(isLifting ? 60 : 0);
  
    emgValues.current.push(emgValue);
    }, 300);
  
    return () => clearInterval(interval);
  }, [isRunning]);
  
  


  // 平滑角度动画
  useEffect(() => {
    const interval = setInterval(() => {
      setAngle((prev) => {
        if (Math.abs(prev - targetAngle) < 2) return targetAngle;
        return prev < targetAngle ? prev + 2 : prev - 2;
      });
    }, 30);
    return () => clearInterval(interval);
  }, [targetAngle]);

  // 计算平均 EMG 值
  const toggleRunning = () => {
    if (isRunning && emgValues.current.length > 0) {
      const sum = emgValues.current.reduce((a, b) => a + b, 0);
      const avg = sum / emgValues.current.length;
      setLastAvgEMG(avg.toFixed(2)); // 显示平均值
      emgValues.current = []; // 清空记录
    }
    setIsRunning((prev) => !prev); // 切换开始/暂停状态
  };

  // 模态框内容显示
  const handleMenuClick = (item) => {
    switch (item) {
      case "Help":
        setModalContent({
          title: "Help - Component Descriptions",
          body: (
            <ul>
              <li><strong>EMGChart：</strong> Real-time display of EMG signal changes.</li>
              <li><strong>MotorGauge：</strong> Display of motor rotation angle</li>
              <li><strong>Status：</strong> The current state of the detected action.</li>
              <li><strong>Start/Pause：</strong> Does the control system perform action recognition and recording.</li>
            </ul>
          ),
        });
        break;
      case "History":
        setModalContent({
          title: "History - Average EMG Value of last record",
          body: lastAvgEMG
            ? <p>Average EMG value of last lifting and resting stage ：<strong>{lastAvgEMG}</strong></p>
            : <p>No record！Strat your action!🏃‍♀️</p>,
        });
        break;
      case "Settings":
        setModalContent({
          title: "Introduction - Why Choose ExoPower?",
          body: (
            <ul>
              <li>⚡ Samrt Motion Recognition Auto Boost</li>
              <li>🎯 Precise angle control </li>
              <li>📊 Simple Dashboard </li>
              <li>📱 Supports remote viewing </li>
            </ul>
          ),
        });
        break;
    }
  };

  return (
    <div className="d-flex">
      <Sidebar onMenuClick={handleMenuClick} />
      <div className="p-4 flex-grow-1">
        <h1 className="mb-4">EMG Dashboard</h1>

        <div className="mb-3">
          <button
            className={`btn ${isRunning ? "btn-danger" : "btn-success"} me-3`}
            onClick={toggleRunning}
          >
            {isRunning ? "Pause ⏸" : "Start ▶️"}
          </button>
          <span className="text-muted">
            Current Status：<strong>{status}</strong>
          </span>
        </div>

        <div className="card">
          <div className="card-body">
            <EMGChart 
            isRunning={isRunning}
            emgValues={emgValues.current}
            latestEMGRef={latestEMGRef} />
            <div className="d-flex align-items-center justify-content-between mt-4">
              <MotorGauge angle={angle} />
              <div className="bg-white text-primary border border-primary rounded px-4 py-3 w-100 ms-4 shadow-sm">
                <strong>Status:</strong> <span className="ms-2">{status}</span>
              </div>
            </div>
          </div>
        </div>
      </div>


      {modalContent && (
        <Modal
          title={modalContent.title}
          onClose={() => setModalContent(null)}
        >
          {modalContent.body}
        </Modal>
      )}
    </div>
  );
}

