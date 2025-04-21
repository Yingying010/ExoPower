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

  const emgValues = useRef([]); // 用于存储EMG数据

  // 模拟 lifting 状态切换与 EMG 值更新
  useEffect(() => {
    if (!isRunning) return;
  
    const interval = setInterval(() => {
      const isLifting = Math.random() > 0.5;
      setStatus(isLifting ? "💪 Lifting" : "🛌 Resting");
      setTargetAngle(isLifting ? 60 : 0);
  
      const emgValue = +(Math.random() * 0.5 + 0.5).toFixed(2); // 0.5~1
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
              <li><strong>EMGChart：</strong> 实时显示肌电信号变化。</li>
              <li><strong>MotorGauge：</strong> 显示马达旋转角度。</li>
              <li><strong>Status：</strong> 当前检测的动作状态。</li>
              <li><strong>Start/Pause：</strong> 控制系统是否进行动作识别与记录。</li>
            </ul>
          ),
        });
        break;
      case "History":
        setModalContent({
          title: "History - 上一次运动平均EMG",
          body: lastAvgEMG
            ? <p>上一次 lifting 和 resting 阶段的平均 EMG 值为：<strong>{lastAvgEMG}</strong></p>
            : <p>暂无记录，请先开始一次运动吧！🏃‍♀️</p>,
        });
        break;
      case "Settings":
        setModalContent({
          title: "Introduction - Why Choose ExoPower?",
          body: (
            <ul>
              <li>⚡ 智能动作识别自动助力</li>
              <li>🎯 精准角度控制</li>
              <li>📊 简洁仪表盘</li>
              <li>📱 支持远程查看</li>
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
            当前状态：<strong>{status}</strong>
          </span>
        </div>

        <div className="card">
          <div className="card-body">
            <EMGChart isRunning={isRunning} emgValues={emgValues.current} />
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
