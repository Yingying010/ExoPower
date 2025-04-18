import React from "react";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";
import EMGChart from "./components/EMGChart";
import StatusCircle from "./components/StatusCircle";
import SummaryCard from "./components/SummaryCard";

export default function App() {
  return (
    <div className="grid grid-cols-6 min-h-screen font-sans bg-gray-50">
      <Sidebar />
      <main className="col-span-5 p-6 space-y-6">
        <Topbar user="Lucy" status="Lifting" />
        <section className="grid grid-cols-3 gap-6">
          <div className="col-span-2">
            <EMGChart />
          </div>
          <div className="col-span-1 flex flex-col gap-4">
            <StatusCircle value={73} label="当前激活率" />
            <SummaryCard title="最大肌电值" value="0.92" />
            <SummaryCard title="最近动作" value="Lifting" />
          </div>
        </section>
      </main>
    </div>
  );
}
