import { Activity, Bell } from "lucide-react";
import type { Patient } from "../types/patient";

export function TopBar({ patients }: { patients: Patient[] }) {
  const critical = patients.filter((p) => p.risk_tier === "critical").length;
  const high = patients.filter((p) => p.risk_tier === "high").length;
  const unackedAlerts = patients.reduce(
    (sum, p) => sum + (p.alerts_total - p.alerts_acknowledged), 0
  );

  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-5">
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-blue-600">
          <Activity size={16} className="text-white" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-slate-900 leading-none">HF Titration Assistant</h1>
          <p className="text-[10px] text-slate-400 mt-0.5">Hospital-at-Home Monitoring</p>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4 text-xs">
          {critical > 0 && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-50 text-rose-600 font-semibold">
              <span className="h-1.5 w-1.5 rounded-full bg-rose-500 animate-pulse" />
              {critical} Critical
            </span>
          )}
          {high > 0 && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-orange-50 text-orange-600 font-semibold">
              {high} High
            </span>
          )}
        </div>

        <button className="relative p-2 rounded-lg hover:bg-slate-100 transition-colors">
          <Bell size={18} className="text-slate-500" />
          {unackedAlerts > 0 && (
            <span className="absolute -top-0.5 -right-0.5 h-4 w-4 flex items-center justify-center rounded-full bg-rose-500 text-[10px] font-bold text-white">
              {unackedAlerts}
            </span>
          )}
        </button>
      </div>
    </header>
  );
}
