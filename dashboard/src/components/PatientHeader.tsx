import { MapPin, Calendar, Heart } from "lucide-react";
import type { Patient } from "../types/patient";
import { RiskBadge } from "./RiskBadge";
import { RiskGauge } from "./RiskGauge";

export function PatientHeader({ patient }: { patient: Patient }) {
  const timeAgo = formatTimeAgo(patient.last_updated);

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5">
      <div className="flex items-start justify-between">
        {/* Left: Patient info */}
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h2 className="text-xl font-bold text-slate-900">{patient.name}</h2>
            <RiskBadge tier={patient.risk_tier} size="lg" />
          </div>

          <div className="flex items-center gap-4 mt-2 text-sm text-slate-500">
            <span>{patient.age}y {patient.sex === "M" ? "Male" : "Female"}</span>
            <span className="text-slate-300">|</span>
            <span>{patient.id}</span>
          </div>

          <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 mt-3 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <Heart size={12} /> {patient.diagnosis}
            </span>
            <span className="flex items-center gap-1.5">
              <MapPin size={12} /> {patient.room}
            </span>
            <span className="flex items-center gap-1.5">
              <Calendar size={12} /> Admitted {patient.admission_date}
            </span>
          </div>

          <div className="flex items-center gap-4 mt-3">
            <Stat label="EF" value={`${patient.ejection_fraction}%`} sub={patient.ef_category} />
            <div className="w-px h-8 bg-slate-100" />
            <Stat label="NYHA" value={`Class ${patient.nyha_class}`} />
            <div className="w-px h-8 bg-slate-100" />
            <Stat label="Updated" value={timeAgo} />
          </div>
        </div>

        {/* Right: Risk gauge */}
        <div className="flex-shrink-0 ml-6">
          <RiskGauge score={patient.risk_score} tier={patient.risk_tier} />
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div>
      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">{label}</span>
      <div className="text-sm font-semibold text-slate-700">
        {value}
        {sub && <span className="text-xs font-normal text-slate-400 ml-1">{sub}</span>}
      </div>
    </div>
  );
}

function formatTimeAgo(isoDate: string): string {
  const diff = Date.now() - new Date(isoDate).getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
