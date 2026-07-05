import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";
import type { Patient } from "../types/patient";
import { RiskBadge } from "./RiskBadge";

function TrendIcon({ trend }: { trend: Patient["trend"] }) {
  if (trend === "rising") return <ArrowUpRight size={14} className="text-rose-500" />;
  if (trend === "falling") return <ArrowDownRight size={14} className="text-emerald-500" />;
  return <Minus size={14} className="text-slate-400" />;
}

function riskScoreColor(score: number): string {
  if (score >= 80) return "text-rose-600";
  if (score >= 60) return "text-orange-600";
  if (score >= 35) return "text-amber-600";
  return "text-emerald-600";
}

interface Props {
  patients: Patient[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export function PatientList({ patients, selectedId, onSelect }: Props) {
  const sorted = [...patients].sort((a, b) => b.risk_score - a.risk_score);

  return (
    <div className="flex flex-col h-full">
      <div className="px-5 py-4 border-b border-slate-200">
        <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">
          Patient Panel
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">{patients.length} active patients</p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {sorted.map((p) => {
          const isSelected = p.id === selectedId;
          const hasUnacked = p.alerts_total - p.alerts_acknowledged > 0;
          return (
            <button
              key={p.id}
              onClick={() => onSelect(p.id)}
              className={`w-full text-left px-5 py-3.5 border-b border-slate-100 transition-colors hover:bg-slate-50 ${
                isSelected ? "bg-blue-50/60 border-l-2 border-l-blue-500" : "border-l-2 border-l-transparent"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm text-slate-900 truncate">{p.name}</span>
                    {hasUnacked && (
                      <span className="flex-shrink-0 h-2 w-2 rounded-full bg-rose-500" />
                    )}
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="text-xs text-slate-400">{p.id}</span>
                    <span className="text-xs text-slate-300">|</span>
                    <span className="text-xs text-slate-400">{p.age}{p.sex}</span>
                    <span className="text-xs text-slate-300">|</span>
                    <span className="text-xs text-slate-400">EF {p.ejection_fraction}%</span>
                  </div>
                </div>
                <div className="flex flex-col items-end gap-1.5 ml-3">
                  <div className="flex items-center gap-1">
                    <span className={`text-lg font-bold tabular-nums ${riskScoreColor(p.risk_score)}`}>
                      {p.risk_score}
                    </span>
                    <TrendIcon trend={p.trend} />
                  </div>
                  <RiskBadge tier={p.risk_tier} />
                  {p.optimization_score != null && p.ef_category !== "HFpEF" && (
                    <span className="text-[10px] font-medium text-blue-500 tabular-nums">
                      GDMT {Math.round(p.optimization_score)}%
                    </span>
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
