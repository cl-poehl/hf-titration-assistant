import { ArrowUp, ArrowDown } from "lucide-react";
import type { ContributingFactor, SuggestedAction, RiskTier } from "../types/patient";

function FactorBar({ factor, maxImpact }: { factor: ContributingFactor; maxImpact: number }) {
  const pct = Math.min(Math.abs(factor.impact) / maxImpact * 100, 100);
  const isRisk = factor.direction === "increasing_risk";

  return (
    <div className="flex items-center gap-3 py-2">
      <div className="flex-shrink-0 w-5 flex justify-center">
        {isRisk
          ? <ArrowUp size={14} className="text-rose-500" />
          : <ArrowDown size={14} className="text-emerald-500" />
        }
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-sm font-medium text-slate-700 truncate">
            {factor.display_name}
          </span>
          <span className="text-xs font-mono text-slate-400 ml-2 flex-shrink-0">
            {factor.value > 0 ? "+" : ""}{factor.value}
          </span>
        </div>
        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ${
              isRisk ? "bg-rose-400" : "bg-emerald-400"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}

const urgencyStyles: Record<string, { bg: string; border: string; icon: string }> = {
  emergent: { bg: "bg-rose-50", border: "border-rose-200", icon: "text-rose-600" },
  urgent: { bg: "bg-orange-50", border: "border-orange-200", icon: "text-orange-600" },
  soon: { bg: "bg-amber-50", border: "border-amber-200", icon: "text-amber-600" },
  routine: { bg: "bg-slate-50", border: "border-slate-200", icon: "text-slate-500" },
};

export function ExplanationPanel({
  factors, action, tier,
}: {
  factors: ContributingFactor[];
  action: SuggestedAction;
  tier: RiskTier;
}) {
  const maxImpact = Math.max(...factors.map((f) => Math.abs(f.impact)), 0.01);
  const style = urgencyStyles[action.urgency] || urgencyStyles.routine;

  return (
    <div className="space-y-5">
      {/* Suggested Action */}
      <div className={`rounded-xl border ${style.border} ${style.bg} p-4`}>
        <div className="flex items-center gap-2 mb-2">
          <span className={`text-xs font-bold uppercase tracking-wider ${style.icon}`}>
            {action.urgency}
          </span>
          <span className="text-xs text-slate-400">Suggested Action</span>
        </div>
        <p className="text-sm font-medium text-slate-800 leading-relaxed">
          {action.action}
        </p>
        <p className="text-xs text-slate-500 mt-2 leading-relaxed">
          {action.rationale}
        </p>
      </div>

      {/* Contributing Factors */}
      <div>
        <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-2">
          Contributing Factors
        </h3>
        <div className="bg-white rounded-xl border border-slate-200 p-4">
          <div className="divide-y divide-slate-50">
            {factors.map((f) => (
              <FactorBar key={f.feature} factor={f} maxImpact={maxImpact} />
            ))}
          </div>
          <p className="text-[10px] text-slate-300 mt-3 pt-2 border-t border-slate-100">
            SHAP-derived impact on patient stability and GDMT tolerance. Each bar
            shows how much this factor shifts the prediction relative to the
            population baseline.
          </p>
        </div>
      </div>
    </div>
  );
}
