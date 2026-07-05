import { Check, X, ArrowUp, Minus, Shield, Pill } from "lucide-react";
import type {
  Medication,
  TitrationRecommendation,
  TitrationSafetyCheck,
} from "../types/patient";

// ---------------------------------------------------------------------------
// Drug class display helpers
// ---------------------------------------------------------------------------
const DRUG_CLASS_LABELS: Record<string, string> = {
  raasi: "RAASi",
  beta_blocker: "Beta-blocker",
  mra: "MRA",
  sglt2i: "SGLT2i",
};

const ACTION_STYLES: Record<
  string,
  { bg: string; text: string; label: string }
> = {
  uptitrate: { bg: "bg-emerald-100", text: "text-emerald-700", label: "Uptitrate" },
  initiate: { bg: "bg-blue-100", text: "text-blue-700", label: "Initiate" },
  hold: { bg: "bg-amber-100", text: "text-amber-700", label: "Hold" },
  at_target: { bg: "bg-slate-100", text: "text-slate-600", label: "At target" },
  not_indicated: { bg: "bg-slate-50", text: "text-slate-400", label: "N/A" },
};

// ---------------------------------------------------------------------------
// Optimization Gauge
// ---------------------------------------------------------------------------
function OptimizationGauge({ score }: { score: number }) {
  const color =
    score >= 75 ? "text-emerald-600" : score >= 50 ? "text-amber-500" : "text-rose-500";
  const ring =
    score >= 75 ? "stroke-emerald-500" : score >= 50 ? "stroke-amber-400" : "stroke-rose-500";
  const circumference = 2 * Math.PI * 36;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="flex items-center gap-4">
      <div className="relative w-20 h-20">
        <svg viewBox="0 0 80 80" className="w-full h-full -rotate-90">
          <circle cx="40" cy="40" r="36" fill="none" strokeWidth="6" className="stroke-slate-100" />
          <circle
            cx="40"
            cy="40"
            r="36"
            fill="none"
            strokeWidth="6"
            strokeLinecap="round"
            className={ring}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.7s ease" }}
          />
        </svg>
        <span className={`absolute inset-0 flex items-center justify-center text-lg font-bold ${color}`}>
          {Math.round(score)}%
        </span>
      </div>
      <div>
        <p className="text-sm font-semibold text-slate-700">GDMT Optimization</p>
        <p className="text-xs text-slate-400">% of target doses achieved</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Safety Check Indicator
// ---------------------------------------------------------------------------
function SafetyIndicator({ check }: { check: TitrationSafetyCheck }) {
  return (
    <div className="flex items-center gap-1.5 text-xs">
      {check.passed ? (
        <Check size={12} className="text-emerald-500" />
      ) : (
        <X size={12} className="text-rose-500" />
      )}
      <span className={check.passed ? "text-slate-500" : "text-rose-600 font-medium"}>
        {check.check_name}
      </span>
      {check.current_value !== null && (
        <span className="text-slate-400 tabular-nums">
          ({check.current_value})
        </span>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dose Progress Bar
// ---------------------------------------------------------------------------
function DoseBar({
  current,
  target,
}: {
  current: number;
  target: number;
}) {
  const pct = target > 0 ? Math.min((current / target) * 100, 100) : 0;
  return (
    <div className="mt-1.5">
      <div className="flex justify-between text-[10px] text-slate-400 mb-0.5 tabular-nums">
        <span>{current} mg</span>
        <span>{target} mg target</span>
      </div>
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full bg-blue-400 transition-all duration-700"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-Pillar Card
// ---------------------------------------------------------------------------
function PillarCard({
  med,
  rec,
}: {
  med: Medication;
  rec?: TitrationRecommendation;
}) {
  const style = ACTION_STYLES[rec?.action ?? "not_indicated"] ?? ACTION_STYLES.not_indicated;
  const isNotIndicated = med.status === "not_indicated";

  return (
    <div
      className={`rounded-xl border p-4 ${
        isNotIndicated
          ? "border-slate-100 bg-slate-50/50 opacity-60"
          : "border-slate-200 bg-white"
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Pill size={14} className="text-slate-400" />
          <span className="text-sm font-semibold text-slate-700">
            {DRUG_CLASS_LABELS[med.drug_class] ?? med.drug_class}
          </span>
        </div>
        {rec && (
          <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${style.bg} ${style.text}`}>
            {style.label}
          </span>
        )}
      </div>

      {/* Drug name */}
      <p className="text-xs text-slate-500 mb-2">
        {med.generic_name.replace(/_/g, "/")}
      </p>

      {/* Dose bar */}
      {!isNotIndicated && (
        <DoseBar current={med.current_dose_mg} target={med.target_dose_mg} />
      )}

      {/* Safety checks */}
      {rec && rec.safety_checks.length > 0 && (
        <div className="mt-3 space-y-1">
          {rec.safety_checks.map((c) => (
            <SafetyIndicator key={c.check_name} check={c} />
          ))}
        </div>
      )}

      {/* Tolerance score */}
      {rec?.tolerance_score != null && (
        <div className="mt-3 flex items-center gap-1.5">
          <Shield size={12} className="text-blue-400" />
          <span className="text-xs text-slate-500">
            Tolerance:{" "}
            <span className="font-semibold text-slate-700">{rec.tolerance_score}%</span>
          </span>
        </div>
      )}

      {/* Rationale */}
      {rec && !isNotIndicated && (
        <p className="text-[10px] text-slate-400 mt-2 leading-relaxed">
          {rec.rationale}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Titration Action Bar (summary of actionable items)
// ---------------------------------------------------------------------------
function TitrationActionBar({ recommendations }: { recommendations: TitrationRecommendation[] }) {
  const actionable = recommendations.filter(
    (r) => r.action === "uptitrate" || r.action === "initiate"
  );
  if (actionable.length === 0) return null;

  return (
    <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3">
      <div className="flex items-center gap-2 mb-1.5">
        <ArrowUp size={14} className="text-emerald-600" />
        <span className="text-xs font-bold uppercase tracking-wider text-emerald-700">
          Titration Opportunities
        </span>
      </div>
      <div className="space-y-1">
        {actionable.map((r) => (
          <p key={r.drug_class} className="text-sm text-emerald-800">
            <span className="font-medium">
              {DRUG_CLASS_LABELS[r.drug_class]}:
            </span>{" "}
            {r.action === "initiate" ? "Start" : "Increase"}{" "}
            {r.generic_name.replace(/_/g, "/")}{" "}
            {r.next_dose_mg != null && <span className="tabular-nums">to {r.next_dose_mg} mg</span>}
          </p>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main MedicationPanel
// ---------------------------------------------------------------------------
export function MedicationPanel({
  medications,
  recommendations,
  optimizationScore,
}: {
  medications: Medication[];
  recommendations: TitrationRecommendation[];
  optimizationScore: number;
}) {
  // Build a map from drug_class → recommendation for quick lookup
  const recMap = new Map(recommendations.map((r) => [r.drug_class, r]));

  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">
        GDMT Status
      </h3>

      <div className="space-y-4">
        {/* Top row: gauge + action bar */}
        <div className="flex flex-wrap items-start gap-4">
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <OptimizationGauge score={optimizationScore} />
          </div>
          <div className="flex-1 min-w-[260px]">
            <TitrationActionBar recommendations={recommendations} />
            {recommendations.filter((r) => r.action === "uptitrate" || r.action === "initiate").length === 0 && (
              <div className="rounded-xl border border-slate-200 bg-white p-3 flex items-center gap-2">
                <Minus size={14} className="text-slate-400" />
                <span className="text-sm text-slate-500">
                  No titration changes recommended at this time.
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Per-pillar cards */}
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {medications.map((med) => (
            <PillarCard
              key={med.drug_class}
              med={med}
              rec={recMap.get(med.drug_class)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
