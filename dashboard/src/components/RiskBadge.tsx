import type { RiskTier } from "../types/patient";

const tierConfig: Record<RiskTier, { bg: string; text: string; ring: string; label: string }> = {
  low: { bg: "bg-emerald-50", text: "text-emerald-700", ring: "ring-emerald-200", label: "Low" },
  medium: { bg: "bg-amber-50", text: "text-amber-700", ring: "ring-amber-200", label: "Medium" },
  high: { bg: "bg-orange-50", text: "text-orange-700", ring: "ring-orange-200", label: "High" },
  critical: { bg: "bg-rose-50", text: "text-rose-700", ring: "ring-rose-200", label: "Critical" },
};

export function RiskBadge({ tier, size = "sm" }: { tier: RiskTier; size?: "sm" | "lg" }) {
  const cfg = tierConfig[tier];
  const sizeClass = size === "lg" ? "px-3 py-1.5 text-sm" : "px-2 py-0.5 text-xs";
  return (
    <span className={`inline-flex items-center font-semibold rounded-full ring-1 ${cfg.bg} ${cfg.text} ${cfg.ring} ${sizeClass}`}>
      {tier === "critical" && (
        <span className="mr-1.5 h-2 w-2 rounded-full bg-rose-500 animate-pulse" />
      )}
      {cfg.label}
    </span>
  );
}
