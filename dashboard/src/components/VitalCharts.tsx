import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import type { VitalReading } from "../types/patient";

interface ChartConfig {
  label: string;
  key: keyof VitalReading;
  color: string;
  unit: string;
  warningHigh?: number;
  warningLow?: number;
}

const charts: ChartConfig[] = [
  { label: "Weight", key: "weight_kg", color: "#6366f1", unit: "kg" },
  { label: "SpO2", key: "spo2", color: "#10b981", unit: "%", warningLow: 92 },
  { label: "Heart Rate", key: "heart_rate", color: "#f43f5e", unit: "bpm", warningHigh: 100 },
  { label: "Respiratory Rate", key: "respiratory_rate", color: "#8b5cf6", unit: "/min", warningHigh: 20 },
  { label: "Systolic BP", key: "systolic_bp", color: "#f97316", unit: "mmHg" },
  { label: "Diastolic BP", key: "diastolic_bp", color: "#f59e0b", unit: "mmHg" },
];

function MiniChart({ config, data }: { config: ChartConfig; data: VitalReading[] }) {
  // Aggregate to daily averages for cleaner display
  const dailyMap = new Map<number, number[]>();
  data.forEach((v) => {
    const vals = dailyMap.get(v.day) || [];
    vals.push(v[config.key] as number);
    dailyMap.set(v.day, vals);
  });

  const chartData = Array.from(dailyMap.entries())
    .sort(([a], [b]) => a - b)
    .map(([day, vals]) => ({
      day: `D${day}`,
      value: +(vals.reduce((s, v) => s + v, 0) / vals.length).toFixed(1),
    }));

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          {config.label}
        </h4>
        <span className="text-lg font-bold text-slate-800 tabular-nums">
          {chartData[chartData.length - 1]?.value}
          <span className="text-xs font-normal text-slate-400 ml-0.5">{config.unit}</span>
        </span>
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
          <XAxis
            dataKey="day" tick={{ fontSize: 10, fill: "#94a3b8" }}
            axisLine={false} tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#94a3b8" }}
            axisLine={false} tickLine={false}
            domain={["auto", "auto"]}
          />
          <Tooltip
            contentStyle={{
              fontSize: 12, background: "#fff", border: "1px solid #e2e8f0",
              borderRadius: 8, boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.05)",
            }}
            formatter={(val) => [`${val} ${config.unit}`, config.label] as [string, string]}
          />
          {config.warningHigh && (
            <ReferenceLine y={config.warningHigh} stroke="#fbbf24" strokeDasharray="4 4" />
          )}
          {config.warningLow && (
            <ReferenceLine y={config.warningLow} stroke="#fbbf24" strokeDasharray="4 4" />
          )}
          <Line
            type="monotone" dataKey="value" stroke={config.color}
            strokeWidth={2} dot={false} activeDot={{ r: 3, strokeWidth: 0 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function VitalCharts({ vitals }: { vitals: VitalReading[] }) {
  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">
        Vital Trends
      </h3>
      <div className="grid grid-cols-3 gap-3">
        {charts.map((cfg) => (
          <MiniChart key={cfg.key} config={cfg} data={vitals} />
        ))}
      </div>
    </div>
  );
}
