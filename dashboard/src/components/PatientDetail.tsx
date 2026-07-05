import type { Patient } from "../types/patient";
import { PatientHeader } from "./PatientHeader";
import { VitalCharts } from "./VitalCharts";
import { ExplanationPanel } from "./ExplanationPanel";
import { MedicationPanel } from "./MedicationPanel";

export function PatientDetail({ patient }: { patient: Patient }) {
  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="p-5 space-y-5">
        <PatientHeader patient={patient} />

        <div className="grid grid-cols-5 gap-5">
          {/* Vital charts — takes 3 cols */}
          <div className="col-span-3">
            <VitalCharts vitals={patient.vitals} />
          </div>

          {/* Explanation panel — takes 2 cols */}
          <div className="col-span-2">
            <ExplanationPanel
              factors={patient.top_factors}
              action={patient.suggested_action}
              tier={patient.risk_tier}
            />
          </div>
        </div>

        {/* GDMT Medication Panel */}
        {patient.medications && patient.medications.length > 0 && (
          <MedicationPanel
            medications={patient.medications}
            recommendations={patient.gdmt_recommendations ?? []}
            optimizationScore={patient.optimization_score ?? 0}
          />
        )}

        {/* Lab results table */}
        {patient.labs.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">
              Lab Results
            </h3>
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-slate-50/50">
                    <th className="px-4 py-2.5 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider">Day</th>
                    <th className="px-4 py-2.5 text-right text-xs font-semibold text-slate-400 uppercase tracking-wider">BNP (pg/mL)</th>
                    <th className="px-4 py-2.5 text-right text-xs font-semibold text-slate-400 uppercase tracking-wider">Creatinine (mg/dL)</th>
                    <th className="px-4 py-2.5 text-right text-xs font-semibold text-slate-400 uppercase tracking-wider">Potassium (mEq/L)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {patient.labs.map((lab, i) => (
                    <tr key={i} className="hover:bg-slate-50/50 transition-colors">
                      <td className="px-4 py-2.5 text-slate-600 font-medium">Day {lab.day}</td>
                      <td className={`px-4 py-2.5 text-right tabular-nums ${lab.bnp_pg_ml != null && lab.bnp_pg_ml > 600 ? "text-rose-600 font-semibold" : "text-slate-600"}`}>
                        {lab.bnp_pg_ml ?? <span className="text-slate-300">—</span>}
                      </td>
                      <td className={`px-4 py-2.5 text-right tabular-nums ${lab.creatinine_mg_dl != null && lab.creatinine_mg_dl > 1.5 ? "text-orange-600 font-semibold" : "text-slate-600"}`}>
                        {lab.creatinine_mg_dl ?? <span className="text-slate-300">—</span>}
                      </td>
                      <td className={`px-4 py-2.5 text-right tabular-nums ${lab.potassium_meq_l != null && (lab.potassium_meq_l > 5.0 || lab.potassium_meq_l < 3.5) ? "text-orange-600 font-semibold" : "text-slate-600"}`}>
                        {lab.potassium_meq_l ?? <span className="text-slate-300">—</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
