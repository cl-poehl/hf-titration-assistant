import { useEffect, useState } from "react";
import { TopBar } from "./components/TopBar";
import { PatientList } from "./components/PatientList";
import { PatientDetail } from "./components/PatientDetail";
import { usePatients } from "./hooks/usePatients";
import { Activity, Loader2, Wifi, WifiOff } from "lucide-react";

export default function App() {
  const { patients, loading, error, isLive } = usePatients();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // Auto-select highest-risk patient once data loads
  useEffect(() => {
    if (patients.length > 0 && selectedId === null) {
      const sorted = [...patients].sort(
        (a, b) => b.risk_score - a.risk_score,
      );
      setSelectedId(sorted[0]?.id ?? null);
    }
  }, [patients, selectedId]);

  const selectedPatient = patients.find((p) => p.id === selectedId) ?? null;

  return (
    <div className="h-screen flex flex-col bg-slate-50">
      <TopBar patients={patients} />

      {/* Connection status banner */}
      {!loading && (
        <div
          className={`flex items-center gap-2 px-4 py-1.5 text-xs font-medium ${
            isLive
              ? "bg-emerald-50 text-emerald-700"
              : "bg-amber-50 text-amber-700"
          }`}
        >
          {isLive ? (
            <>
              <Wifi size={14} />
              <span>
                Live — {patients.length} patients from prediction service
              </span>
            </>
          ) : (
            <>
              <WifiOff size={14} />
              <span>
                Demo mode — API unavailable{error ? ` (${error})` : ""}.
                Showing sample data.
              </span>
            </>
          )}
        </div>
      )}

      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar: Patient list */}
        <aside className="w-80 flex-shrink-0 bg-white border-r border-slate-200 overflow-hidden flex flex-col">
          {loading ? (
            <div className="flex items-center justify-center h-full text-slate-400">
              <Loader2 size={24} className="animate-spin mr-2" />
              <span className="text-sm">Loading patients…</span>
            </div>
          ) : (
            <PatientList
              patients={patients}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
        </aside>

        {/* Main content */}
        <main className="flex-1 overflow-hidden bg-slate-50">
          {loading ? (
            <div className="flex items-center justify-center h-full text-slate-400">
              <Loader2 size={32} className="animate-spin" />
            </div>
          ) : selectedPatient ? (
            <PatientDetail patient={selectedPatient} />
          ) : (
            <div className="flex items-center justify-center h-full text-slate-400">
              <div className="text-center">
                <Activity size={48} className="mx-auto mb-3 text-slate-300" />
                <p className="text-sm">Select a patient to view details</p>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
