import { useEffect, useState } from "react";
import type { Patient } from "../types/patient";
import { fetchPatients } from "../api/client";
import { mockPatients } from "../data/mockPatients";

interface UsePatientsResult {
  patients: Patient[];
  loading: boolean;
  error: string | null;
  isLive: boolean;
}

export function usePatients(): UsePatientsResult {
  const [patients, setPatients] = useState<Patient[]>(mockPatients);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await fetchPatients();
        if (!cancelled) {
          setPatients(data);
          setIsLive(true);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          // Keep mock data visible
          setIsLive(false);
          setError(
            err instanceof Error ? err.message : "Failed to reach API",
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return { patients, loading, error, isLive };
}
