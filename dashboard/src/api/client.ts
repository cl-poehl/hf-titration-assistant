import type { Patient } from "../types/patient";

const BASE = "/api";

export async function fetchPatients(): Promise<Patient[]> {
  const res = await fetch(`${BASE}/patients`);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  const data = await res.json();
  return data.patients as Patient[];
}

export async function fetchPatient(id: string): Promise<Patient> {
  const res = await fetch(`${BASE}/patients/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return (await res.json()) as Patient;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}
