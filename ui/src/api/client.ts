import type { EtlStats, EtlVideo, EtlFrame, EtlAnnotation } from '../components/etl/etlTypes'

const API_BASE = ''

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

function apiPost<T>(path: string, body: unknown): Promise<T> {
  return apiFetch<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const fetchExperiments = () => apiFetch('/api/experiments')
export const fetchInstanceStatus = (id: string) => apiFetch(`/api/training/${id}/status`)
export const fetchArtifacts = (id: string) => apiFetch(`/api/export/${id}/artifacts`)
export const fetchJobStatus = (id: string) => apiFetch(`/api/export/jobs/${id}`)
export const fetchMetrics = (id: string) => apiFetch(`/api/experiments/${id}/metrics`)
export const fetchEtlStats = () => apiFetch('/api/etl/stats')
export const fetchEtlVideos = () => apiFetch('/api/etl/videos')
export const fetchEtlFrames = (id: string)  => apiFetch<EtlFrame[]>(`/api/etl/videos/${id}/frames`)
export const fetchEtlAnnotations = (id: string)  => apiFetch<EtlAnnotation[]>(`/api/etl/frames/${id}/annotations`)
export const fetchCicd = () => apiFetch('/api/deploy/cicd')
export const fetchReleases = () => apiFetch('/api/deploy/releases')
export const fetchAirflow = () => apiFetch('/api/ops/airflow')
export const fetchAirflowRuns = () => apiFetch('/api/ops/airflow/runs')
export const fetchAirflowRunTasks = (id: string) => apiFetch(`/api/ops/airflow/runs/${id}/tasks`)
export const fetchRay = () => apiFetch('/api/ops/ray')

export const launchTraining = (req: unknown) => apiPost('/api/training/launch', req)
export const stopTraining = (req: unknown) => apiPost('/api/training/stop', req)
export const exportSavedModel = (req: unknown) => apiPost('/api/export/savedmodel', req)
export const exportOnnx = (req: unknown) => apiPost('/api/export/onnx', req)

