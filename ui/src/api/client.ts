import type { EtlStats, EtlVideo, EtlFrame, EtlAnnotation } from '../components/etl/etlTypes'
import type { ConfigLibrary, RegisterRequest, RegisterResponse } from '../components/create/createTypes'



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
export const fetchDags = () => apiFetch<{ dag_id: string; label: string; schedule: string }[]>('/api/ops/dags')
export const fetchAirflow = (dagId = 'etl_pipeline') => apiFetch(`/api/ops/airflow?dag_id=${encodeURIComponent(dagId)}`)
export const fetchAirflowRuns = (dagId = 'etl_pipeline') => apiFetch(`/api/ops/airflow/runs?dag_id=${encodeURIComponent(dagId)}`)
export const fetchAirflowRunTasks = (id: string, dagId = 'etl_pipeline') =>
  apiFetch(`/api/ops/airflow/runs/${id}/tasks?dag_id=${encodeURIComponent(dagId)}`)
export const fetchRay = () => apiFetch('/api/ops/ray')

export const launchTraining = (req: unknown) => apiPost('/api/training/launch', req)
export const stopTraining = (req: unknown) => apiPost('/api/training/stop', req)
export const exportSavedModel = (req: unknown) => apiPost('/api/export/savedmodel', req)
export const exportOnnx = (req: unknown) => apiPost('/api/export/onnx', req)

export const fetchConfigLibrary   = () => apiFetch<ConfigLibrary>('/api/experiments/config-library')
export const refreshConfigLibrary = () => apiPost<{ status: string }>('/api/experiments/config-library/refresh', {})
export const saveConfig = (req: { category: string; name: string; content_yaml: string }) =>
  apiPost<{ path: string; name: string }>('/api/experiments/config-library/save', req)
export const registerExperiment  = (req: RegisterRequest)  => apiPost<RegisterResponse>('/api/experiments/register', req)

export const fetchDatasets   = () => apiFetch<{ datasets: import('../components/anchors/anchorTypes').DatasetEntry[] }>('/api/preprocessing/datasets').then(r => r.datasets)
export const fetchPriors     = () => apiFetch<{ priors: string[] }>('/api/preprocessing/priors').then(r => r.priors)
export const fetchBoxDims    = (dataset: string, split: string) =>
  apiFetch<{ points: [number, number][] }>(`/api/preprocessing/ledger/box-sizes?dataset=${encodeURIComponent(dataset)}&split=${encodeURIComponent(split)}`)
    .then(r => ({ norm: r.points }))
export const deriveCluster   = (req: unknown) => apiPost<{ status: number; result: import('../components/anchors/anchorTypes').ClusterResult }>('/api/preprocessing/clustering/derive', req)
export const exportCluster   = (req: unknown) => apiPost<{ status: number; result: import('../components/anchors/anchorTypes').ClusterResult }>('/api/preprocessing/clustering/export', req)
export const launchTfrecords = (req: unknown) => apiPost<{ status: number; dag_run_id: string }>('/api/preprocessing/tfrecords/launch', req)

