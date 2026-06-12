export const fmtDur = (s: number | null | undefined): string => {
  if (s == null) return '—'
  return s >= 60 ? `${(s / 60).toFixed(1)}m` : `${s.toFixed(1)}s`
}

export const fmtTime = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export const fmtDate = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })
}

export const TASK_COLOR: Record<string, string> = {
  success:         '#65c16a',
  failed:          '#e84855',
  upstream_failed: '#e84855',
  running:         '#00d4a0',
  queued:          '#e88548',
  pending:         '#e88548',
  skipped:         '#494e4d',
}
