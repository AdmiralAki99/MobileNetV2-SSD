export type Status =
  | 'pending' | 'running' | 'success' | 'failed'
  | 'completed' | 'processing' | 'queued' | 'skipped'
  | 'upstream_failed' | 'current' | 'deprecated' | 'archived'

export const STATUS_COLOR: Record<Status, string> = {
  pending:         'var(--warning)',
  running:         'var(--accent)',
  success:         'var(--success)',
  failed:          'var(--danger)',
  completed:       'var(--success)',
  processing:      'var(--accent)',
  queued:          'var(--warning)',
  skipped:         'var(--text-tertiary)',
  upstream_failed: 'var(--danger)',
  current:         'var(--success)',
  deprecated:      'var(--warning)',
  archived:        'var(--text-tertiary)',
}
