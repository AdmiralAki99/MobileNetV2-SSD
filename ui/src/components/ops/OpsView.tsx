import { useState, useEffect } from 'react'
import { DagGraph, type Task } from './DagGraph'
import { StatusBadge } from '../StatusBadge'
import { PillButton } from '../PillButton'
import { fmtDur, fmtTime, fmtDate } from './utils'
import { fetchAirflowRunTasks } from '../../api/client'

interface AirflowRun {
  run_id: string
  state: string
  run_type?: string
  duration?: number
  start_date?: string
  end_date?: string
}

interface AirflowData {
  dag_id?: string
  schedule?: string
  last_run?: AirflowRun
  tasks?: Task[]
}

interface RayNode {
  id: string
  ip?: string
  status: string
  cpu_pct?: number
  instance_type?: string
}

interface RayData {
  status?: string
  dashboard_url?: string
  resources?: {
    cpu_used?: number
    cpu_total?: number
    memory_used_gb?: number
    memory_total_gb?: number
  }
  nodes?: RayNode[]
}

interface Props {
  airflowData?: AirflowData
  rayData?: RayData
  runsData?: AirflowRun[]
}

export const OpsView = ({ airflowData = {}, rayData = {}, runsData = [] }: Props) => {
  const [selectedRun, setSelectedRun] = useState<AirflowRun | null>(null)
  const [runTasks,    setRunTasks]    = useState<Task[] | null>(null)
  const nodes = rayData.nodes ?? []
  const res   = rayData.resources ?? {}

  useEffect(() => {
    if (!selectedRun) { setRunTasks(null); return }
    fetchAirflowRunTasks(selectedRun.run_id)
      .then((t: any) => setRunTasks(t))
      .catch(() => setRunTasks([]))
  }, [selectedRun?.run_id])

  const activeRun   = selectedRun ?? airflowData.last_run ?? {}
  const activeTasks = selectedRun ? (runTasks ?? []) : (airflowData.tasks ?? [])
  const activeDur   = (activeRun as AirflowRun).start_date && (activeRun as AirflowRun).end_date
    ? (new Date((activeRun as AirflowRun).end_date!).getTime() - new Date((activeRun as AirflowRun).start_date!).getTime()) / 1000
    : ((activeRun as AirflowRun).duration ?? null)

  return (
    <div data-testid="ops-view" style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Airflow card */}
      <div data-testid="airflow-card" style={{ padding: '14px 18px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 18 }}>
          <div>
            <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-tertiary)', letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: 3 }}>
              Airflow DAG {selectedRun && <span style={{ color: 'var(--accent)' }}>· history</span>}
            </div>
            <span data-testid="dag-id" style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>
              {airflowData.dag_id ?? '—'}
            </span>
          </div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 20, alignItems: 'center' }}>
            <div>
              <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>Schedule</div>
              <span data-testid="dag-schedule" style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                {airflowData.schedule ?? '—'}
              </span>
            </div>
            <div>
              <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>{selectedRun ? 'Run' : 'Last Run'}</div>
              <StatusBadge status={(activeRun as AirflowRun).state ?? 'unknown'} />
            </div>
            <div>
              <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>Duration</div>
              <span data-testid="dag-duration" style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                {fmtDur(activeDur)}
              </span>
            </div>
            {selectedRun && (
              <button data-testid="back-to-latest" onClick={() => setSelectedRun(null)}
                style={{ fontSize: '11px', color: 'var(--text-tertiary)', background: 'none', border: '1px solid var(--border-subtle)', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', fontFamily: 'inherit' }}>
                ✕ latest
              </button>
            )}
          </div>
        </div>
        <DagGraph tasks={activeTasks} />
      </div>

      {/* Task table + Ray */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 260px', gap: 16 }}>
        <div data-testid="task-table" style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 12, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Task Runs</div>
          {activeTasks.length === 0
            ? <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '16px 4px' }}>No task runs.</div>
            : activeTasks.map(t => (
              <div key={t.task_id} data-testid={`task-row-${t.task_id}`}
                style={{ display: 'grid', gridTemplateColumns: '1fr 90px 64px 90px', gap: 8, padding: '8px 4px', borderBottom: '1px solid var(--border-subtle)', alignItems: 'center' }}>
                <span style={{ fontSize: '12px', color: 'var(--text-primary)', fontFamily: 'monospace' }}>{t.task_id}</span>
                <StatusBadge status={t.state} />
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtDur(t.duration)}</span>
                <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{fmtTime(t.start_date)}</span>
              </div>
            ))
          }
        </div>

        <div data-testid="ray-panel" style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', letterSpacing: '0.5px', textTransform: 'uppercase' }}>Ray Cluster</div>
            <StatusBadge status={rayData.status ?? 'stopped'} />
          </div>
          {res.cpu_total != null && res.cpu_used != null && (
            <div data-testid="cpu-bar">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>CPU</span>
                <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{res.cpu_used.toFixed(1)} / {res.cpu_total.toFixed(1)}</span>
              </div>
              <div style={{ height: 4, borderRadius: 999, background: 'var(--bg-pill)', overflow: 'hidden' }}>
                <div style={{ height: '100%', borderRadius: 999, width: `${(res.cpu_used / res.cpu_total) * 100}%`, background: 'var(--accent)' }} />
              </div>
            </div>
          )}
          <div>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 10, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
              Nodes ({nodes.length})
            </div>
            {nodes.length === 0
              ? <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>No nodes active.</span>
              : nodes.map(n => (
                <div key={n.id} data-testid={`node-${n.id}`}
                  style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                  <div style={{ width: 7, height: 7, borderRadius: '50%', background: n.status === 'alive' ? '#65c16a' : '#e84855' }} />
                  <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)', fontFamily: 'monospace', flex: 1 }}>{n.ip ?? n.id}</span>
                </div>
              ))
            }
          </div>
          {rayData.status === 'running' && rayData.dashboard_url && (
            <PillButton onClick={() => window.open(rayData.dashboard_url, '_blank')} style={{ width: '100%', justifyContent: 'center' }}>
              Ray Dashboard →
            </PillButton>
          )}
        </div>
      </div>

      {/* Run history */}
      <div data-testid="run-history" style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 12, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Run History</div>
        {runsData.length === 0
          ? <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '16px 4px' }}>No runs yet.</div>
          : runsData.map(r => (
            <div key={r.run_id} data-testid={`run-row-${r.run_id}`}
              onClick={() => setSelectedRun(selectedRun?.run_id === r.run_id ? null : r)}
              style={{
                display: 'grid', gridTemplateColumns: '1fr 90px 80px 70px 120px', gap: 8,
                padding: '8px 4px', borderBottom: '1px solid var(--border-subtle)', cursor: 'pointer',
                background: selectedRun?.run_id === r.run_id ? 'rgba(0,212,160,0.06)' : 'transparent',
              }}>
              <span style={{ fontSize: '11.5px', color: 'var(--text-primary)', fontFamily: 'monospace' }}>{r.run_id}</span>
              <StatusBadge status={r.state} />
              <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{r.run_type ?? '—'}</span>
              <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtDur(r.duration)}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{fmtDate(r.start_date)}</span>
            </div>
          ))
        }
      </div>
    </div>
  )
}
