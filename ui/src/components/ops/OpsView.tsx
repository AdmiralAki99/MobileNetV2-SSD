import { useState, useEffect } from 'react'
import { DagGraph, type Task } from './DagGraph'
import { StatusBadge } from '../StatusBadge'
import { PillButton } from '../PillButton'
import { fmtDur, fmtTime, fmtDate } from './utils'
import { usePolling } from '../../api/hooks'
import { fetchDags, fetchAirflow, fetchAirflowRuns, fetchAirflowRunTasks, fetchRay } from '../../api/client'

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

interface DagInfo {
  dag_id: string
  label: string
  schedule: string
}


const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div style={{
    fontSize: '9px', fontWeight: 700, letterSpacing: '1.4px',
    textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)',
    marginBottom: 10, paddingLeft: 2,
  }}>
    {children}
  </div>
)

const Glass = ({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) => (
  <div style={{ position: 'relative', ...style }}>
    <svg aria-hidden style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none', zIndex: 1 }} width="14" height="14">
      <path d="M14 1 L1 1 L1 14" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
    </svg>
    <svg aria-hidden style={{ position: 'absolute', top: 0, right: 0, pointerEvents: 'none', zIndex: 1 }} width="14" height="14">
      <path d="M0 1 L13 1 L13 14" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
    </svg>
    <svg aria-hidden style={{ position: 'absolute', bottom: 0, left: 0, pointerEvents: 'none', zIndex: 1 }} width="14" height="14">
      <path d="M14 13 L1 13 L1 0" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
    </svg>
    <svg aria-hidden style={{ position: 'absolute', bottom: 0, right: 0, pointerEvents: 'none', zIndex: 1 }} width="14" height="14">
      <path d="M0 13 L13 13 L13 0" fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
    </svg>
    {children}
  </div>
)

const ColHeader = ({ children }: { children: React.ReactNode }) => (
  <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.2)', fontWeight: 700, letterSpacing: '0.8px', textTransform: 'uppercase' }}>
    {children}
  </span>
)


export const OpsView = () => {
  const [dagId,       setDagId]       = useState('etl_pipeline')
  const [selectedRun, setSelectedRun] = useState<AirflowRun | null>(null)
  const [runTasks,    setRunTasks]    = useState<Task[] | null>(null)

  const { data: dagsList } = usePolling(fetchDags, 60000)
  const { data: airflow }  = usePolling(() => fetchAirflow(dagId) as Promise<AirflowData>, 10000)
  const { data: runs }     = usePolling(() => fetchAirflowRuns(dagId) as Promise<AirflowRun[]>, 10000)
  const { data: ray }      = usePolling(fetchRay as () => Promise<RayData>, 15000)

  const dags: DagInfo[]       = dagsList ?? [{ dag_id: 'etl_pipeline', label: 'ETL Pipeline', schedule: '0 2 * * *' }]
  const airflowData: AirflowData = airflow ?? {}
  const rayData: RayData         = ray ?? {}
  const runsData: AirflowRun[]   = runs ?? []
  const nodes = rayData.nodes ?? []
  const res   = rayData.resources ?? {}

  useEffect(() => { setSelectedRun(null) }, [dagId])

  useEffect(() => {
    if (!selectedRun) { setRunTasks(null); return }
    fetchAirflowRunTasks(selectedRun.run_id, dagId)
      .then((t: any) => setRunTasks(t))
      .catch(() => setRunTasks([]))
  }, [selectedRun?.run_id, dagId])

  const activeRun   = selectedRun ?? airflowData.last_run ?? {}
  const activeTasks = selectedRun ? (runTasks ?? []) : (airflowData.tasks ?? [])
  const activeDur   = (activeRun as AirflowRun).start_date && (activeRun as AirflowRun).end_date
    ? (new Date((activeRun as AirflowRun).end_date!).getTime() - new Date((activeRun as AirflowRun).start_date!).getTime()) / 1000
    : ((activeRun as AirflowRun).duration ?? null)

  const cpuPct = res.cpu_total && res.cpu_used != null ? (res.cpu_used / res.cpu_total) * 100 : null
  const memPct = res.memory_total_gb && res.memory_used_gb != null ? (res.memory_used_gb / res.memory_total_gb) * 100 : null

  return (
    <div
      data-testid="ops-view"
      style={{
        height: '100%', overflowY: 'auto',
        padding: '22px 24px 40px',
        display: 'flex', flexDirection: 'column', gap: 20,
        background: `
          radial-gradient(ellipse 55% 30% at 80% -5%, rgba(0,180,120,0.06) 0%, transparent 70%),
          radial-gradient(ellipse 40% 20% at 0% 85%,  rgba(100,120,180,0.05) 0%, transparent 60%),
          var(--bg-primary)
        `,
      }}
    >

      <div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
          <SectionLabel>Airflow DAG{selectedRun ? ' · history run' : ''}</SectionLabel>
          <div data-testid="dag-tabs" style={{ display: 'flex', gap: 4, paddingLeft: 2 }}>
            {dags.map(d => (
              <button
                key={d.dag_id}
                data-testid={`dag-tab-${d.dag_id}`}
                onClick={() => setDagId(d.dag_id)}
                style={{
                  fontSize: '10px', fontFamily: 'monospace', fontWeight: 600,
                  padding: '4px 10px', borderRadius: 4, cursor: 'pointer',
                  border: `1px solid ${dagId === d.dag_id ? 'rgba(0,212,160,0.35)' : 'rgba(255,255,255,0.08)'}`,
                  background: dagId === d.dag_id ? 'rgba(0,212,160,0.1)' : 'transparent',
                  color: dagId === d.dag_id ? '#00d4a0' : 'rgba(255,255,255,0.3)',
                  transition: 'all 0.15s',
                }}
              >
                {d.label}
              </button>
            ))}
          </div>
        </div>
        <Glass>
          <div data-testid="airflow-card" style={{ padding: '16px 18px 14px' }}>

            <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginBottom: 18 }}>
              <div>
                <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.2)', marginBottom: 5 }}>
                  DAG ID
                </div>
                <span data-testid="dag-id" style={{ fontSize: '14px', fontWeight: 700, color: '#dde2e0', fontFamily: 'monospace' }}>
                  {airflowData.dag_id ?? '—'}
                </span>
              </div>

              <div style={{ marginLeft: 'auto', display: 'flex', gap: 24, alignItems: 'flex-end' }}>
                <div>
                  <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '0.8px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.2)', marginBottom: 4 }}>Schedule</div>
                  <span data-testid="dag-schedule" style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)', fontFamily: 'monospace' }}>
                    {airflowData.schedule ?? '—'}
                  </span>
                </div>
                <div>
                  <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '0.8px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.2)', marginBottom: 4 }}>
                    {selectedRun ? 'Run' : 'Last Run'}
                  </div>
                  <StatusBadge status={(activeRun as AirflowRun).state ?? 'unknown'} />
                </div>
                <div>
                  <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '0.8px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.2)', marginBottom: 4 }}>Duration</div>
                  <span data-testid="dag-duration" style={{ fontSize: '11px', color: 'rgba(255,255,255,0.45)', fontFamily: 'monospace' }}>
                    {fmtDur(activeDur)}
                  </span>
                </div>
                {selectedRun && (
                  <button
                    data-testid="back-to-latest"
                    onClick={() => setSelectedRun(null)}
                    style={{ fontSize: '10px', color: 'rgba(255,255,255,0.3)', background: 'none', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 4, padding: '4px 10px', cursor: 'pointer', fontFamily: 'monospace', letterSpacing: '0.4px' }}
                  >
                    ✕ latest
                  </button>
                )}
              </div>
            </div>

            <DagGraph tasks={activeTasks} />
          </div>
        </Glass>
      </div>

      <div>
        <SectionLabel>Tasks &amp; Cluster</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 250px', gap: 14 }}>

          <Glass>
            <div data-testid="task-table" style={{ padding: '14px 16px 10px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 90px 64px 90px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: 4 }}>
                {['Task','Status','Duration','Start'].map(h => <ColHeader key={h}>{h}</ColHeader>)}
              </div>
              {activeTasks.length === 0 ? (
                <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', padding: '16px 4px', fontFamily: 'monospace' }}>No task runs.</div>
              ) : activeTasks.map(t => (
                <div key={t.task_id} data-testid={`task-row-${t.task_id}`}
                  style={{ display: 'grid', gridTemplateColumns: '1fr 90px 64px 90px', gap: 8, padding: '8px 4px', borderBottom: '1px solid rgba(255,255,255,0.04)', alignItems: 'center' }}>
                  <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.6)', fontFamily: 'monospace' }}>{t.task_id}</span>
                  <StatusBadge status={t.state} />
                  <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{fmtDur(t.duration)}</span>
                  <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>{fmtTime(t.start_date)}</span>
                </div>
              ))}
            </div>
          </Glass>

          <Glass>
            <div data-testid="ray-panel" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.2)' }}>
                  Ray Cluster
                </span>
                <StatusBadge status={rayData.status ?? 'stopped'} />
              </div>

              {cpuPct !== null && (
                <div data-testid="cpu-bar">
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.35)' }}>CPU</span>
                    <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>{res.cpu_used!.toFixed(1)} / {res.cpu_total!.toFixed(1)}</span>
                  </div>
                  <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${cpuPct}%`, background: '#8a9a6a', opacity: 0.65, transition: 'width 0.5s' }} />
                  </div>
                </div>
              )}

              {memPct !== null && (
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                    <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.35)' }}>Memory</span>
                    <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>{res.memory_used_gb!.toFixed(1)} / {res.memory_total_gb!.toFixed(1)} GB</span>
                  </div>
                  <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${memPct}%`, background: '#6a7a8a', opacity: 0.65, transition: 'width 0.5s' }} />
                  </div>
                </div>
              )}

              <div>
                <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', marginBottom: 8 }}>
                  Nodes ({nodes.length})
                </div>
                {nodes.length === 0 ? (
                  <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', fontFamily: 'monospace' }}>No nodes active.</span>
                ) : nodes.map(n => (
                  <div key={n.id} data-testid={`node-${n.id}`}
                    style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <div style={{ width: 5, height: 5, background: n.status === 'alive' ? '#8a9a6a' : '#8a6a6a', flexShrink: 0 }} />
                    <span style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.35)', fontFamily: 'monospace', flex: 1 }}>{n.ip ?? n.id}</span>
                  </div>
                ))}
              </div>

              {rayData.status === 'running' && rayData.dashboard_url && (
                <PillButton onClick={() => window.open(rayData.dashboard_url, '_blank')} style={{ width: '100%', justifyContent: 'center' }}>
                  Ray Dashboard →
                </PillButton>
              )}
            </div>
          </Glass>
        </div>
      </div>

      <div>
        <SectionLabel>Run History</SectionLabel>
        <Glass>
          <div data-testid="run-history" style={{ padding: '14px 16px 10px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 90px 80px 70px 120px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: 4 }}>
              {['Run ID','Status','Type','Duration','Started'].map(h => <ColHeader key={h}>{h}</ColHeader>)}
            </div>
            {runsData.length === 0 ? (
              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', padding: '16px 4px', fontFamily: 'monospace' }}>No runs yet.</div>
            ) : runsData.map(r => (
              <div
                key={r.run_id}
                data-testid={`run-row-${r.run_id}`}
                onClick={() => setSelectedRun(selectedRun?.run_id === r.run_id ? null : r)}
                style={{
                  display: 'grid', gridTemplateColumns: '1fr 90px 80px 70px 120px', gap: 8,
                  padding: '8px 4px', borderBottom: '1px solid rgba(255,255,255,0.04)', cursor: 'pointer',
                  borderLeft: `2px solid ${selectedRun?.run_id === r.run_id ? 'rgba(138,154,106,0.7)' : 'transparent'}`,
                  paddingLeft: 6,
                  background: selectedRun?.run_id === r.run_id ? 'rgba(138,154,106,0.05)' : 'transparent',
                  transition: 'background 0.12s',
                }}
              >
                <span style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.5)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.run_id}</span>
                <StatusBadge status={r.state} />
                <span style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{r.run_type ?? '—'}</span>
                <span style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{fmtDur(r.duration)}</span>
                <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>{fmtDate(r.start_date)}</span>
              </div>
            ))}
          </div>
        </Glass>
      </div>
    </div>
  )
}
