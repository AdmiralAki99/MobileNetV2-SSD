import { useState } from 'react'
import { StatusBadge } from '../StatusBadge'
import { fmtDur } from '../ops/utils'
import { TASK_COLOR } from '../ops/utils'

interface Stage {
  id: string
  name: string
  status: string
  duration?: number | null
}

interface CiRun {
  id: string
  branch: string
  commit: string
  commit_message?: string
  trigger?: string
  started_at?: string
  status: string
  stages?: Stage[]
}

interface Release {
  version: string
  model: string
  experiment: string
  map_score: number
  released_at?: string
  status: string
  targets?: string[]
  artifacts?: { saved_model: boolean; onnx: boolean; tensorrt: boolean }
}

interface Props {
  cicdData?: { current_run?: CiRun; recent_runs?: CiRun[] }
  releasesData?: Release[]
}

const STAGE_LOGS: Record<string, string[]> = {
  lint:        ['$ ruff check src/', 'All checks passed.', '✓ Lint & type check complete (18.4s)'],
  unit:        ['$ pytest tests/unit/ -v', '32 passed in 42.1s'],
  integration: ['$ pytest tests/integration/ -v', 'tests/integration/test_etl_pipeline.py::test_consensus_engine RUNNING...'],
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


const StagePipeline = ({ stages, selected, onSelect }: { stages: Stage[]; selected: Stage | null; onSelect: (s: Stage | null) => void }) => (
  <div data-testid="stage-pipeline" style={{ display: 'flex', alignItems: 'flex-start', overflowX: 'auto', padding: '6px 0 2px', gap: 0 }}>
    {stages.map((stage, i) => {
      const isLast = i === stages.length - 1
      const isSel  = selected?.id === stage.id
      const color  = TASK_COLOR[stage.status] ?? 'rgba(255,255,255,0.15)'
      return (
        <div key={stage.id} style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
          <div
            data-testid={`stage-${stage.id}`}
            onClick={() => onSelect(isSel ? null : stage)}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 82, cursor: 'pointer' }}
          >
            <div style={{
              width: 32, height: 32,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: stage.status === 'pending' ? 'transparent' : `${color}14`,
              border: `1px solid ${stage.status === 'pending' ? 'rgba(255,255,255,0.08)' : `${color}66`}`,
              outline: isSel ? `1px solid ${color}44` : 'none',
              outlineOffset: 3,
            }}>
              {stage.status === 'success' && <span style={{ fontSize: 12, color, opacity: 0.9 }}>✓</span>}
              {stage.status === 'failed'  && <span style={{ fontSize: 12, color, opacity: 0.9 }}>✗</span>}
              {stage.status === 'running' && <span style={{ fontSize: 10, color, opacity: 0.9 }}>▶</span>}
            </div>
            <span style={{ fontSize: '9.5px', color: isSel ? '#8a9a6a' : 'rgba(255,255,255,0.35)', marginTop: 6, textAlign: 'center', whiteSpace: 'nowrap' }}>
              {stage.name}
            </span>
            <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.18)', fontFamily: 'monospace', marginTop: 2 }}>
              {stage.duration != null ? fmtDur(stage.duration) : ''}
            </span>
          </div>
          {!isLast && (
            <div style={{ width: 18, height: 1, flexShrink: 0, marginBottom: 22, background: 'rgba(255,255,255,0.06)' }} />
          )}
        </div>
      )
    })}
  </div>
)


export const DeployView = ({ cicdData, releasesData = [] }: Props) => {
  const run        = cicdData?.current_run ?? {} as CiRun
  const recentRuns = cicdData?.recent_runs ?? []
  const [selectedStage, setSelectedStage] = useState<Stage | null>(null)

  return (
    <div
      data-testid="deploy-view"
      style={{
        height: '100%', overflowY: 'auto',
        padding: '22px 24px 40px',
        display: 'flex', flexDirection: 'column', gap: 20,
        background: `
          radial-gradient(ellipse 50% 28% at 20% -5%, rgba(0,160,100,0.06) 0%, transparent 70%),
          radial-gradient(ellipse 40% 20% at 95% 80%, rgba(160,100,40,0.05) 0%, transparent 60%),
          var(--bg-primary)
        `,
      }}
    >

      <div>
        <SectionLabel>Current Run</SectionLabel>
        <Glass>
          <div data-testid="current-run-card" style={{ padding: '16px 18px 14px' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 18 }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                  <span data-testid="run-branch" style={{ fontSize: '13px', fontWeight: 700, color: '#dde2e0', fontFamily: 'monospace' }}>
                    {run.branch}
                  </span>
                  <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace' }}>
                    @{run.commit}
                  </span>
                  <StatusBadge status={run.status} />
                </div>
                <div
                  data-testid="run-commit-message"
                  style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace' }}
                >
                  {run.commit_message}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 24, flexShrink: 0, alignItems: 'flex-end' }}>
                <div>
                  <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '0.8px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.2)', marginBottom: 4 }}>Trigger</div>
                  <span data-testid="run-trigger" style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', fontFamily: 'monospace' }}>
                    {run.trigger ?? '—'}
                  </span>
                </div>
              </div>
            </div>
            <StagePipeline stages={run.stages ?? []} selected={selectedStage} onSelect={setSelectedStage} />
          </div>
        </Glass>
      </div>

      <div>
        <SectionLabel>Pipeline Detail</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '230px 1fr', gap: 14 }}>

          <Glass>
            <div data-testid="recent-runs" style={{ padding: '14px 16px 10px' }}>
              <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', marginBottom: 12 }}>
                Recent Runs
              </div>
              {recentRuns.length === 0 ? (
                <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', fontFamily: 'monospace' }}>No recent runs.</div>
              ) : recentRuns.map(r => (
                <div key={r.id} data-testid={`recent-run-${r.id}`}
                  style={{ padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
                    <span style={{ fontSize: '10.5px', fontFamily: 'monospace', color: 'rgba(255,255,255,0.5)' }}>{r.branch}</span>
                    <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>@{r.commit}</span>
                    <div style={{ marginLeft: 'auto' }}><StatusBadge status={r.status} /></div>
                  </div>
                </div>
              ))}
            </div>
          </Glass>

          <Glass>
            <div data-testid="stage-log" style={{ padding: '14px 16px', minHeight: 140 }}>
              {!selectedStage ? (
                <div style={{ color: 'rgba(255,255,255,0.15)', fontSize: '11px', fontFamily: 'monospace' }}>
                  Click a stage above to view logs
                </div>
              ) : (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                    <span data-testid="log-stage-name"
                      style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.3)' }}>
                      {selectedStage.name}
                    </span>
                    <StatusBadge status={selectedStage.status} />
                  </div>
                  <div data-testid="log-lines" style={{ fontFamily: 'monospace', fontSize: '11px', lineHeight: 1.85 }}>
                    {(STAGE_LOGS[selectedStage.id] ?? []).map((line, i) => (
                      <div key={i} style={{
                        color: line.startsWith('$') ? '#8a9a6a'
                          : line.startsWith('✓') ? 'rgba(138,154,106,0.7)'
                          : 'rgba(255,255,255,0.35)',
                      }}>
                        {line || ' '}
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </Glass>
        </div>
      </div>

      <div>
        <SectionLabel>Release History</SectionLabel>
        <Glass>
          <div data-testid="release-history" style={{ padding: '14px 16px 10px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '90px 90px 1fr 80px 80px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: 4 }}>
              {['Version','Status','Experiment','mAP','Targets'].map(h => <ColHeader key={h}>{h}</ColHeader>)}
            </div>
            {releasesData.length === 0 ? (
              <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', padding: '16px 4px', fontFamily: 'monospace' }}>No releases.</div>
            ) : releasesData.map(r => (
              <div
                key={r.version}
                data-testid={`release-${r.version}`}
                style={{ display: 'grid', gridTemplateColumns: '90px 90px 1fr 80px 80px', gap: 8, padding: '8px 4px', borderBottom: '1px solid rgba(255,255,255,0.04)', alignItems: 'center' }}
              >
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#dde2e0', fontFamily: 'monospace' }}>{r.version}</span>
                <StatusBadge status={r.status} />
                <span style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.35)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.experiment}</span>
                <span style={{ fontSize: '11px', color: '#8a9a6a', fontFamily: 'monospace', fontWeight: 600 }}>
                  mAP {(r.map_score * 100).toFixed(1)}%
                </span>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  {(r.targets ?? []).map(t => (
                    <span key={t} style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace', border: '1px solid rgba(255,255,255,0.08)', padding: '1px 5px' }}>
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Glass>
      </div>
    </div>
  )
}
