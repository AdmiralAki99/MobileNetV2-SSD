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

const StagePipeline = ({ stages, selected, onSelect }: { stages: Stage[]; selected: Stage | null; onSelect: (s: Stage | null) => void }) => (
  <div data-testid="stage-pipeline" style={{ display: 'flex', alignItems: 'flex-start', overflowX: 'auto', padding: '6px 0 2px' }}>
    {stages.map((stage, i) => {
      const isLast = i === stages.length - 1
      const isSel  = selected?.id === stage.id
      const color  = TASK_COLOR[stage.status] ?? '#494e4d'
      return (
        <div key={stage.id} style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
          <div
            data-testid={`stage-${stage.id}`}
            onClick={() => onSelect(isSel ? null : stage)}
            style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 82, cursor: 'pointer' }}
          >
            <div style={{
              width: 34, height: 34, borderRadius: '50%',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: stage.status === 'pending' ? 'transparent' : `${color}1e`,
              border: `2px solid ${stage.status === 'pending' ? 'rgba(255,255,255,0.10)' : color}`,
              boxShadow: isSel ? `0 0 0 3px ${color}33` : 'none',
            }}>
              {stage.status === 'success' && <span style={{ fontSize: 13, color }}>✓</span>}
              {stage.status === 'failed'  && <span style={{ fontSize: 13, color }}>✗</span>}
            </div>
            <span style={{ fontSize: '10px', color: isSel ? 'var(--accent)' : 'var(--text-secondary)', marginTop: 6, textAlign: 'center', whiteSpace: 'nowrap' }}>
              {stage.name}
            </span>
            <span style={{ fontSize: '9px', color: 'var(--text-tertiary)', fontFamily: 'monospace', marginTop: 2 }}>
              {stage.duration != null ? fmtDur(stage.duration) : ''}
            </span>
          </div>
          {!isLast && <div style={{ width: 18, height: 2, flexShrink: 0, marginBottom: 22, background: 'rgba(255,255,255,0.07)' }} />}
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
    <div data-testid="deploy-view" style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

      <div data-testid="current-run-card" style={{ padding: '14px 18px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 18 }}>
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: 4 }}>Current Run</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <span data-testid="run-branch" style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{run.branch}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>@{run.commit}</span>
              <StatusBadge status={run.status} />
            </div>
            <div data-testid="run-commit-message" style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{run.commit_message}</div>
          </div>
          <div style={{ display: 'flex', gap: 20, flexShrink: 0 }}>
            <div>
              <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>Trigger</div>
              <span data-testid="run-trigger" style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{run.trigger ?? '—'}</span>
            </div>
          </div>
        </div>
        <StagePipeline stages={run.stages ?? []} selected={selectedStage} onSelect={setSelectedStage} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', gap: 16 }}>
        <div data-testid="recent-runs" style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 12, textTransform: 'uppercase' }}>Recent Runs</div>
          {recentRuns.map(r => (
            <div key={r.id} data-testid={`recent-run-${r.id}`} style={{ padding: '8px 0', borderBottom: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                <span style={{ fontSize: '11.5px', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{r.branch}</span>
                <span style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>@{r.commit}</span>
                <div style={{ marginLeft: 'auto' }}><StatusBadge status={r.status} /></div>
              </div>
            </div>
          ))}
        </div>

        <div data-testid="stage-log" style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', minHeight: 160 }}>
          {!selectedStage
            ? <div style={{ color: 'var(--text-tertiary)', fontSize: '12px' }}>Click a stage above to view logs</div>
            : (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <span data-testid="log-stage-name" style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>{selectedStage.name}</span>
                  <StatusBadge status={selectedStage.status} />
                </div>
                <div data-testid="log-lines" style={{ fontFamily: 'monospace', fontSize: '11px', lineHeight: 1.75 }}>
                  {(STAGE_LOGS[selectedStage.id] ?? []).map((line, i) => (
                    <div key={i} style={{ color: line.startsWith('$') ? 'var(--accent)' : line.startsWith('✓') ? 'var(--success)' : 'var(--text-secondary)' }}>
                      {line || ' '}
                    </div>
                  ))}
                </div>
              </>
            )
          }
        </div>
      </div>

      <div data-testid="release-history" style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 14, textTransform: 'uppercase' }}>Release History</div>
        {releasesData.length === 0
          ? <div style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>No releases.</div>
          : releasesData.map(r => (
            <div key={r.version} data-testid={`release-${r.version}`} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 4px', borderBottom: '1px solid var(--border-subtle)' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{r.version}</span>
              <StatusBadge status={r.status} />
              <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>mAP {(r.map_score * 100).toFixed(1)}%</span>
              <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{r.experiment}</span>
            </div>
          ))
        }
      </div>
    </div>
  )
}
