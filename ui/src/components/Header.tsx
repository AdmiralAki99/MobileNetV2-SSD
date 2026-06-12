import { TabGroup } from './TabGroup'
import { PillButton } from './PillButton'
import { SearchBar } from './SearchBar'
import { GearSvg } from './icons'

export type ViewMode = 'pipeline' | 'metrics' | 'etl' | 'ops' | 'deploy' | 'config'

const TAB_TO_VIEW: Record<string, ViewMode> = {
  Pipeline: 'pipeline', Metrics: 'metrics', ETL: 'etl',
  Ops: 'ops', Deploy: 'deploy', Config: 'config',
}

const VIEW_TO_TAB: Record<ViewMode, string> = {
  pipeline: 'Pipeline', metrics: 'Metrics', etl: 'ETL',
  ops: 'Ops', deploy: 'Deploy', config: 'Config',
}

const STATUS_FILTERS = ['all', 'pending', 'running', 'success', 'failed']

interface Props {
  viewMode: ViewMode
  setViewMode: (v: ViewMode) => void
  statusFilter: string
  setStatusFilter: (s: string) => void
}

export const Header = ({ viewMode, setViewMode, statusFilter, setStatusFilter }: Props) => (
  <div style={{
    display: 'flex', alignItems: 'center', gap: 16, padding: '0 20px',
    height: 52, borderBottom: '1px solid var(--border-subtle)',
    background: 'var(--bg-secondary)', flexShrink: 0,
  }}>
    <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.3px', marginRight: 8, whiteSpace: 'nowrap' }}>
      sentinel<span style={{ color: 'var(--accent)' }}>{'>'}</span>
    </div>
    <TabGroup
      tabs={['Pipeline', 'Metrics', 'ETL', 'Ops', 'Deploy', 'Config']}
      active={VIEW_TO_TAB[viewMode]}
      onChange={t => setViewMode(TAB_TO_VIEW[t])}
    />
    {viewMode === 'pipeline' && (
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1, overflow: 'hidden', minWidth: 0 }}>
        {STATUS_FILTERS.map(s => (
          <PillButton key={s} active={statusFilter === s} onClick={() => setStatusFilter(s)}>{s}</PillButton>
        ))}
      </div>
    )}
    {viewMode !== 'pipeline' && <div style={{ flex: 1 }} />}
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
      <SearchBar />
      <button style={{
        width: 34, height: 34, borderRadius: '50%', border: '1px solid var(--border-subtle)',
        background: 'var(--bg-pill)', color: 'var(--text-secondary)',
        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <GearSvg size={15} />
      </button>
    </div>
  </div>
)
