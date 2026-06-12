import { useState } from 'react'
import { Header, type ViewMode } from './components/Header'

interface Props {
  initialView?: ViewMode
}

export const App = ({ initialView = 'pipeline' }: Props) => {
  const [viewMode, setViewMode]       = useState<ViewMode>(initialView)
  const [statusFilter, setStatusFilter] = useState('all')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--bg-primary)' }}>
      <Header
        viewMode={viewMode}
        setViewMode={setViewMode}
        statusFilter={statusFilter}
        setStatusFilter={setStatusFilter}
      />
      <main style={{ flex: 1, overflow: 'hidden' }} data-testid={`view-${viewMode}`}>
        {viewMode === 'pipeline' && <div>Pipeline view</div>}
        {viewMode === 'metrics'  && <div>Metrics view</div>}
        {viewMode === 'etl'      && <div>ETL view</div>}
        {viewMode === 'ops'      && <div>Ops view</div>}
        {viewMode === 'deploy'   && <div>Deploy view</div>}
        {viewMode === 'config'   && <div>Config view</div>}
      </main>
    </div>
  )
}
