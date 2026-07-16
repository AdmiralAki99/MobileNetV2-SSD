import { useState } from 'react'
import { Header, type ViewMode } from './components/Header'
import { PipelineView } from './components/pipeline/PipelineView'
import { MetricsView } from './components/metrics/MetricsView'
import { EtlView } from './components/etl/EtlView'
import { OpsView } from './components/ops/OpsView'
import { DeployView } from './components/deploy/DeployView'
import { ConfigView } from './components/config/ConfigView'
import { AnchorView } from './components/anchors/AnchorView'
import { DatasetView } from './components/dataset/DatasetView'
import { usePolling } from './api/hooks'
import { fetchEtlStats, fetchEtlVideos } from './api/client'

interface Props {
  initialView?: ViewMode
}

export const App = ({ initialView = 'pipeline' }: Props) => {
  const [viewMode, setViewMode]         = useState<ViewMode>(initialView)
  const [statusFilter, setStatusFilter] = useState('all')

  const { data: etlStats  } = usePolling(fetchEtlStats,  30000, viewMode === 'etl')
  const { data: etlVideos } = usePolling(fetchEtlVideos, 30000, viewMode === 'etl')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--bg-primary)' }}>
      <Header
        viewMode={viewMode}
        setViewMode={setViewMode}
        statusFilter={statusFilter}
        setStatusFilter={setStatusFilter}
      />
      <main style={{ flex: 1, overflow: 'hidden' }} data-testid={`view-${viewMode}`}>
        {viewMode === 'pipeline' && <PipelineView statusFilter={statusFilter} />}
        {viewMode === 'metrics'  && <MetricsView />}
        {viewMode === 'etl'      && <EtlView statsData={etlStats ?? undefined} videosData={etlVideos ?? undefined} />}
        {viewMode === 'ops'      && <OpsView />}
        {viewMode === 'deploy'   && <DeployView />}
        {viewMode === 'config'   && <ConfigView />}
        {viewMode === 'anchors'  && <AnchorView />}
        {viewMode === 'dataset'  && <DatasetView />}
      </main>
    </div>
  )
}
