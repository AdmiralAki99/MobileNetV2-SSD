import { useState, useEffect } from 'react'
import { ActionBtn } from './ActionBtn'
import { ArtifactRow } from './ArtifactRow'
import { ArcGauge } from './ArcGauge'
import { PillButton } from '../PillButton'
import { usePolling } from '../../api/hooks'
import {
  fetchInstanceStatus, fetchArtifacts, fetchJobStatus,
  launchTraining, stopTraining, exportSavedModel, exportOnnx,
} from '../../api/client'
import { type Experiment } from '../../types/experiment'

interface InstanceInfo {
  tensorboard_url?: string
  instance_type?: string
  public_ip?: string
}

interface ArtifactStatus {
  saved_model: boolean
  onnx: boolean
  priors: boolean
}

interface Props {
  selectedExp: Experiment | null
  onRefresh: () => void
}

export const DetailPanel = ({ selectedExp, onRefresh }: Props) => {
  const [instanceInfo, setInstanceInfo] = useState<InstanceInfo | null>(null)
  const [launching,    setLaunching]    = useState(false)
  const [stopping,     setStopping]     = useState(false)
  const [jobId,        setJobId]        = useState<string | null>(null)
  const [jobStatus,    setJobStatus]    = useState<string | null>(null)
  const [artifacts,    setArtifacts]    = useState<ArtifactStatus | null>(null)

  useEffect(() => {
    if (!selectedExp?.ec2_instance) { setInstanceInfo(null); return }
    fetchInstanceStatus(selectedExp.ec2_instance)
      .then((res: any) => setInstanceInfo(res?.message || null))
      .catch(() => setInstanceInfo(null))
  }, [selectedExp?.ec2_instance])

  useEffect(() => {
    if (!selectedExp) { setArtifacts(null); return }
    fetchArtifacts(selectedExp.experiment_id)
      .then((res: any) => setArtifacts(res?.artifat_status || null))
      .catch(() => setArtifacts(null))
  }, [selectedExp?.experiment_id, selectedExp?.fingerprint])

  const { data: jobData } = usePolling(
    () => fetchJobStatus(jobId!),
    3000,
    !!jobId && jobStatus === 'running',
  )
  useEffect(() => {
    if ((jobData as any)?.job_status?.status) setJobStatus((jobData as any).job_status.status)
  }, [jobData])

  const handleLaunch = async () => {
    if (!selectedExp) return
    setLaunching(true)
    try {
      await launchTraining({
        experiment_id:   selectedExp.experiment_id,
        fingerprint:     selectedExp.fingerprint,
        config_filename: selectedExp.config_filename ?? `${selectedExp.experiment_id}.yaml`,
      })
      onRefresh()
    } catch (e) { console.error('Launch failed', e) }
    finally { setLaunching(false) }
  }

  const handleStop = async () => {
    if (!selectedExp) return
    setStopping(true)
    try {
      await stopTraining({
        instance_id:   selectedExp.ec2_instance,
        experiment_id: selectedExp.experiment_id,
        fingerprint:   selectedExp.fingerprint,
      })
      onRefresh()
    } catch (e) { console.error('Stop failed', e) }
    finally { setStopping(false) }
  }

  const handleExportModel = async () => {
    if (!selectedExp) return
    try {
      const res: any = await exportSavedModel({
        experiment_id:      selectedExp.experiment_id,
        fingerprint:        selectedExp.fingerprint,
        checkpoint_s3_path: selectedExp.checkpoint_s3_path,
        config_filename:    selectedExp.config_filename ?? `${selectedExp.experiment_id}.yaml`,
      })
      setJobId(res.job_id); setJobStatus('running')
    } catch (e) { console.error('Export SavedModel failed', e) }
  }

  const handleExportOnnx = async () => {
    if (!selectedExp) return
    try {
      const res: any = await exportOnnx({ experiment_id: selectedExp.experiment_id })
      setJobId(res.job_id); setJobStatus('running')
    } catch (e) { console.error('Export ONNX failed', e) }
  }

  if (!selectedExp) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
        <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Select an experiment</span>
      </div>
    )
  }

  const canLaunch      = selectedExp.status === 'pending' || selectedExp.status === 'failed'
  const canStop        = selectedExp.status === 'running' && !!selectedExp.ec2_instance
  const canExportModel = !!selectedExp.checkpoint_s3_path
  const canExportOnnx  = !!artifacts?.saved_model
  const mapPct         = selectedExp.best_metric != null ? Math.round(selectedExp.best_metric * 100) : null

  return (
    <div data-testid="detail-panel" style={{ display: 'flex', flexDirection: 'column', gap: 12, padding: '18px 14px 14px 10px', overflowY: 'auto' }}>
      <div>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>
          {selectedExp.experiment_id}
        </div>
      </div>

      <div style={{
        display: 'flex', flexDirection: 'column', gap: 1,
        padding: '2px 0 6px',
      }}>
        <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)', marginBottom: 10 }} />

        {mapPct !== null && (
          <div style={{ padding: '6px 8px 10px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <ArcGauge
              testId="map-score"
              value={mapPct / 100}
              label="mAP"
              color="#00d4a0"
              size={62}
            />
          </div>
        )}

        <div style={{ padding: '10px 8px 6px' }}>
          <ArcGauge
            value={selectedExp.best_epoch != null ? Math.min(selectedExp.best_epoch / 200, 1) : 0}
            label="Training Progress"
            sublabel={selectedExp.best_epoch != null ? `ep ${selectedExp.best_epoch}` : undefined}
            color="#e8924a"
            size={62}
          />
        </div>

        <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)', marginTop: 6 }} />
      </div>

      {selectedExp.failure_reason && (
        <div data-testid="failure-reason" style={{
          padding: '8px 12px', borderRadius: 10, wordBreak: 'break-word',
          background: 'rgba(232,72,85,0.08)', border: '1px solid rgba(232,72,85,0.25)',
          fontSize: '11px', color: 'var(--danger)',
        }}>
          {selectedExp.failure_reason}
        </div>
      )}

      {instanceInfo?.tensorboard_url && (
        <PillButton onClick={() => window.open(instanceInfo!.tensorboard_url, '_blank')}
          style={{ width: '100%', justifyContent: 'center' }}>
          TensorBoard →
        </PillButton>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <ActionBtn label="Launch" color="#65c16a" disabled={!canLaunch} busy={launching} onClick={handleLaunch} />
        <ActionBtn label="Stop"   color="#e84855" disabled={!canStop}   busy={stopping}  onClick={handleStop} />
      </div>

      {artifacts && (
        <div data-testid="artifacts">
          <ArtifactRow label="SavedModel" ok={artifacts.saved_model} />
          <ArtifactRow label="ONNX"       ok={artifacts.onnx} />
          <ArtifactRow label="Priors"     ok={artifacts.priors} />
        </div>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <ActionBtn label="Export SavedModel" color="#00d4a0" disabled={!canExportModel} onClick={handleExportModel} />
        <ActionBtn label="Export ONNX"       color="#00d4a0" disabled={!canExportOnnx}  onClick={handleExportOnnx} />
      </div>

      {jobId && (
        <div data-testid="job-status" style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>
          Export job: {jobStatus}
        </div>
      )}
    </div>
  )
}
