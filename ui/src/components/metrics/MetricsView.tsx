import { StatTile } from './StatTile'
import { LineChart } from './LineChart'
import { ClassAPChart } from './ClassAPChart'
import { ConfusionMatrix } from './ConfusionMatrix'
import { DetectionCard } from './DetectionCard'
import {
  MOCK_TRAIN_LOSS, MOCK_VAL_LOSS, MOCK_MAP_CURVE,
  MOCK_CLASS_AP, MOCK_CONF, MOCK_IMAGES,
} from './mockData'
import { type Experiment } from '../../types/experiment'

interface MetricsData {
  train_loss?: number[]
  val_loss?: number[]
  map_curve?: number[]
  class_ap?: Record<string, number>
  conf_mat?: number[][]
  images?: { id: number | string; label: string; boxes: { x: number; y: number; w: number; h: number; cls: string; score: number }[] }[]
}

interface Props {
  metricsData?: MetricsData
  selectedExp?: Experiment | null
}

export const MetricsView = ({ metricsData, selectedExp }: Props) => {
  const d = metricsData ?? {}
  const trainLoss = d.train_loss ?? MOCK_TRAIN_LOSS
  const valLoss   = d.val_loss   ?? MOCK_VAL_LOSS
  const mapCurve  = d.map_curve  ?? MOCK_MAP_CURVE
  const classAP   = d.class_ap   ?? MOCK_CLASS_AP
  const confMat  = d.conf_mat ?? MOCK_CONF
  const images   = d.images   ?? MOCK_IMAGES


  const bestMAP    = Math.max(...mapCurve)
  const bestEpoch  = mapCurve.indexOf(bestMAP) + 1
  const finalTrain = trainLoss[trainLoss.length - 1]
  const finalVal   = valLoss[valLoss.length - 1]
  const mAP        = Object.values(classAP).reduce((s, v) => s + v, 0) / Object.values(classAP).length

  return (
    <div
      data-testid="metrics-view"
      style={{ height: '100%', overflowY: 'auto', padding: '18px 20px 32px', display: 'flex', flexDirection: 'column', gap: 14 }}
    >
      <div data-testid="stat-strip" style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 10 }}>
        <StatTile label="Best mAP@0.5"    value={`${(bestMAP * 100).toFixed(1)}%`}    sub={`epoch ${bestEpoch}`}                accentColor="var(--accent)"           sparkData={mapCurve} />
        <StatTile label="Mean AP"          value={`${(mAP * 100).toFixed(1)}%`}         sub="20 VOC classes"                      accentColor="#7c9ef5"                  sparkData={mapCurve} />
        <StatTile label="Final Train Loss" value={finalTrain.toFixed(3)}                sub="epoch 200"                           accentColor="var(--text-secondary)"    sparkData={trainLoss} />
        <StatTile label="Final Val Loss"   value={finalVal.toFixed(3)}                  sub="epoch 200"                           accentColor="#e88548"                  sparkData={valLoss} />
        <StatTile label="Total Steps"
          value={selectedExp?.total_steps != null ? selectedExp.total_steps.toLocaleString() : '—'}
          sub={selectedExp?.experiment_id ?? 'no experiment'}
          accentColor="var(--text-tertiary)"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
        <LineChart title="Loss Curves" sub="train & validation · lower is better"
          series={[{ label: 'Train', data: trainLoss }, { label: 'Val', data: valLoss }]} />
        <LineChart title="mAP@0.5" sub="VOC2012 validation · higher is better"
          series={[{ label: 'mAP', data: mapCurve }]} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 14, alignItems: 'start' }}>
        <ClassAPChart data={classAP} />
        <ConfusionMatrix matrix={confMat} />
      </div>

      <div data-testid="detection-samples" style={{ padding: '20px 22px', borderRadius: 14, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
          <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>Detection Samples</span>
          <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>epoch 200 · val set · hover boxes for score</span>
        </div>
        <div style={{ display: 'flex', gap: 12, overflowX: 'auto', paddingBottom: 4 }}>
          {images.map(img => <DetectionCard key={img.id} img={img} />)}
        </div>
      </div>

    </div>
  )
}
