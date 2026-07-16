import { StatTile } from './StatTile'
import { LineChart } from './LineChart'
import { LRChart } from './LRChart'
import { NMSHealthChart } from './NMSHealthChart'
import { ClassAPChart } from './ClassAPChart'
import { ConfusionMatrix } from './ConfusionMatrix'
import { DetectionCard } from './DetectionCard'
import {
  MOCK_TRAIN_LOSS, MOCK_VAL_LOSS, MOCK_MAP_CURVE,
  MOCK_CLASS_AP, MOCK_CONF, MOCK_IMAGES,
  MOCK_LR_CURVE, MOCK_NMS_MEAN_SCORE, MOCK_NMS_AVG_DET, MOCK_NMS_ZERO_DET,
} from './mockData'
import { type Experiment } from '../../types/experiment'

interface MetricsData {
  train_loss?: number[]
  val_loss?: number[]
  map_curve?: number[]
  class_ap?: Record<string, number>
  conf_mat?: number[][]
  images?: { id: number | string; label: string; boxes: { x: number; y: number; w: number; h: number; cls: string; score: number }[] }[]
  lr_curve?: number[]
  nms_mean_score?: number[]
  nms_avg_det?: number[]
  nms_zero_det?: number[]
}

interface Props {
  metricsData?: MetricsData
  selectedExp?: Experiment | null
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

export const MetricsView = ({ metricsData, selectedExp }: Props) => {
  const d = metricsData ?? {}
  const trainLoss = d.train_loss ?? MOCK_TRAIN_LOSS
  const valLoss   = d.val_loss   ?? MOCK_VAL_LOSS
  const mapCurve  = d.map_curve  ?? MOCK_MAP_CURVE
  const classAP    = d.class_ap        ?? MOCK_CLASS_AP
  const confMat    = d.conf_mat        ?? MOCK_CONF
  const images     = d.images          ?? MOCK_IMAGES
  const lrCurve    = d.lr_curve        ?? MOCK_LR_CURVE
  const nmsScore   = d.nms_mean_score  ?? MOCK_NMS_MEAN_SCORE
  const nmsAvgDet  = d.nms_avg_det     ?? MOCK_NMS_AVG_DET
  const nmsZeroDet = d.nms_zero_det    ?? MOCK_NMS_ZERO_DET

  const bestMAP    = Math.max(...mapCurve)
  const bestEpoch  = mapCurve.indexOf(bestMAP) + 1
  const finalTrain = trainLoss[trainLoss.length - 1]
  const finalVal   = valLoss[valLoss.length - 1]
  const mAP        = Object.values(classAP).reduce((s, v) => s + v, 0) / Object.values(classAP).length

  return (
    <div
      data-testid="metrics-view"
      style={{
        height: '100%', overflowY: 'auto',
        padding: '22px 24px 40px',
        display: 'flex', flexDirection: 'column', gap: 20,
        background: `
          radial-gradient(ellipse 70% 35% at 55% -5%,  rgba(0,200,140,0.07) 0%, transparent 70%),
          radial-gradient(ellipse 40% 25% at 5%  95%,  rgba(180,100,40,0.05) 0%, transparent 60%),
          var(--bg-primary)
        `,
      }}
    >
      <div
        data-testid="stat-strip"
        style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 1 }}
      >
        <StatTile label="Best mAP@0.5"    value={`${(bestMAP * 100).toFixed(1)}%`}    sub={`epoch ${bestEpoch}`}                              accentColor="#8a9a6a" sparkData={mapCurve}  />
        <StatTile label="Mean AP"          value={`${(mAP * 100).toFixed(1)}%`}         sub="20 VOC classes"                                    accentColor="#6a7a8a" sparkData={mapCurve}  />
        <StatTile label="Final Train Loss" value={finalTrain.toFixed(3)}                sub="epoch 200"                                         accentColor="#7a8a6a" sparkData={trainLoss} />
        <StatTile label="Final Val Loss"   value={finalVal.toFixed(3)}                  sub="epoch 200"                                         accentColor="#8a6a6a" sparkData={valLoss}   />
        <StatTile label="Total Steps"
          value={selectedExp?.total_steps != null ? selectedExp.total_steps.toLocaleString() : '—'}
          sub={selectedExp?.experiment_id ?? 'no experiment'}
          accentColor="#6a7a8a"
        />
      </div>

      <div>
        <SectionLabel>Training Curves</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <Glass>
            <div style={{ padding: '18px 20px 14px' }}>
              <LineChart title="Loss Curves" sub="train & validation · lower is better"
                series={[{ label: 'Train', data: trainLoss }, { label: 'Val', data: valLoss }]} />
            </div>
          </Glass>
          <Glass>
            <div style={{ padding: '18px 20px 14px' }}>
              <LineChart title="mAP@0.5" sub="VOC2012 validation · higher is better"
                series={[{ label: 'mAP', data: mapCurve }]} />
            </div>
          </Glass>
        </div>
      </div>

      <div>
        <SectionLabel>Optimizer &amp; Detection Health</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <Glass>
            <div style={{ padding: '18px 20px 14px' }}>
              <LRChart data={lrCurve} />
            </div>
          </Glass>
          <Glass>
            <div style={{ padding: '18px 20px 14px' }}>
              <NMSHealthChart meanScore={nmsScore} avgDetections={nmsAvgDet} zeroDetRatio={nmsZeroDet} />
            </div>
          </Glass>
        </div>
      </div>

      <div>
        <SectionLabel>Evaluation</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', gap: 12, alignItems: 'start' }}>
          <Glass>
            <ClassAPChart data={classAP} />
          </Glass>
          <Glass>
            <ConfusionMatrix matrix={confMat} />
          </Glass>
        </div>
      </div>

      <div>
        <SectionLabel>Detection Samples</SectionLabel>
        <Glass>
          <div
            data-testid="detection-samples"
            style={{ padding: '16px 18px 14px' }}
          >
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              marginBottom: 12,
            }}>
              <span style={{ fontSize: '10px', fontWeight: 600, color: 'rgba(255,255,255,0.5)' }}>
                epoch 200 · val set
              </span>
              <span style={{ fontSize: '9px', color: 'rgba(255,255,255,0.18)', fontFamily: 'monospace' }}>
                hover boxes for score
              </span>
            </div>
            <div style={{ display: 'flex', gap: 10, overflowX: 'auto', paddingBottom: 4 }}>
              {images.map(img => <DetectionCard key={img.id} img={img} />)}
            </div>
          </div>
        </Glass>
      </div>
    </div>
  )
}
