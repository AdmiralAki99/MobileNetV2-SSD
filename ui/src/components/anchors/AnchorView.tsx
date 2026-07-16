import { useEffect, useRef, useState } from 'react'
import { PillButton } from '../PillButton'
import { fetchDatasets, fetchPriors, fetchBoxDims, deriveCluster, exportCluster } from '../../api/client'
import type { DatasetEntry, ClusterResult, BoxDims } from './anchorTypes'

// ── palette ────────────────────────────────────────────────────────────────
const PALETTE = ['#4e79a7','#f28e2b','#59a14f','#b07aa1','#e15759','#76b7b2','#edc948','#ff9da7']

// ── KMeans helpers ──────────────────────────────────────────────────────────
function dist2([ax, ay]: [number,number], [bx, by]: [number,number]) {
  return (ax-bx)**2 + (ay-by)**2
}

function assignPoints(pts: [number,number][], centroids: [number,number][]): number[] {
  return pts.map(p => {
    let best = 0, bestD = Infinity
    centroids.forEach((c, i) => { const d = dist2(p, c); if (d < bestD) { bestD = d; best = i } })
    return best
  })
}

function updateCentroids(pts: [number,number][], assignments: number[], k: number, prev: [number,number][]): [number,number][] {
  return Array.from({ length: k }, (_, ci) => {
    const cluster = pts.filter((_, i) => assignments[i] === ci)
    if (cluster.length === 0) return prev[ci]
    return [
      cluster.reduce((s, p) => s + p[0], 0) / cluster.length,
      cluster.reduce((s, p) => s + p[1], 0) / cluster.length,
    ] as [number,number]
  })
}

function kmeans2D(pts: [number,number][], k: number, iters = 25): { centroids: [number,number][]; assignments: number[] } {
  if (pts.length === 0) return { centroids: [], assignments: [] }
  // k-means++ seeding
  const used = new Set<number>()
  const centroids: [number,number][] = []
  const first = Math.floor(Math.random() * pts.length)
  centroids.push(pts[first]); used.add(first)
  for (let ci = 1; ci < k; ci++) {
    const dists = pts.map((p, i) => used.has(i) ? 0 : Math.min(...centroids.map(c => dist2(p, c))))
    const total = dists.reduce((s, d) => s + d, 0)
    let r = Math.random() * total, idx = 0
    for (let i = 0; i < dists.length; i++) { r -= dists[i]; if (r <= 0) { idx = i; break } }
    centroids.push(pts[idx]); used.add(idx)
  }
  let assignments = assignPoints(pts, centroids)
  let current = centroids
  for (let iter = 0; iter < iters; iter++) {
    const next = updateCentroids(pts, assignments, k, current)
    assignments = assignPoints(pts, next)
    current = next
  }
  return { centroids: current, assignments }
}

function clusterEllipse(pts: [number,number][], assignments: number[], ci: number, cx: number, cy: number) {
  const cluster = pts.filter((_, i) => assignments[i] === ci)
  if (cluster.length < 2) return { rx: 0.01, ry: 0.01 }
  const varX = cluster.reduce((s, p) => s + (p[0] - cx) ** 2, 0) / cluster.length
  const varY = cluster.reduce((s, p) => s + (p[1] - cy) ** 2, 0) / cluster.length
  return { rx: Math.max(Math.sqrt(varX) * 2, 0.004), ry: Math.max(Math.sqrt(varY) * 2, 0.004) }
}

// ── interactive scatter ─────────────────────────────────────────────────────
const SW = 460, SH = 360
const SP = { top: 16, right: 24, bottom: 40, left: 48 }

type ScaleType = 'linear' | 'sqrt' | 'log'
const CLIP_OPTIONS = [0.80, 0.90, 0.95, 0.99, 1.0] as const
const CLIP_LABELS: Record<number, string> = { 0.80: '80%', 0.90: '90%', 0.95: '95%', 0.99: '99%', 1.0: 'All' }

interface ScatterProps {
  dims: BoxDims | null
  k: number
  onCentroidsChange: (centroids: [number,number][]) => void
}

function InteractiveScatter({ dims, k, onCentroidsChange }: ScatterProps) {
  const [centroids, setCentroids] = useState<[number,number][]>([])
  const [assignments, setAssignments] = useState<number[]>([])
  const [scaleType, setScaleType] = useState<ScaleType>('sqrt')
  const [clip, setClip] = useState<number>(0.95)
  const dragging = useRef<number | null>(null)
  const svgRef = useRef<SVGSVGElement>(null)

  // re-cluster when points or k change
  useEffect(() => {
    if (!dims || dims.norm.length === 0) { setCentroids([]); setAssignments([]); return }
    const result = kmeans2D(dims.norm, k)
    setCentroids(result.centroids)
    setAssignments(result.assignments)
    onCentroidsChange(result.centroids)
  }, [dims, k]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!dims || dims.norm.length === 0) {
    return (
      <div style={{
        width: '100%', aspectRatio: `${SW}/${SH}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: '1px dashed var(--border-subtle)', borderRadius: 8,
        color: 'var(--text-tertiary)', fontSize: 12,
      }}>
        Select a dataset to load box distribution
      </div>
    )
  }

  const points = dims.norm

  const pct = (vals: number[], p: number) => {
    const s = [...vals].sort((a, b) => a - b)
    return s[Math.min(Math.floor(s.length * p), s.length - 1)]
  }
  const xMax = Math.max(pct(points.map(p => p[0]), clip) * 1.05, 0.01)
  const yMax = Math.max(pct(points.map(p => p[1]), clip) * 1.05, 0.01)

  const PW = SW - SP.left - SP.right
  const PH = SH - SP.top - SP.bottom
  const EPS = 1e-6

  const fwd = (v: number, max: number): number => {
    const t = Math.max(v, 0) / max
    if (scaleType === 'sqrt') return Math.sqrt(t)
    if (scaleType === 'log')  return Math.log1p(t * Math.E) / Math.log1p(Math.E)
    return t
  }
  const inv = (t: number, max: number): number => {
    if (scaleType === 'sqrt') return t * t * max
    if (scaleType === 'log')  return (Math.expm1(t * Math.log1p(Math.E)) / Math.E) * max
    return t * max
  }

  const xSc = (v: number) => SP.left + fwd(v, xMax) * PW
  const ySc = (v: number) => SH - SP.bottom - fwd(v, yMax) * PH
  const fromSvgX = (px: number) => Math.max(EPS, inv(Math.max(0, (px - SP.left) / PW), xMax))
  const fromSvgY = (py: number) => Math.max(EPS, inv(Math.max(0, (SH - SP.bottom - py) / PH), yMax))

  const makeTicks = (max: number) =>
    [0, 0.25, 0.5, 0.75, 1.0].map(f => parseFloat((f * max).toFixed(3)))
  const xTicks = makeTicks(xMax)
  const yTicks = makeTicks(yMax)

  // downsample for render performance
  const sampled = points.length > 1200
    ? points.filter((_, i) => i % Math.ceil(points.length / 1200) === 0)
    : points
  const sampledIdx = points.length > 1200
    ? points.map((_, i) => i).filter(i => i % Math.ceil(points.length / 1200) === 0)
    : points.map((_, i) => i)

  // ── drag handlers ──────────────────────────────────────────────────────
  const svgToData = (clientX: number, clientY: number): [number, number] => {
    const svg = svgRef.current!
    const rect = svg.getBoundingClientRect()
    const svgX = (clientX - rect.left) * (SW / rect.width)
    const svgY = (clientY - rect.top) * (SH / rect.height)
    return [
      Math.min(xMax, fromSvgX(svgX)),
      Math.min(yMax, fromSvgY(svgY)),
    ]
  }

  const onPointerDown = (e: React.PointerEvent, ci: number) => {
    e.stopPropagation()
    ;(e.currentTarget as SVGElement).setPointerCapture(e.pointerId)
    dragging.current = ci
  }

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragging.current === null) return
    const ci = dragging.current
    const [dx, dy] = svgToData(e.clientX, e.clientY)
    setCentroids(prev => {
      const next = prev.map((c, i) => i === ci ? [dx, dy] as [number,number] : c)
      const newAssign = assignPoints(points, next)
      setAssignments(newAssign)
      onCentroidsChange(next)
      return next
    })
  }

  const onPointerUp = () => { dragging.current = null }

  const pillStyle = (active: boolean) => ({
    padding: '3px 9px', borderRadius: 4, border: '1px solid var(--border-subtle)',
    fontSize: 10, cursor: 'pointer', fontFamily: 'monospace',
    background: active ? 'var(--accent, #5a9a6a)' : 'var(--bg-pill)',
    color: active ? '#fff' : 'var(--text-tertiary)',
  } as const)

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 6 }}>
        <div style={{ display: 'flex', gap: 3 }}>
          {(['linear', 'sqrt', 'log'] as ScaleType[]).map(s => (
            <button key={s} style={pillStyle(scaleType === s)} onClick={() => setScaleType(s)}>{s}</button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 3 }}>
          {CLIP_OPTIONS.map(c => (
            <button key={c} style={pillStyle(clip === c)} onClick={() => setClip(c)}>{CLIP_LABELS[c]}</button>
          ))}
        </div>
      </div>
    <svg
      ref={svgRef}
      viewBox={`0 0 ${SW} ${SH}`}
      width="100%"
      style={{ display: 'block', cursor: dragging.current !== null ? 'grabbing' : 'default' }}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerLeave={onPointerUp}
    >
      {/* grid */}
      {xTicks.map((t: number, i: number) => (
        <line key={i} x1={xSc(t)} x2={xSc(t)} y1={SP.top} y2={SH - SP.bottom}
          stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
      ))}
      {yTicks.map((t: number, i: number) => (
        <line key={i} x1={SP.left} x2={SW - SP.right} y1={ySc(t)} y2={ySc(t)}
          stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
      ))}

      {/* axes labels */}
      {xTicks.map((t: number, i: number) => (
        <text key={i} x={xSc(t)} y={SH - SP.bottom + 13} textAnchor="middle"
          fontSize={8} fill="rgba(255,255,255,0.25)">{t.toFixed(2)}</text>
      ))}
      {yTicks.map((t: number, i: number) => (
        <text key={i} x={SP.left - 4} y={ySc(t) + 3} textAnchor="end"
          fontSize={8} fill="rgba(255,255,255,0.25)">{t.toFixed(2)}</text>
      ))}
      <text x={(SP.left + SW - SP.right) / 2} y={SH - 6} textAnchor="middle"
        fontSize={8.5} fill="rgba(255,255,255,0.3)">norm width</text>
      <text transform="rotate(-90)" x={-(SP.top + SH - SP.bottom) / 2} y={12}
        textAnchor="middle" fontSize={8.5} fill="rgba(255,255,255,0.3)">norm height</text>

      {/* axes */}
      <line x1={SP.left} x2={SW - SP.right} y1={SH - SP.bottom} y2={SH - SP.bottom}
        stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
      <line x1={SP.left} x2={SP.left} y1={SP.top} y2={SH - SP.bottom}
        stroke="rgba(255,255,255,0.08)" strokeWidth={1} />

      {/* points (colored by cluster) */}
      {sampled.map(([w, h], si) => {
        const ci = assignments[sampledIdx[si]] ?? 0
        return (
          <circle key={si} cx={xSc(w)} cy={ySc(h)} r={1.8}
            fill={PALETTE[ci % PALETTE.length]} opacity={0.35} />
        )
      })}

      {/* cluster ellipses + centroids */}
      {centroids.map(([cx, cy], ci) => {
        const { rx, ry } = clusterEllipse(points, assignments, ci, cx, cy)
        const color = PALETTE[ci % PALETTE.length]
        const sx = xSc(cx), sy = ySc(cy)
        const erx = xSc(cx + rx) - xSc(cx)
        const ery = ySc(cy) - ySc(cy + ry)
        const S = 6 // cross arm half-length
        return (
          <g key={ci}>
            {/* ellipse */}
            <ellipse cx={sx} cy={sy} rx={Math.max(erx, 4)} ry={Math.max(ery, 4)}
              fill="none" stroke={color} strokeWidth={1.5} opacity={0.6}
              strokeDasharray="6 3" />
            {/* draggable centroid cross */}
            <g
              style={{ cursor: 'grab' }}
              onPointerDown={e => onPointerDown(e, ci)}
            >
              <circle cx={sx} cy={sy} r={10} fill="transparent" />
              <line x1={sx - S} y1={sy - S} x2={sx + S} y2={sy + S}
                stroke="#e07070" strokeWidth={2.5} strokeLinecap="round" />
              <line x1={sx + S} y1={sy - S} x2={sx - S} y2={sy + S}
                stroke="#e07070" strokeWidth={2.5} strokeLinecap="round" />
            </g>
          </g>
        )
      })}
    </svg>
    </div>
  )
}

// ── main view ───────────────────────────────────────────────────────────────
const selectStyle = {
  background: 'var(--bg-pill, #1a1f1a)',
  border: '1px solid var(--border-subtle)',
  borderRadius: 6, color: 'var(--text-primary)', fontSize: 12,
  padding: '5px 10px', cursor: 'pointer', outline: 'none', height: 30,
} as const

const K_OPTIONS = [2, 3, 4, 5, 6, 7, 8]

export const AnchorView = () => {
  const [datasets, setDatasets]     = useState<DatasetEntry[]>([])
  const [priors, setPriors]         = useState<string[]>([])
  const [dataset, setDataset]       = useState('')
  const [split, setSplit]           = useState('')
  const [priorsFile, setPriorsFile] = useState('')
  const [k, setK]                   = useState(3)
  const [dims, setDims]             = useState<BoxDims | null>(null)
  const [liveCentroids, setLiveCentroids] = useState<[number,number][]>([])
  const [result, setResult]         = useState<ClusterResult | null>(null)
  const [running, setRunning]       = useState(false)
  const [exporting, setExporting]   = useState(false)
  const [error, setError]           = useState<string | null>(null)

  useEffect(() => {
    fetchDatasets().then((d: DatasetEntry[]) => {
      setDatasets(d)
      if (d.length > 0) { setDataset(d[0].name); setSplit(d[0].split) }
    }).catch(() => {})
    fetchPriors().then((p: string[]) => {
      setPriors(p)
      if (p.length > 0) setPriorsFile(p[0])
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!dataset || !split) return
    fetchBoxDims(dataset, split).then(d => setDims(d)).catch(() => setDims(null))
  }, [dataset, split])

  const uniqueDatasets = [...new Set(datasets.map(d => d.name))]
  const splitsForDataset = datasets.filter(d => d.name === dataset).map(d => d.split)

  // derive aspect ratios from live 2D centroids
  const derivedARs = liveCentroids
    .map(([w, h]) => h > 0 ? parseFloat((w / h).toFixed(3)) : 1)
    .sort((a, b) => a - b)

  const handleEvaluate = async () => {
    if (!dataset || !split || !priorsFile) return
    setRunning(true); setError(null); setResult(null)
    try {
      const res = await deriveCluster({ dataset, split, algorithm: 'kmeans', num_aspect_ratios: k, priors: priorsFile })
      setResult(res.result)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Evaluation failed')
    } finally {
      setRunning(false)
    }
  }

  const handleExport = async () => {
    if (!priorsFile) return
    setExporting(true)
    try {
      await exportCluster({ dataset, split, algorithm: 'kmeans', num_aspect_ratios: k, priors: priorsFile, out: priorsFile })
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setExporting(false)
    }
  }

  return (
    <div style={{ height: '100%', overflow: 'auto', padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.2px' }}>Anchor Tuning</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
          Drag cluster centroids to refine aspect ratios — clusters update live
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <select value={dataset} onChange={e => { setDataset(e.target.value); setResult(null) }} style={selectStyle}>
          {uniqueDatasets.length === 0
            ? <option value="">No datasets</option>
            : uniqueDatasets.map(n => <option key={n} value={n}>{n}</option>)
          }
        </select>

        <select value={split} onChange={e => { setSplit(e.target.value); setResult(null) }} style={selectStyle}>
          {splitsForDataset.map(s => <option key={s} value={s}>{s}</option>)}
        </select>

        <select value={priorsFile} onChange={e => setPriorsFile(e.target.value)} style={selectStyle}>
          {priors.length === 0
            ? <option value="">No templates</option>
            : priors.map(p => <option key={p} value={p}>{p}</option>)
          }
        </select>

        <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
          <span style={{ fontSize: 11, color: 'var(--text-tertiary)', marginRight: 2, flexShrink: 0 }}>k</span>
          {K_OPTIONS.map(n => (
            <PillButton key={n} active={k === n} onClick={() => setK(n)}>{n}</PillButton>
          ))}
        </div>

        <button
          onClick={handleEvaluate}
          disabled={running || !dataset || !priorsFile}
          style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: running ? 'var(--bg-pill)' : 'var(--accent, #5a9a6a)',
            color: '#fff', fontSize: 12, fontWeight: 600,
            opacity: (!dataset || !priorsFile) ? 0.4 : 1, flexShrink: 0,
          }}
        >
          {running ? 'Evaluating…' : 'Evaluate Fitness'}
        </button>
      </div>

      {error && (
        <div style={{ fontSize: 11, color: '#e07070', background: 'rgba(224,112,112,0.08)', borderRadius: 6, padding: '8px 12px' }}>
          {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 20, flex: 1, minHeight: 0 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: 8 }}>
            Box Distribution
            {dims && <span style={{ fontWeight: 400, marginLeft: 6 }}>({dims.norm.length.toLocaleString()} boxes)</span>}
          </div>
          <InteractiveScatter dims={dims} k={k} onCentroidsChange={setLiveCentroids} />

          {derivedARs.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
              <span style={{ fontSize: 10, color: 'var(--text-tertiary)' }}>aspect ratios (w/h):</span>
              {derivedARs.map((ar, i) => (
                <span key={i} style={{
                  fontSize: 11, fontFamily: 'monospace', fontWeight: 600,
                  color: PALETTE[i % PALETTE.length],
                  background: `${PALETTE[i % PALETTE.length]}18`,
                  borderRadius: 4, padding: '2px 7px',
                }}>
                  {ar}
                </span>
              ))}
            </div>
          )}
        </div>

        <div style={{ width: 220, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <FitnessPanel result={result} />
          {(result || liveCentroids.length > 0) && (
            <button
              onClick={handleExport}
              disabled={exporting}
              style={{
                padding: '7px 16px', borderRadius: 6, border: '1px solid var(--border-subtle)',
                background: 'var(--bg-pill)', color: 'var(--text-primary)',
                fontSize: 12, cursor: 'pointer', fontWeight: 500,
              }}
            >
              {exporting ? 'Exporting…' : `Export → ${priorsFile}`}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function FitnessPanel({ result }: { result: ClusterResult | null }) {
  const tile = (label: string, value: string, color?: string) => (
    <div style={{
      padding: '10px 12px', borderRadius: 6,
      background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)',
    }}>
      <div style={{ fontSize: 9, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.4px', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: color ?? 'var(--text-primary)', fontFamily: 'monospace' }}>{value}</div>
    </div>
  )

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
        Fitness {!result && <span style={{ fontWeight: 400, opacity: 0.6 }}>(click Evaluate)</span>}
      </div>
      {result ? (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {tile('Mean IoU', result.fitness.mean_iou.toFixed(3), '#7c9ef5')}
            {tile('Recall @0.5', result.fitness['recall@0.5'].toFixed(3), '#7c9ef5')}
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {tile('Min Scale', result.min_scale.toFixed(4))}
            {tile('Max Scale', result.max_scale.toFixed(4))}
          </div>
        </>
      ) : (
        <div style={{ fontSize: 12, color: 'var(--text-tertiary)', padding: '8px 0' }}>
          Drag centroids to explore, then evaluate fitness against the backend priors.
        </div>
      )}
    </div>
  )
}
