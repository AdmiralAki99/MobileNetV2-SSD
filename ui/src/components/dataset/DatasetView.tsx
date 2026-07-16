import { useEffect, useState } from 'react'
import { fetchDatasets, fetchBoxDims, launchTfrecords } from '../../api/client'
import type { LedgerEntry, BoxDimsResponse } from './datasetTypes'

const BAR_COL = '#8a9a6a'
const HW = 420, HH = 200
const HP = { top: 12, right: 16, bottom: 32, left: 44 }
const NUM_BINS = 24

function computeBins(norms: [number, number][], numBins: number) {
  if (norms.length === 0) return []
  const areas = norms.map(([w, h]) => w * h)
  const maxA = Math.max(...areas)
  const binW = maxA / numBins
  const counts = new Array(numBins).fill(0)
  for (const a of areas) {
    const idx = Math.min(Math.floor(a / binW), numBins - 1)
    counts[idx]++
  }
  return counts.map((count, i) => ({ lo: i * binW, hi: (i + 1) * binW, count }))
}

function AreaHistogram({ dims }: { dims: BoxDimsResponse | null }) {
  if (!dims || dims.norm.length === 0) {
    return (
      <div style={{
        width: '100%', height: HH + HP.top + HP.bottom,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        border: '1px dashed var(--border-subtle)', borderRadius: 8,
        color: 'var(--text-tertiary)', fontSize: 12,
      }}>
        Select a split to view box size distribution
      </div>
    )
  }

  const bins = computeBins(dims.norm, NUM_BINS)
  const maxCount = Math.max(...bins.map(b => b.count), 1)
  const plotW = HW - HP.left - HP.right
  const plotH = HH - HP.top - HP.bottom
  const barW = plotW / NUM_BINS
  const gap = 1

  const yTicks = [0, 0.25, 0.5, 0.75, 1.0].map(f => Math.round(f * maxCount))

  return (
    <svg viewBox={`0 0 ${HW} ${HH + HP.top + HP.bottom}`} width="100%" style={{ display: 'block' }}>
      {yTicks.map((t, i) => {
        const y = HP.top + plotH - (t / maxCount) * plotH
        return (
          <g key={i}>
            <line x1={HP.left} x2={HW - HP.right} y1={y} y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth={1} />
            <text x={HP.left - 4} y={y + 3} textAnchor="end" fontSize={8} fill="rgba(255,255,255,0.25)">
              {t >= 1000 ? `${(t / 1000).toFixed(0)}k` : t}
            </text>
          </g>
        )
      })}

      {bins.map((bin, i) => {
        const barH = (bin.count / maxCount) * plotH
        const x = HP.left + i * barW + gap / 2
        const y = HP.top + plotH - barH
        return (
          <rect key={i} x={x} y={y} width={barW - gap} height={barH}
            fill={BAR_COL} opacity={0.75} rx={1} />
        )
      })}

      <line x1={HP.left} x2={HW - HP.right} y1={HP.top + plotH} y2={HP.top + plotH}
        stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
      <line x1={HP.left} x2={HP.left} y1={HP.top} y2={HP.top + plotH}
        stroke="rgba(255,255,255,0.08)" strokeWidth={1} />

      <text x={HW / 2} y={HP.top + plotH + HP.bottom - 4} textAnchor="middle"
        fontSize={8.5} fill="rgba(255,255,255,0.3)">norm area (w × h)</text>
    </svg>
  )
}

const selectStyle = {
  background: 'var(--bg-pill, #1a1f1a)',
  border: '1px solid var(--border-subtle)',
  borderRadius: 6, color: 'var(--text-primary)', fontSize: 12,
  padding: '5px 10px', cursor: 'pointer', outline: 'none', height: 30,
} as const

export const DatasetView = () => {
  const [entries, setEntries]       = useState<LedgerEntry[]>([])
  const [selected, setSelected]     = useState<LedgerEntry | null>(null)
  const [dims, setDims]             = useState<BoxDimsResponse | null>(null)
  const [dimsLoading, setDimsLoading] = useState(false)
  const [configPath, setConfigPath] = useState('')
  const [statsOnly, setStatsOnly]   = useState(false)
  const [launching, setLaunching]   = useState(false)
  const [launchMsg, setLaunchMsg]   = useState<string | null>(null)
  const [error, setError]           = useState<string | null>(null)

  useEffect(() => {
    fetchDatasets()
      .then((d: LedgerEntry[]) => setEntries(d))
      .catch(() => {})
  }, [])

  const handleSelect = (entry: LedgerEntry) => {
    if (selected?.name === entry.name && selected?.split === entry.split) return
    setSelected(entry)
    setDims(null)
    setDimsLoading(true)
    fetchBoxDims(entry.name, entry.split)
      .then((d: BoxDimsResponse) => setDims(d))
      .catch(() => setDims(null))
      .finally(() => setDimsLoading(false))
  }

  const handleLaunch = async () => {
    if (!configPath.trim()) return
    setLaunching(true); setLaunchMsg(null); setError(null)
    try {
      const res = await launchTfrecords({ config_path: configPath.trim(), stats_only: statsOnly })
      setLaunchMsg(`Triggered — run ID: ${res.dag_run_id}`)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Launch failed')
    } finally {
      setLaunching(false)
    }
  }

  const totalImages = entries.reduce((s, e) => s + e.num_images, 0)
  const totalBoxes  = entries.reduce((s, e) => s + e.num_boxes, 0)

  return (
    <div style={{ height: '100%', overflow: 'auto', padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.2px' }}>Dataset Health</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 2 }}>
          Ledger summary and TFRecord generation launcher
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <input
          value={configPath}
          onChange={e => setConfigPath(e.target.value)}
          placeholder="configs/experiments/exp002_cloud_run.yaml"
          style={{
            ...selectStyle, width: 340, padding: '5px 10px',
            fontFamily: 'monospace', fontSize: 11,
          }}
        />
        <div style={{ display: 'flex', gap: 1, borderRadius: 6, overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
          {(['Full', 'Stats only'] as const).map((label, i) => {
            const active = i === 0 ? !statsOnly : statsOnly
            return (
              <button key={label} onClick={() => setStatsOnly(i === 1)} style={{
                padding: '5px 12px', fontSize: 12, border: 'none', cursor: 'pointer',
                background: active ? 'var(--accent, #5a9a6a)' : 'var(--bg-pill)',
                color: active ? '#fff' : 'var(--text-secondary)',
              }}>{label}</button>
            )
          })}
        </div>
        <button
          onClick={handleLaunch}
          disabled={launching || !configPath.trim()}
          style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: 'var(--accent, #5a9a6a)', color: '#fff',
            fontSize: 12, fontWeight: 600,
            opacity: !configPath.trim() ? 0.4 : 1,
          }}
        >
          {launching ? 'Launching…' : 'Generate TFRecords'}
        </button>
        {launchMsg && <span style={{ fontSize: 11, color: '#8a9a6a' }}>{launchMsg}</span>}
        {error && <span style={{ fontSize: 11, color: '#e07070' }}>{error}</span>}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, maxWidth: 480 }}>
        {[
          { label: 'Splits', value: entries.length },
          { label: 'Images', value: totalImages.toLocaleString() },
          { label: 'Boxes', value: totalBoxes.toLocaleString() },
        ].map(({ label, value }) => (
          <div key={label} style={{
            padding: '12px 14px', borderRadius: 8,
            background: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)',
          }}>
            <div style={{ fontSize: 9, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 4 }}>{label}</div>
            <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{value}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 20, flex: 1, minHeight: 0 }}>
        <div style={{ flex: '0 0 420px', overflow: 'auto' }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
            Splits
          </div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr>
                {['Dataset', 'Split', 'Images', 'Boxes'].map(h => (
                  <th key={h} style={{
                    padding: '6px 10px', textAlign: 'left',
                    fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)',
                    textTransform: 'uppercase', letterSpacing: '0.4px',
                    borderBottom: '1px solid var(--border-subtle)',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {entries.length === 0 ? (
                <tr>
                  <td colSpan={4} style={{ padding: '16px 10px', color: 'var(--text-tertiary)', fontSize: 12 }}>
                    No datasets in ledger
                  </td>
                </tr>
              ) : (
                entries.map((e, i) => {
                  const isSelected = selected?.name === e.name && selected?.split === e.split
                  return (
                    <tr
                      key={i}
                      onClick={() => handleSelect(e)}
                      style={{
                        cursor: 'pointer',
                        background: isSelected ? 'rgba(138,154,106,0.12)' : 'transparent',
                        borderLeft: isSelected ? `2px solid ${BAR_COL}` : '2px solid transparent',
                      }}
                    >
                      <td style={{ padding: '8px 10px', color: 'var(--text-primary)', fontFamily: 'monospace', fontSize: 11 }}>{e.name}</td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-secondary)' }}>{e.split}</td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-secondary)', textAlign: 'right', fontFamily: 'monospace' }}>{e.num_images.toLocaleString()}</td>
                      <td style={{ padding: '8px 10px', color: 'var(--text-secondary)', textAlign: 'right', fontFamily: 'monospace' }}>{e.num_boxes.toLocaleString()}</td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 8 }}>
            Box Area Distribution
            {selected && <span style={{ fontWeight: 400, marginLeft: 6 }}>{selected.name} / {selected.split}</span>}
          </div>
          {dimsLoading
            ? <div style={{ fontSize: 12, color: 'var(--text-tertiary)', padding: '16px 0' }}>Loading…</div>
            : <AreaHistogram dims={dims} />
          }
        </div>
      </div>
    </div>
  )
}
