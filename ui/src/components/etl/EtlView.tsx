import { useState, useEffect } from 'react'
import { StatusBadge } from '../StatusBadge'
import { fetchEtlFrames, fetchEtlAnnotations } from '../../api/client'
import type { EtlVideo, EtlFrame, EtlAnnotation, EtlStats } from './etlTypes'

const ANNO_COLORS = ['#00d4a0', '#e88548', '#7b9ef0', '#e84855', '#65c16a', '#c67ef0', '#f0d07e']
const annoColor = (name: string) =>
  ANNO_COLORS[[...name].reduce((h, c) => Math.abs((h * 31 + c.charCodeAt(0)) | 0), 0) % ANNO_COLORS.length]

const MODELS = ['yolov8', 'rtdetr', 'grounding_dino']

const SEC: React.CSSProperties = {
  fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)',
  letterSpacing: '0.5px', textTransform: 'uppercase',
}

interface Props { statsData?: EtlStats; videosData?: EtlVideo[] }

export const EtlView = ({ statsData, videosData }: Props) => {
  const stats  = statsData  ?? {}
  const videos = videosData ?? []
  const dist   = stats.class_distribution ?? []
  const maxCount = dist.reduce((m, c) => Math.max(m, c.count), 1)

  const [selectedVideo,  setSelectedVideo]  = useState<EtlVideo | null>(null)
  const [selectedFrame,  setSelectedFrame]  = useState<EtlFrame | null>(null)
  const [frames,         setFrames]         = useState<EtlFrame[]>([])
  const [annotations,    setAnnotations]    = useState<EtlAnnotation[]>([])

  useEffect(() => {
    if (!selectedVideo) { setFrames([]); setSelectedFrame(null); setAnnotations([]); return }
    fetchEtlFrames(selectedVideo.id).then(setFrames)
    setSelectedFrame(null)
    setAnnotations([])
  }, [selectedVideo?.id])

  useEffect(() => {
    if (!selectedFrame) { setAnnotations([]); return }
    fetchEtlAnnotations(selectedFrame.id).then(setAnnotations)
  }, [selectedFrame?.id])

  const summaryCards = [
    { label: 'Videos',      val: stats.total_videos      },
    { label: 'Frames',      val: stats.total_frames      },
    { label: 'Annotations', val: stats.total_annotations },
  ]

  return (
    <div data-testid="etl-view" style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
        {summaryCards.map(({ label, val }) => (
          <div key={label} data-testid={`stat-${label.toLowerCase()}`}
            style={{ padding: '14px 18px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ ...SEC, marginBottom: 6 }}>{label}</div>
            <span style={{ fontSize: '28px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>
              {val != null ? val.toLocaleString() : '—'}
            </span>
          </div>
        ))}
      </div>

      {/* Video table + class distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 260px', gap: 16 }}>

        {/* Video table */}
        <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ ...SEC, marginBottom: 12 }}>Videos — click a row to inspect</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 62px 46px 86px 58px 90px 76px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid var(--border-subtle)' }}>
            {['Filename', 'Duration', 'FPS', 'Resolution', 'Frames', 'Annotations', 'Status'].map(h => (
              <span key={h} style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontWeight: 600, letterSpacing: '0.4px', textTransform: 'uppercase' }}>{h}</span>
            ))}
          </div>
          <div data-testid="video-list" style={{ maxHeight: 320, overflowY: 'auto' }}>
            {videos.length === 0
              ? <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '16px 4px' }}>No videos processed yet.</div>
              : videos.map(v => {
                  const isSel = selectedVideo?.id === v.id
                  return (
                    <div
                      key={v.id}
                      data-testid={`video-row-${v.id}`}
                      role="button"
                      aria-selected={isSel}
                      onClick={() => setSelectedVideo(isSel ? null : v)}
                      style={{
                        display: 'grid', gridTemplateColumns: '1fr 62px 46px 86px 58px 90px 76px', gap: 8,
                        padding: '8px 4px', paddingLeft: 6, borderBottom: '1px solid var(--border-subtle)',
                        alignItems: 'center', cursor: 'pointer',
                        background: isSel ? 'rgba(0,212,160,0.06)' : 'transparent',
                        borderLeft: isSel ? '2px solid var(--accent)' : '2px solid transparent',
                      }}
                    >
                      <span style={{ fontSize: '12px', color: isSel ? 'var(--accent)' : 'var(--text-primary)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v.filename}</span>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.duration.toFixed(1)}s</span>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.fps.toFixed(0)}</span>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.width}×{v.height}</span>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.frames}</span>
                      <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.annotations.toLocaleString()}</span>
                      <StatusBadge status={v.status as never} />
                    </div>
                  )
                })
            }
          </div>
        </div>

        {/* Class distribution */}
        <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ ...SEC, marginBottom: 14 }}>Class Distribution</div>
          <div data-testid="class-dist" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {dist.map(c => (
              <div key={c.class_name} data-testid={`dist-row-${c.class_name}`}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>{c.class_name}</span>
                  <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{c.count.toLocaleString()}</span>
                </div>
                <div style={{ height: 4, borderRadius: 999, background: 'var(--bg-pill)', overflow: 'hidden' }}>
                  <div data-testid={`dist-bar-${c.class_name}`} style={{ height: '100%', borderRadius: 999, width: `${(c.count / maxCount) * 100}%`, background: 'var(--accent)' }}/>
                </div>
              </div>
            ))}
            {dist.length === 0 && <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>No annotations yet.</span>}
          </div>
        </div>
      </div>

      {/* Frame inspector */}
      {selectedVideo && (
        <div data-testid="frame-inspector" style={{ borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid rgba(0,212,160,0.25)', overflow: 'hidden' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(0,212,160,0.04)' }}>
            <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent)', fontFamily: 'monospace' }}>{selectedVideo.filename}</span>
            <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{frames.length} frames sampled</span>
            <button
              data-testid="close-inspector"
              onClick={() => setSelectedVideo(null)}
              style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-tertiary)', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit', padding: '2px 8px', borderRadius: 6 }}
            >✕ close</button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '190px 1fr', minHeight: 260 }}>
            {/* Frame list */}
            <div data-testid="frame-list" style={{ borderRight: '1px solid var(--border-subtle)', overflowY: 'auto', maxHeight: 440 }}>
              <div style={{ ...SEC, padding: '10px 12px 6px' }}>Frames</div>
              {frames.length === 0 && <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '12px' }}>No frames.</div>}
              {frames.map(f => {
                const isSel = selectedFrame?.id === f.id
                return (
                  <div
                    key={f.id}
                    data-testid={`frame-row-${f.id}`}
                    role="button"
                    aria-selected={isSel}
                    onClick={() => setSelectedFrame(isSel ? null : f)}
                    style={{
                      padding: '8px 12px', cursor: 'pointer',
                      background: isSel ? 'rgba(0,212,160,0.08)' : 'transparent',
                      borderLeft: isSel ? '2px solid var(--accent)' : '2px solid transparent',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                      <span style={{ fontSize: '11.5px', fontWeight: 600, color: isSel ? 'var(--accent)' : 'var(--text-primary)', fontFamily: 'monospace' }}>#{f.frame_index}</span>
                      <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{f.annotation_count} ann</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>t={f.timestamp_s.toFixed(1)}s</span>
                      <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>Δ={f.scene_change_score.toFixed(2)}</span>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Annotation detail */}
            <div data-testid="annotation-detail" style={{ padding: '14px 16px', overflowY: 'auto', maxHeight: 440 }}>
              {!selectedFrame ? (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-tertiary)', fontSize: '12px' }}>
                  Select a frame to inspect annotations
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div data-testid="annotation-table">
                    <div style={{ ...SEC, marginBottom: 8 }}>Model Votes — Frame #{selectedFrame.frame_index}</div>
                    {annotations.map(ann => (
                      <div key={ann.id} data-testid={`ann-row-${ann.id}`} style={{ display: 'grid', gridTemplateColumns: '90px 44px 52px 68px 68px 110px', gap: 6, padding: '6px 0', borderBottom: '1px solid var(--border-subtle)', alignItems: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                          <div style={{ width: 6, height: 6, borderRadius: 1, background: annoColor(ann.class_name), flexShrink: 0 }}/>
                          <span style={{ fontSize: '11px', color: 'var(--text-primary)' }}>{ann.class_name}</span>
                        </div>
                        <span style={{ fontSize: '11px', fontWeight: 600, fontFamily: 'monospace', color: ann.votes === 3 ? 'var(--success)' : 'var(--warning)' }}>{ann.votes}/3</span>
                        <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{(ann.consensus_score * 100).toFixed(0)}%</span>
                        {MODELS.map(m => (
                          <span key={m} style={{ fontSize: '11px', fontFamily: 'monospace', color: ann.model_confidences[m] != null ? 'var(--text-primary)' : 'var(--text-tertiary)' }}>
                            {ann.model_confidences[m] != null ? (ann.model_confidences[m]! * 100).toFixed(0) + '%' : '—'}
                          </span>
                        ))}
                      </div>
                    ))}
                    {annotations.length === 0 && <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '12px 0' }}>No annotation data for this frame.</div>}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
