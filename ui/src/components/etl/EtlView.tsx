import { useState, useEffect } from 'react'
import { StatusBadge } from '../StatusBadge'
import { fetchEtlFrames, fetchEtlAnnotations } from '../../api/client'
import type { EtlVideo, EtlFrame, EtlAnnotation, EtlStats } from './etlTypes'

const ANNO_COLORS = ['#8a9a6a', '#6a7a8a', '#9a8a6a', '#6a8a7a', '#7a6a8a', '#8a7a6a', '#7a8a9a']
const annoColor = (name: string) =>
  ANNO_COLORS[[...name].reduce((h, c) => Math.abs((h * 31 + c.charCodeAt(0)) | 0), 0) % ANNO_COLORS.length]

const MODELS = ['yolov8', 'rtdetr', 'grounding_dino']


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

const ColHeader = ({ children }: { children: React.ReactNode }) => (
  <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.2)', fontWeight: 700, letterSpacing: '0.8px', textTransform: 'uppercase' }}>
    {children}
  </span>
)


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
    <div
      data-testid="etl-view"
      style={{
        height: '100%', overflowY: 'auto',
        padding: '22px 24px 40px',
        display: 'flex', flexDirection: 'column', gap: 20,
        background: `
          radial-gradient(ellipse 60% 30% at 30% -5%, rgba(0,200,140,0.06) 0%, transparent 70%),
          radial-gradient(ellipse 35% 20% at 95% 90%, rgba(140,100,40,0.05) 0%, transparent 60%),
          var(--bg-primary)
        `,
      }}
    >

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 1 }}>
        {summaryCards.map(({ label, val }) => (
          <div
            key={label}
            data-testid={`stat-${label.toLowerCase()}`}
            style={{ position: 'relative', padding: '18px 20px 16px' }}
          >
            <div style={{
              position: 'absolute', top: 0, left: 20, right: 20, height: '1px',
              background: 'linear-gradient(90deg, transparent, rgba(138,154,106,0.4), transparent)',
            }} />
            <span style={{ display: 'block', fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.22)', marginBottom: 8 }}>
              {label}
            </span>
            <span style={{ display: 'block', fontSize: '32px', fontWeight: 700, color: '#dde2e0', letterSpacing: '-1.5px', lineHeight: 1, fontVariantNumeric: 'tabular-nums' }}>
              {val != null ? val.toLocaleString() : '—'}
            </span>
          </div>
        ))}
      </div>

      <div>
        <SectionLabel>Dataset Overview</SectionLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 240px', gap: 14 }}>

          <Glass>
            <div style={{ padding: '14px 16px 10px' }}>
              <div style={{ fontSize: '9px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', marginBottom: 12 }}>
                Videos — click a row to inspect
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 60px 44px 82px 54px 88px 72px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                {['Filename','Duration','FPS','Resolution','Frames','Annotations','Status'].map(h => (
                  <ColHeader key={h}>{h}</ColHeader>
                ))}
              </div>

              <div data-testid="video-list" style={{ maxHeight: 300, overflowY: 'auto' }}>
                {videos.length === 0 ? (
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.2)', padding: '16px 4px', fontFamily: 'monospace' }}>
                    No videos processed yet.
                  </div>
                ) : videos.map(v => {
                  const isSel = selectedVideo?.id === v.id
                  return (
                    <div
                      key={v.id}
                      data-testid={`video-row-${v.id}`}
                      role="button"
                      aria-selected={isSel}
                      onClick={() => setSelectedVideo(isSel ? null : v)}
                      style={{
                        display: 'grid', gridTemplateColumns: '1fr 60px 44px 82px 54px 88px 72px', gap: 8,
                        padding: '8px 4px',
                        borderBottom: '1px solid rgba(255,255,255,0.035)',
                        borderLeft: `2px solid ${isSel ? 'rgba(138,154,106,0.7)' : 'transparent'}`,
                        paddingLeft: 6,
                        alignItems: 'center', cursor: 'pointer',
                        background: isSel ? 'rgba(138,154,106,0.05)' : 'transparent',
                        transition: 'background 0.15s',
                      }}
                    >
                      <span style={{ fontSize: '11px', color: isSel ? '#8a9a6a' : 'rgba(255,255,255,0.6)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v.filename}</span>
                      <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{v.duration.toFixed(1)}s</span>
                      <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{v.fps.toFixed(0)}</span>
                      <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{v.width}×{v.height}</span>
                      <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{v.frames}</span>
                      <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>{v.annotations.toLocaleString()}</span>
                      <StatusBadge status={v.status as never} />
                    </div>
                  )
                })}
              </div>
            </div>
          </Glass>

          <Glass>
            <div style={{ padding: '14px 16px 12px' }}>
              <div style={{ fontSize: '9px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', marginBottom: 14 }}>
                Class Distribution
              </div>
              <div data-testid="class-dist" style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {dist.map(c => {
                  const pct = (c.count / maxCount) * 100
                  return (
                    <div key={c.class_name} data-testid={`dist-row-${c.class_name}`}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                        <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.4)' }}>{c.class_name}</span>
                        <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>{c.count.toLocaleString()}</span>
                      </div>
                      <div style={{ height: 4, background: 'rgba(255,255,255,0.04)', overflow: 'hidden' }}>
                        <div
                          data-testid={`dist-bar-${c.class_name}`}
                          style={{ height: '100%', width: `${pct}%`, background: '#8a9a6a', opacity: 0.55, transition: 'width 0.6s cubic-bezier(0.22,1,0.36,1)' }}
                        />
                      </div>
                    </div>
                  )
                })}
                {dist.length === 0 && (
                  <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', fontFamily: 'monospace' }}>No annotations yet.</span>
                )}
              </div>
            </div>
          </Glass>
        </div>
      </div>

      {selectedVideo && (
        <div>
          <SectionLabel>Frame Inspector</SectionLabel>
          <Glass>
            <div data-testid="frame-inspector" style={{ overflow: 'hidden' }}>

              <div style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '10px 16px',
                borderBottom: '1px solid rgba(255,255,255,0.05)',
              }}>
                <span style={{ fontSize: '11px', fontWeight: 700, color: '#8a9a6a', fontFamily: 'monospace' }}>
                  {selectedVideo.filename}
                </span>
                <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>
                  {frames.length} frames sampled
                </span>
                <button
                  data-testid="close-inspector"
                  onClick={() => setSelectedVideo(null)}
                  style={{
                    marginLeft: 'auto', fontSize: '10px', color: 'rgba(255,255,255,0.25)',
                    background: 'none', border: '1px solid rgba(255,255,255,0.08)',
                    cursor: 'pointer', fontFamily: 'monospace', padding: '3px 10px',
                    borderRadius: 4, letterSpacing: '0.5px',
                  }}
                >
                  ✕ close
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr', minHeight: 240 }}>

                <div data-testid="frame-list" style={{ borderRight: '1px solid rgba(255,255,255,0.05)', overflowY: 'auto', maxHeight: 420 }}>
                  <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', padding: '10px 12px 6px' }}>
                    Frames
                  </div>
                  {frames.length === 0 && (
                    <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', padding: '12px', fontFamily: 'monospace' }}>No frames.</div>
                  )}
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
                          background: isSel ? 'rgba(138,154,106,0.06)' : 'transparent',
                          borderLeft: `2px solid ${isSel ? 'rgba(138,154,106,0.7)' : 'transparent'}`,
                          transition: 'background 0.12s',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                          <span style={{ fontSize: '11px', fontWeight: 600, color: isSel ? '#8a9a6a' : 'rgba(255,255,255,0.55)', fontFamily: 'monospace' }}>
                            #{f.frame_index}
                          </span>
                          <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace' }}>{f.annotation_count} ann</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.18)', fontFamily: 'monospace' }}>t={f.timestamp_s.toFixed(1)}s</span>
                          <span style={{ fontSize: '9.5px', color: 'rgba(255,255,255,0.18)', fontFamily: 'monospace' }}>Δ={f.scene_change_score.toFixed(2)}</span>
                        </div>
                      </div>
                    )
                  })}
                </div>

                <div data-testid="annotation-detail" style={{ padding: '14px 16px', overflowY: 'auto', maxHeight: 420 }}>
                  {!selectedFrame ? (
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'rgba(255,255,255,0.15)', fontSize: '11px', fontFamily: 'monospace' }}>
                      Select a frame to inspect annotations
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                      <div data-testid="annotation-table">
                        <div style={{ display: 'grid', gridTemplateColumns: '90px 44px 52px 68px 68px 110px', gap: 6, marginBottom: 6 }}>
                          {['Class', 'Votes', 'Cons.', ...MODELS].map(h => <ColHeader key={h}>{h}</ColHeader>)}
                        </div>
                        <div style={{ fontSize: '8.5px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', marginBottom: 8 }}>
                          Model Votes — Frame #{selectedFrame.frame_index}
                        </div>
                        {annotations.map(ann => (
                          <div
                            key={ann.id}
                            data-testid={`ann-row-${ann.id}`}
                            style={{
                              display: 'grid', gridTemplateColumns: '90px 44px 52px 68px 68px 110px', gap: 6,
                              padding: '7px 0', borderBottom: '1px solid rgba(255,255,255,0.04)', alignItems: 'center',
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                              <div style={{ width: 5, height: 5, background: annoColor(ann.class_name), flexShrink: 0 }} />
                              <span style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.55)' }}>{ann.class_name}</span>
                            </div>
                            <span style={{
                              fontSize: '11px', fontWeight: 700, fontFamily: 'monospace',
                              color: ann.votes === 3 ? '#8a9a6a' : 'rgba(232,133,72,0.8)',
                            }}>
                              {ann.votes}/3
                            </span>
                            <span style={{ fontSize: '11px', color: 'rgba(255,255,255,0.35)', fontFamily: 'monospace' }}>
                              {(ann.consensus_score * 100).toFixed(0)}%
                            </span>
                            {MODELS.map(m => (
                              <span key={m} style={{ fontSize: '11px', fontFamily: 'monospace', color: ann.model_confidences[m] != null ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.12)' }}>
                                {ann.model_confidences[m] != null ? (ann.model_confidences[m]! * 100).toFixed(0) + '%' : '—'}
                              </span>
                            ))}
                          </div>
                        ))}
                        {annotations.length === 0 && (
                          <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.18)', padding: '12px 0', fontFamily: 'monospace' }}>
                            No annotation data for this frame.
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </Glass>
        </div>
      )}
    </div>
  )
}
