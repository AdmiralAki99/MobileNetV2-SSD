import { useState, useMemo, useEffect, useRef, useCallback } from 'react'
import { launchTraining, fetchConfigLibrary, refreshConfigLibrary, registerExperiment, saveConfig } from '../../api/client'
import type { ConfigLibrary, ExperimentConfig } from '../create/createTypes'


const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div style={{
    fontSize: '8px', fontWeight: 700, letterSpacing: '1.5px',
    textTransform: 'uppercase', color: 'rgba(255,255,255,0.15)',
    marginBottom: 12,
  }}>
    {children}
  </div>
)

const Glass = ({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) => (
  <div style={{ position: 'relative', ...style }}>
    {(['tl','tr','bl','br'] as const).map(c => {
      const top = c.startsWith('t'), left = c.endsWith('l')
      const d = top
        ? left  ? 'M14 1 L1 1 L1 14'  : 'M0 1 L13 1 L13 14'
        : left  ? 'M14 13 L1 13 L1 0' : 'M0 13 L13 13 L13 0'
      return (
        <svg key={c} aria-hidden width="14" height="14" style={{
          position: 'absolute', pointerEvents: 'none', zIndex: 1,
          top: top ? 0 : undefined, bottom: !top ? 0 : undefined,
          left: left ? 0 : undefined, right: !left ? 0 : undefined,
        }}>
          <path d={d} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1" />
        </svg>
      )
    })}
    {children}
  </div>
)


const ICONS: Record<string, React.ReactNode> = {
  backbone: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <rect x="1" y="1" width="11" height="2.5" rx="0.5" fill="currentColor" opacity="0.9"/>
      <rect x="1" y="5.25" width="11" height="2.5" rx="0.5" fill="currentColor" opacity="0.65"/>
      <rect x="1" y="9.5" width="11" height="2.5" rx="0.5" fill="currentColor" opacity="0.4"/>
    </svg>
  ),
  data: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <ellipse cx="6.5" cy="3" rx="4.5" ry="1.8" stroke="currentColor" strokeWidth="1" fill="none"/>
      <path d="M2 3v4c0 1 2 1.8 4.5 1.8S11 8 11 7V3" stroke="currentColor" strokeWidth="1" fill="none"/>
      <path d="M2 7v3c0 1 2 1.8 4.5 1.8S11 11 11 10V7" stroke="currentColor" strokeWidth="1" fill="none"/>
    </svg>
  ),
  augmentation: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M6.5 1 L7.5 4 L10.5 4 L8.2 6 L9.2 9 L6.5 7.2 L3.8 9 L4.8 6 L2.5 4 L5.5 4 Z" stroke="currentColor" strokeWidth="0.9" fill="none" strokeLinejoin="round"/>
      <circle cx="10.5" cy="10.5" r="1.2" fill="currentColor" opacity="0.6"/>
      <circle cx="2.5" cy="10.5" r="0.8" fill="currentColor" opacity="0.4"/>
    </svg>
  ),
  optimizer: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <line x1="1" y1="3.5" x2="12" y2="3.5" stroke="currentColor" strokeWidth="1"/>
      <circle cx="4.5" cy="3.5" r="1.5" fill="var(--bg-primary)" stroke="currentColor" strokeWidth="1"/>
      <line x1="1" y1="9.5" x2="12" y2="9.5" stroke="currentColor" strokeWidth="1"/>
      <circle cx="8.5" cy="9.5" r="1.5" fill="var(--bg-primary)" stroke="currentColor" strokeWidth="1"/>
    </svg>
  ),
  loss: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M1 9 C2.5 9 2.5 4 4 4 C5.5 4 5.5 9 7 9 C8.5 9 8.5 4 10 4 C11 4 11.5 6.5 12 6.5" stroke="currentColor" strokeWidth="1" fill="none" strokeLinecap="round"/>
    </svg>
  ),
  priors: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <rect x="1" y="1" width="4" height="4" rx="0.5" stroke="currentColor" strokeWidth="1" fill="none"/>
      <rect x="8" y="1" width="4" height="4" rx="0.5" stroke="currentColor" strokeWidth="1" fill="none"/>
      <rect x="1" y="8" width="4" height="4" rx="0.5" stroke="currentColor" strokeWidth="1" fill="none"/>
      <rect x="8" y="8" width="4" height="4" rx="0.5" stroke="currentColor" strokeWidth="1" fill="none"/>
      <circle cx="6.5" cy="6.5" r="1" fill="currentColor" opacity="0.5"/>
    </svg>
  ),
  heads: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <circle cx="3" cy="4" r="1.5" stroke="currentColor" strokeWidth="1" fill="none"/>
      <circle cx="3" cy="9" r="1.5" stroke="currentColor" strokeWidth="1" fill="none"/>
      <path d="M4.5 4 L8 6.5 M4.5 9 L8 6.5 M8 6.5 L11 6.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round"/>
      <circle cx="11" cy="6.5" r="1.2" fill="currentColor"/>
    </svg>
  ),
  eval: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <rect x="1" y="7" width="2.5" height="5" rx="0.5" fill="currentColor" opacity="0.4"/>
      <rect x="5.25" y="4" width="2.5" height="8" rx="0.5" fill="currentColor" opacity="0.65"/>
      <rect x="9.5" y="1" width="2.5" height="11" rx="0.5" fill="currentColor" opacity="0.9"/>
    </svg>
  ),
  train: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M3 2 L11 6.5 L3 11 Z" stroke="currentColor" strokeWidth="1" fill="none" strokeLinejoin="round"/>
    </svg>
  ),
  checkpoint: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M2.5 1 L10.5 1 L10.5 12 L6.5 9.5 L2.5 12 Z" stroke="currentColor" strokeWidth="1" fill="none" strokeLinejoin="round"/>
    </svg>
  ),
  runtime: (
    <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
      <path d="M6.5 1 L7.8 4.2 L6.5 4.8 L5.2 4.2 Z" fill="currentColor" opacity="0.9"/>
      <path d="M6.5 12 L7.8 8.8 L6.5 8.2 L5.2 8.8 Z" fill="currentColor" opacity="0.4"/>
      <path d="M1 6.5 L4.2 5.2 L4.8 6.5 L4.2 7.8 Z" fill="currentColor" opacity="0.65"/>
      <path d="M12 6.5 L8.8 5.2 L8.2 6.5 L8.8 7.8 Z" fill="currentColor" opacity="0.65"/>
      <circle cx="6.5" cy="6.5" r="1.8" stroke="currentColor" strokeWidth="1" fill="none"/>
    </svg>
  ),
}


const Spinner = ({ size = 14 }: { size?: number }) => (
  <>
    <div style={{
      width: size, height: size, borderRadius: '50%',
      border: `1.5px solid rgba(255,255,255,0.08)`,
      borderTopColor: 'rgba(255,255,255,0.45)',
      animation: 'cfg-spin 0.65s linear infinite',
      flexShrink: 0,
    }} />
    <style>{`@keyframes cfg-spin { to { transform: rotate(360deg) } }`}</style>
  </>
)


interface CardDef {
  id:       string
  category: string
  name:     string
  filename: string
  tags:     string[]
  content:  Record<string, unknown>
}

const CATEGORY_COLOR: Record<string, string> = {
  backbone:     '#00d4a0',
  data:         '#7c9ef5',
  augmentation: '#c8a45a',
  optimizer:    '#8a9a6a',
  loss:         '#c87070',
  priors:       '#9a7ac8',
  heads:        '#6a9ab0',
  eval:         '#7ab08a',
  train:        '#c8a45a',
  checkpoint:   'rgba(255,255,255,0.4)',
  runtime:      'rgba(255,255,255,0.4)',
}

const API_CAT_TO_SLOT: Record<string, string> = {
  backbones: 'backbone', augmentations: 'augmentation', optimizers: 'optimizer',
  losses: 'loss', data: 'data', priors: 'priors', heads: 'heads',
  eval: 'eval', train: 'train', checkpoint: 'checkpoint', runtime: 'runtime',
  samplers: 'sampler', logging: 'logging', export: 'export',
}

function tagsFromContent(cat: string, c: Record<string, unknown>): string[] {
  const tags: string[] = []
  try {
    if (cat === 'backbone')     { if (c.width_mult) tags.push(`×${c.width_mult}`); if (c.pretrained) tags.push('pretrained') }
    if (cat === 'data')         { if (c.num_classes) tags.push(`${c.num_classes} cls`); const s = c.input_size as number[]|undefined; if (s?.[0]) tags.push(`${s[0]}²`) }
    if (cat === 'optimizer')    { if (c.lr != null) tags.push(`lr ${c.lr}`) }
    if (cat === 'loss')         { const cl = (c.classification as any)?.type ?? c.classification; if (cl) tags.push(String(cl)) }
    if (cat === 'priors')       { const s = c.image_size as number[]|undefined; if (s?.[0]) tags.push(`${s[0]}²`); if (c.min_scale) tags.push(`min ${c.min_scale}`) }
    if (cat === 'train')        { if (c.epochs) tags.push(`${c.epochs} ep`); if (c.batch_size) tags.push(`bs ${c.batch_size}`) }
    if (cat === 'eval')         { if (c.score_threshold != null) tags.push(`thr ${c.score_threshold}`) }
    if (cat === 'runtime')      { if (c.num_workers) tags.push(`${c.num_workers}w`) }
    if (cat === 'checkpoint')   { if (c.keep_last_k) tags.push(`keep ${c.keep_last_k}`) }
  } catch {}
  return tags.slice(0, 3)
}

function libraryToCards(lib: ConfigLibrary): CardDef[] {
  const cards: CardDef[] = []
  for (const [apiCat, items] of Object.entries(lib)) {
    if (!items) continue
    const slotCat = API_CAT_TO_SLOT[apiCat] ?? apiCat
    for (const item of items) {
      const content = item.content ?? {}
      cards.push({
        id:       `${slotCat}__${item.name}`,
        category: slotCat,
        name:     item.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        filename: item.path,
        tags:     tagsFromContent(slotCat, content),
        content,
      })
    }
  }
  return cards
}

const SLOT_ORDER = ['backbone','data','augmentation','optimizer','loss','priors','heads','eval','train','checkpoint','runtime']
const REQUIRED   = new Set(['backbone','data','augmentation','optimizer','loss','priors'])
const SLOT_LABEL: Record<string, string> = {
  backbone:'Backbone', data:'Dataset', augmentation:'Augmentation', optimizer:'Optimizer',
  loss:'Loss', priors:'Anchors', heads:'Head', eval:'Evaluation',
  train:'Training', checkpoint:'Checkpoint', runtime:'Runtime',
}


function serVal(val: unknown, indent: number): string {
  const p = ' '.repeat(indent)
  if (val === null || val === undefined) return 'null'
  if (typeof val === 'boolean' || typeof val === 'number') return String(val)
  if (typeof val === 'string')
    return /[:{}\[\],#&*?|<>=!%@`]/.test(val) ? `"${val.replace(/"/g, '\\"')}"` : val
  if (Array.isArray(val)) {
    if (val.every(v => typeof v !== 'object' || v === null))
      return `[${val.map(v => serVal(v, 0)).join(', ')}]`
    return '\n' + val.map(v => `${p}- ${serVal(v, indent + 2)}`).join('\n')
  }
  if (typeof val === 'object') {
    const ents = Object.entries(val as Record<string, unknown>)
    if (!ents.length) return '{}'
    return '\n' + ents.map(([k, v]) => {
      const r = serVal(v, indent + 2)
      return r.startsWith('\n') ? `${p}${k}:${r}` : `${p}${k}: ${r}`
    }).join('\n')
  }
  return String(val)
}

function serBlock(obj: Record<string, unknown>, base = 2): string[] {
  const p = ' '.repeat(base)
  return Object.entries(obj).map(([k, v]) => {
    const r = serVal(v, base + 2)
    return r.startsWith('\n') ? `${p}${k}:${r}` : `${p}${k}: ${r}`
  })
}


function stripInlineComment(line: string): string {
  let inQuote = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (ch === '"') inQuote = !inQuote
    if (!inQuote && ch === '#' && (i === 0 || line[i - 1] === ' ')) return line.slice(0, i).trimEnd()
  }
  return line
}

function parseScalar(raw: string): unknown {
  const s = raw.trim()
  if (s === '' || s === 'null' || s === '~') return null
  if (s === 'true') return true
  if (s === 'false') return false
  if (/^-?\d+$/.test(s)) return parseInt(s, 10)
  if (/^-?\d*\.\d+$/.test(s)) return parseFloat(s)
  if (/^".*"$/.test(s)) return s.slice(1, -1).replace(/\\"/g, '"')
  if (/^\[.*\]$/.test(s)) return parseInlineArray(s)
  if (s === '{}') return {}
  return s
}

function parseInlineArray(s: string): unknown[] {
  const inner = s.slice(1, -1).trim()
  if (!inner) return []
  const items: string[] = []
  let depth = 0, cur = '', inQuote = false
  for (const ch of inner) {
    if (ch === '"') inQuote = !inQuote
    if (!inQuote) { if (ch === '[') depth++; if (ch === ']') depth-- }
    if (ch === ',' && depth === 0 && !inQuote) { items.push(cur); cur = '' } else { cur += ch }
  }
  if (cur.trim() !== '') items.push(cur)
  return items.map(it => parseScalar(it.trim()))
}

function parseYAML(text: string): Record<string, unknown> {
  const lines: { indent: number; text: string }[] = []
  for (const raw of text.split('\n')) {
    const trimmedEnd = raw.replace(/\s+$/, '')
    const trimmed = trimmedEnd.trim()
    if (!trimmed || trimmed.startsWith('#')) continue
    const indent = trimmedEnd.length - trimmedEnd.trimStart().length
    lines.push({ indent, text: stripInlineComment(trimmed) })
  }

  let pos = 0

  function parseBlock(indent: number): Record<string, unknown> | unknown[] {
    if (pos < lines.length && lines[pos].indent === indent && lines[pos].text.startsWith('- ')) {
      const arr: unknown[] = []
      while (pos < lines.length && lines[pos].indent === indent && lines[pos].text.startsWith('- ')) {
        const itemText = lines[pos].text.slice(2).trim()
        pos++
        if (itemText === '' && pos < lines.length && lines[pos].indent > indent) {
          arr.push(parseBlock(lines[pos].indent))
        } else {
          arr.push(parseScalar(itemText))
        }
      }
      return arr
    }

    const obj: Record<string, unknown> = {}
    while (pos < lines.length && lines[pos].indent === indent) {
      const line = lines[pos]
      const colonIdx = line.text.indexOf(':')
      if (colonIdx === -1) { pos++; continue }
      const key = line.text.slice(0, colonIdx).trim()
      const rest = line.text.slice(colonIdx + 1).trim()
      pos++
      if (rest === '') {
        obj[key] = (pos < lines.length && lines[pos].indent > indent) ? parseBlock(lines[pos].indent) : {}
      } else {
        obj[key] = parseScalar(rest)
      }
    }
    return obj
  }

  return parseBlock(0) as Record<string, unknown>
}


interface Meta {
  id: string; name: string; description: string
  instance_type: string; region: string; spot: boolean
  timeout_hours: number; epochs: number; batch_size: number
}

function toYAML(slots: Record<string, CardDef | null>, meta: Meta): string {
  const L: string[] = []
  L.push(`# sentinel> ${new Date().toISOString().slice(0, 10)}`)
  L.push(''); L.push('experiment:')
  L.push(`  id: ${meta.id || 'exp_custom'}`)
  L.push(`  name: "${meta.name}"`)
  if (meta.description) L.push(`  description: "${meta.description}"`)
  L.push(`  enabled: true`)
  L.push(''); L.push('infrastructure:')
  L.push(`  instance_type: ${meta.instance_type}`)
  L.push(`  region: ${meta.region}`)
  L.push(`  use_spot: ${meta.spot}`)
  L.push(`  timeout_hours: ${meta.timeout_hours}`)
  for (const slotId of SLOT_ORDER) {
    const card = slots[slotId]
    if (!card) continue
    const content = { ...card.content }
    if (slotId === 'train') { content.epochs = meta.epochs; content.batch_size = meta.batch_size }
    L.push(''); L.push(`${slotId}:  # ${card.filename}`)
    L.push(...serBlock(content))
  }
  return L.join('\n')
}


function PaletteItem({ card, dragging, inSlot, onDragStart }: {
  card: CardDef; dragging: boolean; inSlot: boolean
  onDragStart: (e: React.DragEvent, c: CardDef) => void
}) {
  const color = CATEGORY_COLOR[card.category] ?? 'rgba(255,255,255,0.3)'
  const [hov, setHov] = useState(false)
  return (
    <div
      draggable
      data-testid={`palette-item-${card.id}`}
      onDragStart={e => onDragStart(e, card)}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        display: 'flex', alignItems: 'center', gap: 9,
        padding: '7px 14px',
        cursor: dragging ? 'grabbing' : 'grab',
        userSelect: 'none',
        opacity: dragging ? 0.35 : inSlot ? 0.4 : 1,
        background: hov && !dragging ? 'rgba(255,255,255,0.03)' : 'transparent',
        borderLeft: `2px solid ${hov && !dragging ? color : 'transparent'}`,
        transition: 'background 0.12s, border-color 0.12s, opacity 0.12s',
      }}
    >
      <div style={{ color, flexShrink: 0, display: 'flex', alignItems: 'center' }}>
        {ICONS[card.category] ?? <div style={{ width: 13, height: 13, borderRadius: 1, background: color, opacity: 0.5 }} />}
      </div>

      <div style={{ minWidth: 0, flex: 1 }}>
        <div style={{ fontSize: '10.5px', fontWeight: 500, color: inSlot ? 'rgba(255,255,255,0.3)' : '#cdd5d0', lineHeight: 1, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
          {card.name}
        </div>
        {card.tags.length > 0 && (
          <div style={{ display: 'flex', gap: 4, marginTop: 3, flexWrap: 'wrap' }}>
            {card.tags.map(t => (
              <span key={t} style={{ fontSize: '8px', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace' }}>
                {t}
              </span>
            ))}
          </div>
        )}
      </div>

      {inSlot && <div style={{ width: 5, height: 5, borderRadius: '50%', background: color, flexShrink: 0, opacity: 0.7 }} />}
    </div>
  )
}


function Slot({ slotId, card, active, onDrop, onDragOver, onDragLeave, onClear }: {
  slotId: string; card: CardDef | null; active: boolean
  onDrop:      (e: React.DragEvent, s: string) => void
  onDragOver:  (e: React.DragEvent, s: string) => void
  onDragLeave: (e: React.DragEvent) => void
  onClear:     (s: string) => void
}) {
  const color    = card ? CATEGORY_COLOR[card.category] ?? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.12)'
  const required = REQUIRED.has(slotId)
  const label    = SLOT_LABEL[slotId] ?? slotId

  return (
    <div
      data-testid={`slot-${slotId}`}
      onDrop={e => onDrop(e, slotId)}
      onDragOver={e => onDragOver(e, slotId)}
      onDragLeave={onDragLeave}
      style={{
        position: 'relative',
        padding: card ? '9px 11px 10px' : '8px 10px',
        minHeight: 54,
        borderLeft: `2px solid ${card ? color : active ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.06)'}`,
        background: active
          ? 'rgba(255,255,255,0.025)'
          : card
          ? `${color}07`
          : 'transparent',
        transition: 'border-color 0.15s, background 0.18s',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: card ? 5 : 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{
            fontSize: '7.5px', fontWeight: 700, letterSpacing: '1px',
            textTransform: 'uppercase',
            color: card ? color : active ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.15)',
            transition: 'color 0.15s',
          }}>
            {label}
          </span>
          {required && !card && !active && (
            <span style={{ fontSize: '7px', color: 'rgba(200,112,112,0.6)', fontFamily: 'monospace' }}>·req</span>
          )}
        </div>
        {card && (
          <button
            onClick={() => onClear(slotId)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'rgba(255,255,255,0.18)', fontSize: '12px', padding: '0 1px', lineHeight: 1, transition: 'color 0.1s' }}
            onMouseEnter={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.5)')}
            onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.18)')}
          >
            ×
          </button>
        )}
      </div>

      {card && (
        <div>
          <div style={{ fontSize: '11px', fontWeight: 600, color: '#dde2e0', lineHeight: 1 }}>{card.name}</div>
          <div style={{ display: 'flex', gap: 5, marginTop: 5 }}>
            {card.tags.map(t => (
              <span key={t} style={{ fontSize: '8px', color, fontFamily: 'monospace', opacity: 0.7 }}>{t}</span>
            ))}
          </div>
        </div>
      )}

      {!card && (
        <div style={{
          fontSize: '9px', fontFamily: 'monospace', marginTop: 4,
          color: active ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.08)',
          transition: 'color 0.15s',
        }}>
          {active ? 'release to assign' : '—'}
        </div>
      )}
    </div>
  )
}


const F: React.CSSProperties = {
  width: '100%', padding: '6px 9px',
  fontSize: '10.5px', fontFamily: 'monospace',
  background: 'rgba(255,255,255,0.035)',
  border: '1px solid rgba(255,255,255,0.08)',
  color: '#cdd5d0', outline: 'none', boxSizing: 'border-box', borderRadius: 2,
  transition: 'border-color 0.15s',
}

const FL = ({ children }: { children: React.ReactNode }) => (
  <div style={{ fontSize: '8px', fontWeight: 700, letterSpacing: '0.9px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.2)', marginBottom: 5 }}>
    {children}
  </div>
)

function FocusInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  const [foc, setFoc] = useState(false)
  return <input {...props} onFocus={e => { setFoc(true); props.onFocus?.(e) }} onBlur={e => { setFoc(false); props.onBlur?.(e) }} style={{ ...F, ...props.style, borderColor: foc ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.08)' }} />
}

function FocusSelect(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  const [foc, setFoc] = useState(false)
  return <select {...props} onFocus={e => { setFoc(true); props.onFocus?.(e) }} onBlur={e => { setFoc(false); props.onBlur?.(e) }} style={{ ...F, cursor: 'pointer', ...props.style, borderColor: foc ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.08)' }} />
}

const Toggle = ({ value, onChange, label, sub }: { value: boolean; onChange: (v: boolean) => void; label: string; sub?: string }) => (
  <div onClick={() => onChange(!value)} style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', userSelect: 'none' }}>
    <div style={{ width: 28, height: 15, borderRadius: 8, flexShrink: 0, position: 'relative', transition: 'background 0.2s', background: value ? '#00d4a0' : 'rgba(255,255,255,0.1)' }}>
      <div style={{ position: 'absolute', top: 1.5, width: 12, height: 12, borderRadius: '50%', background: '#fff', transition: 'left 0.2s', left: value ? 14 : 1.5 }} />
    </div>
    <div>
      <div style={{ fontSize: '10.5px', color: 'rgba(255,255,255,0.5)' }}>{label}</div>
      {sub && <div style={{ fontSize: '8px', color: 'rgba(255,255,255,0.2)', fontFamily: 'monospace', marginTop: 2 }}>{sub}</div>}
    </div>
  </div>
)


const NEW_COMPONENT_CATEGORIES: { label: string; dir: string }[] = [
  { label: 'Backbone',     dir: 'backbones'     },
  { label: 'Optimizer',    dir: 'optimizers'    },
  { label: 'Loss',         dir: 'losses'        },
  { label: 'Augmentation', dir: 'augmentations' },
  { label: 'Priors',       dir: 'priors'        },
  { label: 'Heads',        dir: 'heads'         },
  { label: 'Training',     dir: 'train'         },
  { label: 'Evaluation',   dir: 'eval'          },
  { label: 'Checkpoint',   dir: 'checkpoint'    },
  { label: 'Runtime',      dir: 'runtime'       },
  { label: 'Sampler',      dir: 'samplers'      },
  { label: 'Logging',      dir: 'logging'       },
  { label: 'Export',       dir: 'export'        },
  { label: 'Dataset',      dir: 'data'          },
]

const NEW_COMPONENT_TEMPLATES: Record<string, string> = {
  backbones: `backbone:\n  name: mobilenetv2\n  width_mult: 1.0\n  output_layers: [C3, C4, C5]\n  pretrained: true\n  weights: imagenet\n  freeze: false\n  freeze_bn: false\n  grad_scale: 0.1\n`,
  optimizers: `optimizer:\n  name: sgd\n  lr: 0.001\n  momentum: 0.9\n  weight_decay: 0.0005\n  nesterov: true\n\nscheduler:\n  name: cosine_warmup\n  interval: step\n  base_lr: 0.001\n  min_lr: 0.00001\n  total_steps: 10000\n  warmup:\n    enabled: true\n    epochs: 5\n    start_factor: 0.1\n    end_factor: 1.0\n    mode: linear\n`,
  losses: `classification:\n  type: cross_entropy\n  weight: 1.0\n\nlocalization:\n  type: smooth_l1\n  weight: 1.0\n  beta: 1.0\n`,
  augmentations: `augmentation:\n  horizontal_flip: true\n  color_jitter:\n    brightness: 0.125\n    contrast: 0.5\n    saturation: 0.5\n    hue: 0.05\n  random_crop:\n    enabled: true\n    min_scale: 0.3\n    max_scale: 1.0\n  normalize:\n    mean: [0.485, 0.456, 0.406]\n    std: [0.229, 0.224, 0.225]\n`,
  priors: `image_size: [300, 300]\nfeature_map_sizes: [38, 19, 10, 5, 3, 1]\nmin_scale: 0.2\nmax_scale: 0.95\naspect_ratios: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]\nvariances: [0.1, 0.1, 0.2, 0.2]\nclip: true\n`,
  heads: `classification_head:\n  num_classes: 21\n  use_sigmoid: false\n  prior_prob: 0.01\n\nregression_head:\n  num_anchors: 6\n`,
  train: `epochs: 200\nbatch_size: 32\nnum_workers: 4\npin_memory: true\ngradient_clip: 10.0\nmixed_precision: true\n`,
  eval: `score_threshold: 0.35\nnms_iou_threshold: 0.45\nmax_detections: 200\niou_threshold: 0.5\n`,
  checkpoint: `keep_last_k: 5\nsave_every_n_epochs: 10\nsave_best: true\nmonitor_metric: mAP\n`,
  runtime: `num_workers: 4\npin_memory: true\ndeterministic: false\nseed: 42\n`,
  samplers: `name: hard_negative_mining\nneg_pos_ratio: 3\nmin_negatives: 0\n`,
  logging: `log_every_n_steps: 50\nlog_images: false\nlog_gradients: false\n`,
  export: `opset: 17\ninput_name: input\noutput_names: [boxes, scores]\n`,
  data: `dataset: voc\nroot: datasets/VOCdevkit\nyear: "2012"\nsplit: trainval\nnum_classes: 21\ninput_size: [300, 300]\n`,
}

function NewComponentModal({ onClose, onSaved }: { onClose: () => void; onSaved: () => void }) {
  const [category, setCategory] = useState(NEW_COMPONENT_CATEGORIES[0].dir)
  const [name,     setName]     = useState('')
  const [content,  setContent]  = useState(NEW_COMPONENT_TEMPLATES[NEW_COMPONENT_CATEGORIES[0].dir] ?? '')
  const [saving,   setSaving]   = useState(false)
  const [result,   setResult]   = useState<{ ok: boolean; msg: string } | null>(null)

  const handleCategoryChange = (dir: string) => {
    setCategory(dir); setContent(NEW_COMPONENT_TEMPLATES[dir] ?? ''); setResult(null)
  }

  const handleSave = async () => {
    if (!name.trim()) { setResult({ ok: false, msg: 'Name is required.' }); return }
    setSaving(true); setResult(null)
    try {
      const res = await saveConfig({ category, name: name.trim(), content_yaml: content })
      setResult({ ok: true, msg: `Saved → ${res.path}` })
      onSaved()
    } catch (e) {
      setResult({ ok: false, msg: String(e) })
    } finally { setSaving(false) }
  }

  return (
    <div style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ background: 'rgba(6,12,9,0.98)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, padding: '22px 24px', maxWidth: 460, width: '92%', position: 'relative' }}>
        {(['tl','tr','bl','br'] as const).map(c => {
          const t = c.startsWith('t'), l = c.endsWith('l')
          const d = t ? (l ? 'M14 1 L1 1 L1 14' : 'M0 1 L13 1 L13 14') : (l ? 'M14 13 L1 13 L1 0' : 'M0 13 L13 13 L13 0')
          return <svg key={c} aria-hidden width="14" height="14" style={{ position: 'absolute', top: t?0:undefined, bottom: !t?0:undefined, left: l?0:undefined, right: !l?0:undefined }}>
            <path d={d} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1"/>
          </svg>
        })}

        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <span style={{ fontSize: '13px', fontWeight: 700, color: '#cdd5d0' }}>New Component</span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.3)', cursor: 'pointer', fontSize: '15px', padding: 0 }}>×</button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '150px 1fr', gap: 10, marginBottom: 12 }}>
          <div>
            <FL>Category</FL>
            <FocusSelect value={category} onChange={e => handleCategoryChange(e.target.value)}>
              {NEW_COMPONENT_CATEGORIES.map(c => <option key={c.dir} value={c.dir}>{c.label}</option>)}
            </FocusSelect>
          </div>
          <div>
            <FL>File name</FL>
            <FocusInput value={name} placeholder="my_component" onChange={e => { setName(e.target.value); setResult(null) }} />
          </div>
        </div>

        <div style={{ fontSize: '9px', fontFamily: 'monospace', color: 'rgba(255,255,255,0.2)', marginBottom: 10 }}>
          → configs/{category === 'data' ? 'data' : `base/${category}`}/{name.trim().toLowerCase().replace(/[^a-z0-9_\-]/g, '_') || '<name>'}.yaml
        </div>

        <FL>YAML content</FL>
        <textarea
          value={content}
          onChange={e => setContent(e.target.value)}
          spellCheck={false}
          rows={12}
          style={{ ...F, fontSize: '10.5px', lineHeight: 1.7, resize: 'vertical', whiteSpace: 'pre' }}
        />

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 14 }}>
          <button
            onClick={handleSave} disabled={saving}
            style={{ padding: '8px 20px', borderRadius: 2, border: '1px solid rgba(0,212,160,0.35)', background: 'rgba(0,212,160,0.1)', color: '#00d4a0', fontFamily: 'inherit', fontSize: '10.5px', fontWeight: 700, cursor: saving ? 'not-allowed' : 'pointer', opacity: saving ? 0.5 : 1, transition: 'opacity 0.15s' }}
          >
            {saving ? 'Saving…' : 'Save Component'}
          </button>
          {result && (
            <span style={{ fontSize: '10px', fontFamily: 'monospace', color: result.ok ? '#00d4a0' : '#c87070' }}>{result.msg}</span>
          )}
        </div>
      </div>
    </div>
  )
}


const DEFAULT_META: Meta = {
  id: 'exp005_custom', name: 'Custom Experiment', description: '',
  instance_type: 'g4dn.xlarge', region: 'us-east-1', spot: true,
  timeout_hours: 12, epochs: 200, batch_size: 32,
}

export const ConfigView = () => {
  const [catalog,    setCatalog]    = useState<CardDef[]>([])
  const [libLoading, setLibLoading] = useState(true)
  const [libError,   setLibError]   = useState<string | null>(null)
  const [slots,      setSlots]      = useState<Record<string, CardDef | null>>({})
  const [meta,       setMeta]       = useState<Meta>({ ...DEFAULT_META })
  const [activeSlot, setActiveSlot] = useState<string | null>(null)
  const [dragCard,   setDragCard]   = useState<CardDef | null>(null)
  const [copied,     setCopied]     = useState(false)
  const [modal,      setModal]      = useState(false)
  const [busy,       setBusy]       = useState(false)
  const [launched,   setLaunched]   = useState<{ fingerprint: string } | null>(null)
  const [launchError,setLaunchError]= useState<string | null>(null)
  const [newCompOpen, setNewCompOpen] = useState(false)

  const generatedYaml = useMemo(() => toYAML(slots, meta), [slots, meta])
  const [yamlText, setYamlText] = useState(generatedYaml)
  const [yamlDirty, setYamlDirty] = useState(false)

  useEffect(() => {
    if (!yamlDirty) setYamlText(generatedYaml)
  }, [generatedYaml, yamlDirty])

  const handleYamlChange = (val: string) => { setYamlText(val); setYamlDirty(true) }
  const handleSyncFromBuilder = () => { setYamlText(generatedYaml); setYamlDirty(false) }

  const loadLibrary = () => {
    setLibLoading(true); setLibError(null)
    fetchConfigLibrary()
      .then((lib: ConfigLibrary) => setCatalog(libraryToCards(lib)))
      .catch(err => setLibError(String(err)))
      .finally(() => setLibLoading(false))
  }
  useEffect(() => { loadLibrary() }, [])

  const handleRefresh = () => {
    setLibLoading(true)
    refreshConfigLibrary().then(() => loadLibrary()).catch(err => { setLibError(String(err)); setLibLoading(false) })
  }

  const grouped = SLOT_ORDER.reduce<Record<string, CardDef[]>>((acc, cat) => {
    acc[cat] = catalog.filter(c => c.category === cat); return acc
  }, {})

  const usedIds = new Set(Object.values(slots).filter(Boolean).map(c => c!.id))

  const onDragStart = (e: React.DragEvent, card: CardDef) => {
    e.dataTransfer.setData('text/plain', JSON.stringify(card))
    e.dataTransfer.effectAllowed = 'copy'
    setDragCard(card)
  }
  const onDragEnd  = () => setDragCard(null)
  const onDragOver = (e: React.DragEvent, slotId: string) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; setActiveSlot(slotId) }
  const onDrop     = (e: React.DragEvent, slotId: string) => {
    e.preventDefault()
    try {
      const card: CardDef = JSON.parse(e.dataTransfer.getData('text/plain'))
      if (card.category !== slotId) return
      setSlots(s => ({ ...s, [slotId]: card }))
      if (slotId === 'train') {
        const ep = card.content.epochs;     if (typeof ep === 'number') setMeta(m => ({ ...m, epochs: ep }))
        const bs = card.content.batch_size; if (typeof bs === 'number') setMeta(m => ({ ...m, batch_size: bs }))
      }
    } catch {}
    setActiveSlot(null); setDragCard(null)
  }
  const onDragLeave = (e: React.DragEvent) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setActiveSlot(null) }
  const onClear     = (slotId: string) => setSlots(s => ({ ...s, [slotId]: null }))
  const setM        = <K extends keyof Meta>(k: K) => (v: Meta[K]) => setMeta(m => ({ ...m, [k]: v }))

  const [yamlW, setYamlW]     = useState(260)
  const resizing              = useRef(false)
  const startX                = useRef(0)
  const startW                = useRef(0)

  const onResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    resizing.current = true
    startX.current   = e.clientX
    startW.current   = yamlW

    const onMove = (ev: MouseEvent) => {
      if (!resizing.current) return
      const delta = startX.current - ev.clientX
      setYamlW(Math.max(180, Math.min(600, startW.current + delta)))
    }
    const onUp = () => {
      resizing.current = false
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [yamlW])

  const handleCopy = () => { navigator.clipboard?.writeText(yamlText); setCopied(true); setTimeout(() => setCopied(false), 2000) }
  const missingRequired = SLOT_ORDER.filter(s => REQUIRED.has(s) && !slots[s])

  const [parsedPreview, parseError] = useMemo(() => {
    try { return [parseYAML(yamlText), null] as const }
    catch (e) { return [null, String(e)] as const }
  }, [yamlText])

  const handleLaunch = async () => {
    setBusy(true); setLaunchError(null)
    try {
      const parsed = parseYAML(yamlText)
      const res = await registerExperiment({
        config: parsed as unknown as ExperimentConfig,
        task_type: 'detector',
        git_commit: null,
      })
      await launchTraining({ experiment_id: res.experiment_id, fingerprint: res.fingerprint })
      setLaunched({ fingerprint: res.fingerprint }); setModal(false)
    } catch (e) { setLaunchError(String(e)) }
    finally { setBusy(false) }
  }

  return (
    <div
      data-testid="config-view"
      style={{
        height: '100%', display: 'grid',
        gridTemplateColumns: `220px 1fr 4px ${yamlW}px`,
        overflow: 'hidden',
        background: `
          radial-gradient(ellipse 55% 45% at 15% 25%, rgba(0,180,120,0.045) 0%, transparent 65%),
          radial-gradient(ellipse 45% 50% at 85% 75%, rgba(124,158,245,0.035) 0%, transparent 65%),
          var(--bg-primary)
        `,
      }}
    >

      <div style={{ borderRight: '1px solid rgba(255,255,255,0.05)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        <div style={{ padding: '20px 14px 14px', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 2 }}>
            <div style={{ fontSize: '18px', fontWeight: 700, color: '#cdd5d0', letterSpacing: '-0.4px', lineHeight: 1 }}>
              Palette
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <button
                onClick={() => setNewCompOpen(true)}
                title="New component"
                style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '3px 0', display: 'flex', alignItems: 'center', color: 'rgba(255,255,255,0.25)', transition: 'color 0.15s' }}
                onMouseEnter={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.55)')}
                onMouseLeave={e => (e.currentTarget.style.color = 'rgba(255,255,255,0.25)')}
              >
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round"/>
                </svg>
              </button>
              <button
                onClick={handleRefresh} disabled={libLoading}
                style={{ background: 'none', border: 'none', cursor: libLoading ? 'default' : 'pointer', padding: '3px 0', display: 'flex', alignItems: 'center', gap: 5 }}
              >
                {libLoading
                  ? <Spinner size={12} />
                  : <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ color: 'rgba(255,255,255,0.25)', transition: 'color 0.15s' }}
                      onMouseEnter={e => ((e.currentTarget as SVGElement).style.color = 'rgba(255,255,255,0.55)')}
                      onMouseLeave={e => ((e.currentTarget as SVGElement).style.color = 'rgba(255,255,255,0.25)')}>
                      <path d="M10 6A4 4 0 1 1 9 3.3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" fill="none"/>
                      <path d="M9 1.5 L9 3.5 L11 3.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                    </svg>
                }
              </button>
            </div>
          </div>
          <div style={{ fontSize: '9px', color: 'rgba(255,255,255,0.15)', fontFamily: 'monospace', marginTop: 6 }}>
            {libError
              ? <span style={{ color: '#c87070' }}>error loading</span>
              : libLoading
              ? 'syncing from S3…'
              : `${catalog.length} configs`}
          </div>
        </div>

        <div style={{ height: '1px', background: 'rgba(255,255,255,0.05)', flexShrink: 0 }} />

        <div style={{ flex: 1, overflowY: 'auto' }} onDragEnd={onDragEnd}>
          {libLoading && catalog.length === 0 ? (
            <div style={{ padding: '30px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {[80, 65, 80, 65, 80].map((w, i) => (
                <div key={i} style={{ height: 28, background: 'rgba(255,255,255,0.04)', borderRadius: 2, width: `${w}%`, animation: 'cfg-pulse 1.4s ease-in-out infinite', animationDelay: `${i * 0.12}s` }} />
              ))}
              <style>{`@keyframes cfg-pulse { 0%,100%{opacity:0.5} 50%{opacity:1} }`}</style>
            </div>
          ) : (
            SLOT_ORDER.map(cat => {
              const cards = grouped[cat]
              if (!cards?.length) return null
              const color = CATEGORY_COLOR[cat] ?? 'rgba(255,255,255,0.3)'
              return (
                <div key={cat}>
                  <div style={{ padding: '10px 14px 4px', display: 'flex', alignItems: 'center', gap: 7 }}>
                    <div style={{ color, display: 'flex', alignItems: 'center' }}>{ICONS[cat]}</div>
                    <span style={{ fontSize: '8px', fontWeight: 700, letterSpacing: '1px', textTransform: 'uppercase', color }}>
                      {SLOT_LABEL[cat] ?? cat}
                    </span>
                  </div>
                  {cards.map(card => (
                    <PaletteItem
                      key={card.id}
                      card={card}
                      dragging={dragCard?.id === card.id}
                      inSlot={usedIds.has(card.id)}
                      onDragStart={onDragStart}
                    />
                  ))}
                  <div style={{ height: '1px', background: 'rgba(255,255,255,0.04)', margin: '4px 0' }} />
                </div>
              )
            })
          )}
        </div>
      </div>

      <div style={{ overflowY: 'auto', padding: '20px 22px 28px', display: 'flex', flexDirection: 'column', gap: 24 }}>

        <div>
          <SectionLabel>Experiment</SectionLabel>
          <Glass style={{ padding: '14px 16px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div><FL>ID</FL><FocusInput value={meta.id} onChange={e => setM('id')(e.target.value)} placeholder="exp005_custom" /></div>
            <div><FL>Name</FL><FocusInput value={meta.name} onChange={e => setM('name')(e.target.value)} /></div>
            <div style={{ gridColumn: '1/-1' }}><FL>Description</FL><FocusInput value={meta.description} onChange={e => setM('description')(e.target.value)} placeholder="optional" /></div>
          </Glass>
        </div>

        <div>
          <SectionLabel>Infrastructure</SectionLabel>
          <Glass style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 68px 68px', gap: 10 }}>
              <div>
                <FL>Instance</FL>
                <FocusSelect value={meta.instance_type} onChange={e => setM('instance_type')(e.target.value)}>
                  {['g4dn.xlarge','g4dn.2xlarge','g5.xlarge','g5.2xlarge','p3.2xlarge','p3.8xlarge'].map(t => <option key={t}>{t}</option>)}
                </FocusSelect>
              </div>
              <div>
                <FL>Region</FL>
                <FocusSelect value={meta.region} onChange={e => setM('region')(e.target.value)}>
                  {['us-east-1','us-east-2','us-west-2','eu-west-1','ap-southeast-1','ap-northeast-1'].map(r => <option key={r}>{r}</option>)}
                </FocusSelect>
              </div>
              <div><FL>Epochs</FL><FocusInput type="number" value={meta.epochs} onChange={e => setM('epochs')(+e.target.value)} min={1} /></div>
              <div><FL>Batch</FL><FocusInput type="number" value={meta.batch_size} onChange={e => setM('batch_size')(+e.target.value)} min={1} /></div>
            </div>
            <Toggle value={meta.spot} onChange={setM('spot')} label="Spot instance" sub="~70% cost reduction" />
          </Glass>
        </div>

        <div>
          <SectionLabel>Composition</SectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
            {SLOT_ORDER.map(slotId => (
              <Slot
                key={slotId} slotId={slotId} card={slots[slotId] ?? null}
                active={activeSlot === slotId}
                onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave} onClear={onClear}
              />
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button
            onClick={() => missingRequired.length === 0 && setModal(true)}
            disabled={missingRequired.length > 0}
            style={{
              padding: '8px 22px', borderRadius: 2, fontFamily: 'inherit',
              fontSize: '10.5px', fontWeight: 700,
              cursor: missingRequired.length ? 'not-allowed' : 'pointer',
              background: missingRequired.length ? 'transparent' : 'rgba(0,212,160,0.1)',
              border: `1px solid ${missingRequired.length ? 'rgba(255,255,255,0.07)' : 'rgba(0,212,160,0.3)'}`,
              color: missingRequired.length ? 'rgba(255,255,255,0.15)' : '#00d4a0',
              transition: 'all 0.15s',
            }}
          >
            Launch
          </button>
          {missingRequired.length > 0 && (
            <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.15)', fontFamily: 'monospace' }}>
              needs: {missingRequired.map(s => SLOT_LABEL[s]).join(', ')}
            </span>
          )}
        </div>
      </div>

      <div
        onMouseDown={onResizeStart}
        style={{
          cursor: 'col-resize', zIndex: 10,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'transparent', transition: 'background 0.15s',
        }}
        onMouseEnter={e => (e.currentTarget.style.background = 'rgba(255,255,255,0.06)')}
        onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
      >
        <div style={{ width: 1, height: '40%', background: 'rgba(255,255,255,0.1)', borderRadius: 1 }} />
      </div>

      <div style={{ borderLeft: 'none', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '14px 16px 10px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0, gap: 8 }}>
          <span style={{ fontSize: '8px', fontWeight: 700, letterSpacing: '1.4px', textTransform: 'uppercase', color: 'rgba(255,255,255,0.18)', display: 'flex', alignItems: 'center', gap: 6 }}>
            YAML
            {yamlDirty && <span style={{ width: 4, height: 4, borderRadius: '50%', background: '#e8924a' }} title="edited" />}
          </span>
          <div style={{ display: 'flex', gap: 6 }}>
            {yamlDirty && (
              <button
                onClick={handleSyncFromBuilder}
                style={{ padding: '3px 9px', borderRadius: 2, fontFamily: 'monospace', fontSize: '8.5px', cursor: 'pointer', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.3)', transition: 'all 0.2s' }}
              >
                sync from builder
              </button>
            )}
            <button
              onClick={handleCopy}
              style={{
                padding: '3px 10px', borderRadius: 2, fontFamily: 'monospace', fontSize: '8.5px', cursor: 'pointer',
                background: copied ? 'rgba(0,212,160,0.1)' : 'rgba(255,255,255,0.04)',
                border: `1px solid ${copied ? 'rgba(0,212,160,0.3)' : 'rgba(255,255,255,0.08)'}`,
                color: copied ? '#00d4a0' : 'rgba(255,255,255,0.28)',
                transition: 'all 0.2s',
              }}
            >
              {copied ? '✓ copied' : 'copy'}
            </button>
          </div>
        </div>

        <div
          data-testid="yaml-preview"
          style={{ flex: 1, overflow: 'auto', display: 'flex', fontFamily: 'monospace', fontSize: '9.5px', lineHeight: 1.9 }}
        >
          <div style={{ width: 36, flexShrink: 0, padding: '12px 0', textAlign: 'right', userSelect: 'none', pointerEvents: 'none' }}>
            {yamlText.split('\n').map((_, i) => (
              <div key={i} style={{ paddingRight: 12, color: 'rgba(255,255,255,0.1)', fontSize: '8.5px', lineHeight: '1.9em' }}>
                {i + 1}
              </div>
            ))}
          </div>
          <textarea
            data-testid="yaml-editor"
            value={yamlText}
            onChange={e => handleYamlChange(e.target.value)}
            spellCheck={false}
            style={{
              flex: 1, padding: '12px 16px 12px 0', resize: 'none', outline: 'none', border: 'none',
              background: 'transparent', color: 'rgba(255,255,255,0.55)',
              fontFamily: 'monospace', fontSize: '9.5px', lineHeight: 1.9,
              minHeight: '100%', boxSizing: 'border-box', whiteSpace: 'pre',
            }}
          />
        </div>

        {parseError && (
          <div style={{ padding: '8px 14px', borderTop: '1px solid rgba(200,112,112,0.25)', background: 'rgba(200,112,112,0.06)', fontSize: '9.5px', fontFamily: 'monospace', color: '#c87070', flexShrink: 0 }}>
            parse error: {parseError}
          </div>
        )}
      </div>

      {modal && (
        <div data-testid="launch-modal" style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: 'rgba(6,12,9,0.98)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 4, padding: '24px 26px', maxWidth: 360, width: '90%', position: 'relative' }}>
            {(['tl','tr','bl','br'] as const).map(c => {
              const t = c.startsWith('t'), l = c.endsWith('l')
              const d = t ? (l ? 'M14 1 L1 1 L1 14' : 'M0 1 L13 1 L13 14') : (l ? 'M14 13 L1 13 L1 0' : 'M0 13 L13 13 L13 0')
              return <svg key={c} aria-hidden width="14" height="14" style={{ position: 'absolute', top: t?0:undefined, bottom: !t?0:undefined, left: l?0:undefined, right: !l?0:undefined }}>
                <path d={d} fill="none" stroke="rgba(255,255,255,0.18)" strokeWidth="1"/>
              </svg>
            })}
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#cdd5d0', marginBottom: 12 }}>Launch Experiment</div>
            {parseError ? (
              <div style={{ fontSize: '10px', color: '#c87070', fontFamily: 'monospace', lineHeight: 1.8, marginBottom: 18 }}>
                YAML parse error — fix the editor before launching:<br/>{parseError}
              </div>
            ) : (
              <div style={{ fontSize: '10px', color: 'rgba(255,255,255,0.28)', fontFamily: 'monospace', lineHeight: 1.8, marginBottom: 18 }}>
                {String((parsedPreview?.infrastructure as any)?.instance_type ?? meta.instance_type)} · {String((parsedPreview?.infrastructure as any)?.region ?? meta.region)}{(parsedPreview?.infrastructure as any)?.use_spot ? ' · spot' : ''}<br/>
                {String((parsedPreview?.train as any)?.epochs ?? meta.epochs)} epochs · batch {String((parsedPreview?.train as any)?.batch_size ?? meta.batch_size)}<br/>
                <span style={{ color: 'rgba(255,255,255,0.15)' }}>{String((parsedPreview?.experiment as any)?.id ?? meta.id)}</span>
              </div>
            )}
            {launchError && (
              <div style={{ fontSize: '10px', color: '#c87070', fontFamily: 'monospace', marginBottom: 14 }}>{launchError}</div>
            )}
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={() => setModal(false)} style={{ flex: 1, padding: '8px', borderRadius: 2, border: '1px solid rgba(255,255,255,0.08)', background: 'transparent', color: 'rgba(255,255,255,0.3)', fontFamily: 'inherit', fontSize: '10.5px', cursor: 'pointer' }}>Cancel</button>
              <button data-testid="confirm-launch" onClick={handleLaunch} disabled={busy || !!parseError} style={{ flex: 2, padding: '8px', borderRadius: 2, border: '1px solid rgba(0,212,160,0.35)', background: 'rgba(0,212,160,0.1)', color: '#00d4a0', fontFamily: 'inherit', fontSize: '10.5px', fontWeight: 700, cursor: (busy || parseError) ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, opacity: parseError ? 0.4 : 1 }}>
                {busy ? <><Spinner size={12} /> launching…</> : 'Confirm Launch'}
              </button>
            </div>
          </div>
        </div>
      )}

      {newCompOpen && (
        <NewComponentModal
          onClose={() => setNewCompOpen(false)}
          onSaved={() => { handleRefresh(); }}
        />
      )}

      {launched && (
        <div data-testid="launch-banner" style={{ position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', zIndex: 300, padding: '10px 16px', borderRadius: 3, background: 'rgba(6,12,9,0.97)', border: '1px solid rgba(0,212,160,0.25)', display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 6, height: 6, borderRadius: 1, background: '#00d4a0', flexShrink: 0 }} />
          <div>
            <div style={{ fontSize: '10.5px', fontWeight: 700, color: '#cdd5d0' }}>Training launched</div>
            <div data-testid="launch-fingerprint" style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.25)', fontFamily: 'monospace', marginTop: 2 }}>fp · {launched.fingerprint}</div>
          </div>
          <button onClick={() => setLaunched(null)} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.2)', cursor: 'pointer', fontSize: '13px', padding: 0, marginLeft: 6 }}>×</button>
        </div>
      )}
    </div>
  )
}
