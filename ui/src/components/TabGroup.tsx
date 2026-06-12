import React from 'react'

const pillBase: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: '6px',
  padding: '7px 14px', borderRadius: '999px', fontSize: '12.5px',
  fontWeight: 500, cursor: 'pointer', transition: 'all 0.25s ease',
  whiteSpace: 'nowrap', fontFamily: 'inherit', outline: 'none',
  lineHeight: 1.2,
}

interface Props {
  tabs: string[]
  active: string
  onChange: (tab: string) => void
}

export const TabGroup = ({ tabs, active, onChange }: Props) => (
  <div style={{
    display: 'flex', gap: 2, padding: 3, borderRadius: 999,
    background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
  }}>
    {tabs.map(t => (
      <button key={t} onClick={() => onChange(t)} style={{
        ...pillBase, padding: '6px 16px', fontSize: '12px',
        border: 'none',
        background: t === active ? '#fff' : 'transparent',
        color: t === active ? '#0a0e0d' : 'var(--text-secondary)',
      }}>{t}</button>
    ))}
  </div>
)
