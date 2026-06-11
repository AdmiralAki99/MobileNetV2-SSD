import React from 'react'

interface Props {
  children: React.ReactNode
  active?: boolean
  onClick?: () => void
  style?: React.CSSProperties
}

const pillBase: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: '6px',
  padding: '7px 14px', borderRadius: '999px', fontSize: '12.5px',
  fontWeight: 500, cursor: 'pointer', transition: 'all 0.25s ease',
  whiteSpace: 'nowrap', fontFamily: 'inherit', outline: 'none',
  lineHeight: 1.2,
}

export const PillButton = ({ children, active, onClick, style: s }: Props) => (
  <button onClick={onClick} style={{
    ...pillBase,
    border: active ? '1px solid rgba(255,255,255,0.15)' : '1px solid var(--border-subtle)',
    background: active ? '#fff' : 'var(--bg-pill)',
    color: active ? '#0a0e0d' : 'var(--text-secondary)',
    ...s,
  }}>
    {children}
  </button>
)
