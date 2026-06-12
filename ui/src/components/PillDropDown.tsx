import { useState, useRef, useEffect } from 'react'
import { ChevronDown } from './icons'

const pillBase: React.CSSProperties = {
  display: 'inline-flex', alignItems: 'center', gap: '6px',
  padding: '7px 14px', borderRadius: '999px', fontSize: '12.5px',
  fontWeight: 500, cursor: 'pointer', transition: 'all 0.25s ease',
  whiteSpace: 'nowrap', fontFamily: 'inherit', outline: 'none',
  lineHeight: 1.2,
}

interface Props {
  label?: string
  value?: string
  options?: string[]
  style?: React.CSSProperties
}

export const PillDropdown = ({ label, value, options = [], style: s }: Props) => {
  const [open, setOpen] = useState(false)
  const [val, setVal] = useState(value)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [])

  return (
    <div ref={ref} style={{ position: 'relative', display: 'inline-flex', ...s }}>
      <button onClick={() => setOpen(!open)} style={{
        ...pillBase,
        border: '1px solid var(--border-subtle)',
        background: 'var(--bg-pill)', color: 'var(--text-secondary)',
      }}>
        {label && <span style={{ color: 'var(--text-tertiary)', fontSize: '11px', marginRight: 2 }}>{label}</span>}
        {val} <ChevronDown />
      </button>
      {open && (
        <div role="listbox" style={{
          position: 'absolute', top: '110%', left: 0, zIndex: 50,
          background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)',
          borderRadius: 12, padding: '4px 0', minWidth: 140,
          boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
        }}>
          {options.map(o => (
            <div key={o} role="option" aria-selected={o === val} onClick={() => { setVal(o); setOpen(false) }} style={{
              padding: '8px 14px', fontSize: '12.5px', cursor: 'pointer',
              color: o === val ? 'var(--accent)' : 'var(--text-secondary)',
              background: o === val ? 'rgba(0,212,160,0.06)' : 'transparent',
            }}>
              {o}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
