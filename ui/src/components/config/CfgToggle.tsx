interface Props {
  label: string
  value: boolean
  onChange: (v: boolean) => void
  sub?: string
}

export const CfgToggle = ({ label, value, onChange, sub }: Props) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '3px 0' }}>
    <div>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</div>
      {sub && <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginTop: 1 }}>{sub}</div>}
    </div>
    <div
      role="switch"
      aria-checked={value}
      aria-label={label}
      onClick={() => onChange(!value)}
      style={{
        width: 36, height: 20, borderRadius: 999, cursor: 'pointer', flexShrink: 0, position: 'relative',
        background: value ? 'var(--accent)' : 'rgba(255,255,255,0.1)',
        transition: 'background 0.2s',
      }}
    >
      <div style={{
        position: 'absolute', top: 3, left: value ? 17 : 3,
        width: 14, height: 14, borderRadius: '50%', background: '#fff',
        transition: 'left 0.18s cubic-bezier(0.34,1.56,0.64,1)',
      }} />
    </div>
  </div>
)
