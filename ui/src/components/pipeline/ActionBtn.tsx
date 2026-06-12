interface Props {
  label: string
  color: string
  disabled?: boolean
  busy?: boolean
  onClick?: () => void
}

export const ActionBtn = ({ label, color, disabled = false, busy = false, onClick }: Props) => (
  <button
    onClick={!disabled && !busy ? onClick : undefined}
    disabled={disabled || busy}
    style={{
      width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '7px 10px', borderRadius: 8, fontSize: '11.5px', fontWeight: 600,
      cursor: (disabled || busy) ? 'not-allowed' : 'pointer', fontFamily: 'inherit',
      transition: 'all 0.2s ease', outline: 'none',
      border: `1px solid ${color}44`,
      background: `${color}14`,
      color: (disabled || busy) ? 'var(--text-tertiary)' : color,
      opacity: (disabled || busy) ? 0.45 : 1,
    }}
  >
    {busy ? '…' : label}
  </button>
)
