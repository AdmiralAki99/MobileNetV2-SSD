import { ShieldCheck, DangerTriangle, AlertCircle } from './icons'

type AlertType = 'protected' | 'danger' | 'alert'

interface Config {
  icon: React.ReactNode
  bg: string
  border: string
  glow: string
}

const ALERT_CONFIG: Record<AlertType, Config> = {
  protected: { icon: <ShieldCheck size={18}/>, bg: 'rgba(101,193,106,0.15)', border: 'rgba(101,193,106,0.3)', glow: 'rgba(101,193,106,0.2)' },
  danger:    { icon: <DangerTriangle size={18}/>, bg: 'rgba(232,72,85,0.15)',  border: 'rgba(232,72,85,0.3)',  glow: 'rgba(232,72,85,0.2)'  },
  alert:     { icon: <AlertCircle size={18}/>,   bg: 'rgba(232,133,72,0.15)', border: 'rgba(232,133,72,0.3)', glow: 'rgba(232,133,72,0.2)' },
}

interface Props {
  type?: AlertType
  style?: React.CSSProperties
  animate?: boolean
}

export const AlertBadge = ({ type = 'protected', style: s, animate = true }: Props) => {
  const config = ALERT_CONFIG[type]
  return (
    <div
      role="status"
      aria-label={type}
      style={{
        width: 34, height: 34, borderRadius: '50%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: config.bg,
        border: `1px solid ${config.border}`,
        boxShadow: `0 0 12px ${config.glow}`,
        animation: animate ? 'pulse 3s ease-in-out infinite' : 'none',
        cursor: 'pointer', transition: 'transform 0.2s',
        ...s,
      }}
    >
      {config.icon}
    </div>
  )
}
