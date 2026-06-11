import { STATUS_COLOR, type Status } from '../constants/status'

interface Props {
  status?: Status | string
}

export const StatusBadge = ({ status }: Props) => {
  const color = STATUS_COLOR[status as Status] ?? '#555'
  return (
    <span style={{
      fontSize: '10px', fontWeight: 600, letterSpacing: '0.5px',
      padding: '3px 8px', borderRadius: 999, textTransform: 'uppercase',
      background: `${color}22`,
      color,
      border: `1px solid ${color}44`,
    }}>
      {status ?? 'unknown'}
    </span>
  )
}