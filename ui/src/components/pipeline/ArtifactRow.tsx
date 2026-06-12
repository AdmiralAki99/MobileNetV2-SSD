import { ShieldCheck, AlertCircle } from '../icons'

interface Props {
  label: string
  ok: boolean
}

export const ArtifactRow = ({ label, ok }: Props) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
    {ok
      ? <ShieldCheck size={13} />
      : <AlertCircle size={13} color="var(--text-tertiary)" />
    }
    <span style={{ fontSize: '11.5px', color: ok ? 'var(--text-primary)' : 'var(--text-tertiary)' }}>
      {label}
    </span>
  </div>
)
