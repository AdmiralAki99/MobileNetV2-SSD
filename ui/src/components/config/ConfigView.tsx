import { useState, useMemo } from 'react'
import { PillButton } from '../PillButton'
import { CfgToggle } from './CfgToggle'
import { launchTraining } from '../../api/client'
import { DEFAULT_CFG, toYAML, type Config } from './configTypes'

const SEC_LABEL: React.CSSProperties = {
  fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)',
  letterSpacing: '0.5px', textTransform: 'uppercase',
}

const inputStyle = (focused: boolean): React.CSSProperties => ({
  width: '100%', padding: '7px 11px', borderRadius: 8,
  fontSize: '12px', fontFamily: 'monospace',
  background: 'var(--bg-pill)',
  border: `1px solid ${focused ? 'var(--accent)' : 'var(--border-subtle)'}`,
  color: 'var(--text-primary)', outline: 'none',
  transition: 'border-color 0.2s', boxSizing: 'border-box' as const,
})

const CfgInput = ({ label, value, onChange, type = 'text', placeholder = '' }: { label?: string; value: string | number; onChange: (v: string | number) => void; type?: string; placeholder?: string }) => {
  const [focused, setFocused] = useState(false)
  return (
    <div>
      {label && <div style={{ ...SEC_LABEL, marginBottom: 6 }}>{label}</div>}
      <input
        type={type} value={value} placeholder={placeholder}
        onChange={e => onChange(type === 'number' ? +e.target.value : e.target.value)}
        onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        style={inputStyle(focused)}
      />
    </div>
  )
}

const ChipGroup = ({ label, options, value, onChange }: { label?: string; options: { label: string; value: string | number }[]; value: string | number; onChange: (v: string | number) => void }) => (
  <div>
    {label && <div style={{ ...SEC_LABEL, marginBottom: 6 }}>{label}</div>}
    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
      {options.map(opt => (
        <PillButton key={String(opt.value)} active={value === opt.value} onClick={() => onChange(opt.value)}
          style={{ padding: '5px 12px', fontSize: '11.5px' }}>
          {opt.label}
        </PillButton>
      ))}
    </div>
  </div>
)

export const ConfigView = () => {
  const [cfg, setCfg]       = useState<Config>({ ...DEFAULT_CFG })
  const [modal, setModal]   = useState(false)
  const [busy, setBusy]     = useState(false)
  const [launched, setLaunched] = useState<{ fingerprint: string } | null>(null)
  const [copied, setCopied] = useState(false)

  const set = (key: keyof Config) => (val: unknown) => setCfg(c => ({ ...c, [key]: val }))
  const yaml = useMemo(() => toYAML(cfg), [cfg])

  const handleCopy = () => {
    navigator.clipboard?.writeText(yaml)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleLaunch = async () => {
    setBusy(true)
    try {
      const fp = Math.random().toString(16).slice(2, 10)
      await launchTraining({ experiment_id: cfg.experiment_id, fingerprint: fp, config_filename: `${cfg.experiment_id}.yaml` })
      setLaunched({ fingerprint: fp })
      setModal(false)
    } catch (e) { console.error(e) }
    finally { setBusy(false) }
  }

  return (
    <div data-testid="config-view" style={{ height: '100%', display: 'flex', overflow: 'hidden' }}>

      {/* Form panel */}
      <div style={{ width: '54%', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 12, padding: '18px 14px 14px 20px', overflowY: 'auto', borderRight: '1px solid var(--border-subtle)' }}>

        <CfgInput label="Experiment ID" value={cfg.experiment_id} onChange={set('experiment_id')} placeholder="exp005_my_run" />

        <div>
          <div style={{ ...SEC_LABEL, marginBottom: 8 }}>Architecture</div>
          <ChipGroup label="Backbone" value={cfg.backbone} onChange={set('backbone')} options={[
            { label: 'MobileNetV2', value: 'mobilenetv2' },
            { label: 'MobileNetV3', value: 'mobilenetv3' },
            { label: 'ResNet-50',   value: 'resnet50' },
          ]} />
        </div>

        <div>
          <div style={{ ...SEC_LABEL, marginBottom: 8 }}>Training</div>
          <ChipGroup label="Optimizer" value={cfg.optimizer} onChange={set('optimizer')} options={[
            { label: 'Adam',  value: 'adam'  },
            { label: 'AdamW', value: 'adamw' },
            { label: 'SGD',   value: 'sgd'   },
          ]} />
        </div>

        <div>
          <div style={{ ...SEC_LABEL, marginBottom: 8 }}>Augmentation</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <CfgToggle label="Horizontal flip"  value={cfg.aug_flip}  onChange={set('aug_flip')} />
            <CfgToggle label="Color jitter"     value={cfg.aug_color} onChange={set('aug_color')} />
            <CfgToggle label="Random crop"      value={cfg.aug_crop}  onChange={set('aug_crop')} />
            <CfgToggle label="Random scale"     value={cfg.aug_scale} onChange={set('aug_scale')} />
          </div>
        </div>

        <div>
          <div style={{ ...SEC_LABEL, marginBottom: 8 }}>Deploy</div>
          <ChipGroup label="Region" value={cfg.region} onChange={set('region')} options={[
            { label: 'us-east-1',      value: 'us-east-1'      },
            { label: 'us-west-2',      value: 'us-west-2'      },
            { label: 'ap-southeast-1', value: 'ap-southeast-1' },
          ]} />
          <div style={{ marginTop: 10 }}>
            <CfgToggle label="Spot instance" sub="~70% cost reduction" value={cfg.spot} onChange={set('spot')} />
          </div>
        </div>

        <PillButton onClick={() => setModal(true)} style={{ width: '100%', justifyContent: 'center' }}>
          Launch Experiment
        </PillButton>
      </div>

      {/* YAML preview panel */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: '18px 16px 14px', overflow: 'hidden' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
          <span style={SEC_LABEL}>YAML Preview</span>
          <PillButton onClick={handleCopy} style={{ padding: '4px 12px', fontSize: '11px' }}>
            {copied ? 'Copied!' : 'Copy'}
          </PillButton>
        </div>
        <div data-testid="yaml-preview" style={{ flex: 1, overflowY: 'auto', fontFamily: 'monospace', fontSize: '11px', lineHeight: 1.75, whiteSpace: 'pre' }}>
          {yaml}
        </div>
      </div>

      {/* Launch modal */}
      {modal && (
        <div data-testid="launch-modal" style={{ position: 'fixed', inset: 0, zIndex: 200, background: 'rgba(0,0,0,0.65)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)', borderRadius: 16, padding: '24px 26px', maxWidth: 400, width: '90%' }}>
            <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 16 }}>Launch Experiment</div>
            <div style={{ fontSize: '11.5px', color: 'var(--text-tertiary)', marginBottom: 18 }}>
              Provisions <strong>{cfg.instance_type}</strong> in <strong>{cfg.region}</strong>{cfg.spot ? ' · spot' : ''}.
            </div>
            <div style={{ display: 'flex', gap: 8, marginTop: 18 }}>
              <PillButton onClick={() => setModal(false)} style={{ flex: 1, justifyContent: 'center' }}>Cancel</PillButton>
              <button
                data-testid="confirm-launch"
                onClick={handleLaunch}
                disabled={busy}
                style={{ flex: 2, padding: '8px 10px', borderRadius: 999, border: '1px solid rgba(0,212,160,0.44)', background: 'rgba(0,212,160,0.14)', color: 'var(--accent)', fontSize: '12.5px', fontFamily: 'inherit', fontWeight: 700, cursor: busy ? 'not-allowed' : 'pointer' }}
              >
                {busy ? '…' : 'Confirm Launch'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Success banner */}
      {launched && (
        <div data-testid="launch-banner" style={{ position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', zIndex: 300, padding: '12px 18px', borderRadius: 12, background: 'var(--bg-elevated)', border: '1px solid rgba(101,193,106,0.35)' }}>
          <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>Training launched</div>
          <div data-testid="launch-fingerprint" style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>fp · {launched.fingerprint}</div>
          <button onClick={() => setLaunched(null)} style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer' }}>✕</button>
        </div>
      )}
    </div>
  )
}
