/* dash-config.jsx — Experiment config builder */

const SEC_LABEL = {
  fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)',
  letterSpacing: '0.5px', textTransform: 'uppercase',
};
const DIVIDER = {
  borderTop: '1px solid var(--border-subtle)', margin: '4px 0',
};

const CfgInput = ({ label, value, onChange, type = 'text', mono = false, placeholder = '' }) => {
  const [foc, setFoc] = React.useState(false);
  return (
    <div>
      {label && <div style={{ ...SEC_LABEL, marginBottom: 6 }}>{label}</div>}
      <input
        type={type} value={value} placeholder={placeholder}
        onChange={e => onChange(type === 'number' ? +e.target.value : e.target.value)}
        onFocus={() => setFoc(true)} onBlur={() => setFoc(false)}
        style={{
          width: '100%', padding: '7px 11px', borderRadius: 8,
          fontSize: '12px', fontFamily: mono ? 'monospace' : 'inherit',
          background: 'var(--bg-pill)',
          border: `1px solid ${foc ? 'var(--accent)' : 'var(--border-subtle)'}`,
          color: 'var(--text-primary)', outline: 'none',
          transition: 'border-color 0.2s', boxSizing: 'border-box',
        }}
      />
    </div>
  );
};

const CfgSelect = ({ label, value, onChange, options }) => {
  const [foc, setFoc] = React.useState(false);
  return (
    <div>
      {label && <div style={{ ...SEC_LABEL, marginBottom: 6 }}>{label}</div>}
      <select value={value} onChange={e => onChange(e.target.value)}
        onFocus={() => setFoc(true)} onBlur={() => setFoc(false)}
        style={{
          width: '100%', padding: '7px 28px 7px 11px', borderRadius: 8,
          fontSize: '12px', fontFamily: 'inherit', cursor: 'pointer',
          background: 'var(--bg-pill)', outline: 'none',
          border: `1px solid ${foc ? 'var(--accent)' : 'var(--border-subtle)'}`,
          color: 'var(--text-primary)', appearance: 'none',
          backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23494e4d'/%3E%3C/svg%3E")`,
          backgroundRepeat: 'no-repeat', backgroundPosition: 'right 10px center',
          transition: 'border-color 0.2s', boxSizing: 'border-box',
        }}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
};

const ChipGroup = ({ label, options, value, onChange }) => (
  <div>
    {label && <div style={{ ...SEC_LABEL, marginBottom: 6 }}>{label}</div>}
    <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
      {options.map(opt => (
        <PillButton key={opt.value} active={value === opt.value} onClick={() => onChange(opt.value)}
          style={{ padding: '5px 12px', fontSize: '11.5px' }}>
          {opt.label}
        </PillButton>
      ))}
    </div>
  </div>
);

const CfgToggle = ({ label, value, onChange, sub }) => (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, padding: '3px 0' }}>
    <div style={{ minWidth: 0 }}>
      <div style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>{label}</div>
      {sub && <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginTop: 1 }}>{sub}</div>}
    </div>
    <div onClick={() => onChange(!value)} style={{
      width: 36, height: 20, borderRadius: 999, cursor: 'pointer', flexShrink: 0, position: 'relative',
      background: value ? 'var(--accent)' : 'rgba(255,255,255,0.1)',
      boxShadow: value ? '0 0 8px rgba(0,212,160,0.3)' : 'none',
      transition: 'background 0.2s, box-shadow 0.2s',
    }}>
      <div style={{
        position: 'absolute', top: 3, left: value ? 17 : 3,
        width: 14, height: 14, borderRadius: '50%', background: '#fff',
        boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
        transition: 'left 0.18s cubic-bezier(0.34,1.56,0.64,1)',
      }}/>
    </div>
  </div>
);

const CfgSlider = ({ label, value, onChange, min, max, step = 1, fmt = v => v }) => (
  <div>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 7 }}>
      <div style={SEC_LABEL}>{label}</div>
      <span style={{ fontSize: '12px', fontFamily: 'monospace', color: 'var(--accent)', fontWeight: 700 }}>{fmt(value)}</span>
    </div>
    <input type="range" min={min} max={max} step={step} value={value}
      onChange={e => onChange(+e.target.value)}
      style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent)' }}/>
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: 'var(--text-tertiary)', marginTop: 2 }}>
      <span>{fmt(min)}</span><span>{fmt(max)}</span>
    </div>
  </div>
);

const Section = ({ title, children }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
    <div style={{ ...SEC_LABEL }}>{title}</div>
    {children}
    <div style={DIVIDER}/>
  </div>
);

const YAMLView = ({ yaml }) => (
  <div style={{ fontFamily: 'monospace', fontSize: '11px', lineHeight: 1.75 }}>
    {yaml.split('\n').map((line, i) => {
      if (!line.trim()) return <div key={i} style={{ height: '0.75em' }}/>;
      if (line.startsWith('#')) return (
        <div key={i} style={{ color: 'var(--text-tertiary)', fontStyle: 'italic' }}>{line}</div>
      );
      const m = line.match(/^(\s*)([\w_]+):\s*(.*)$/);
      if (!m) return <div key={i} style={{ color: 'var(--text-secondary)' }}>{line}</div>;
      const [, indent, key, val] = m;
      const depth = indent.length / 2;
      const kc = depth === 0 ? 'var(--accent)' : depth === 1 ? 'rgba(0,212,160,0.72)' : 'rgba(0,212,160,0.50)';
      let vc = '#e88548';
      if      (val === 'true')  vc = 'var(--success)';
      else if (val === 'false') vc = 'var(--danger)';
      else if (val === 'null' || val === '') vc = null;
      else if (!isNaN(+val) || /^\d+e[+-]?\d+$/i.test(val)) vc = '#7c9ef5';
      else if (val.startsWith('[')) vc = '#7c9ef5';
      return (
        <div key={i}>
          <span style={{ color: 'var(--text-tertiary)' }}>{indent}</span>
          <span style={{ color: kc }}>{key}:</span>
          {vc && val && <span style={{ color: vc }}> {val}</span>}
        </div>
      );
    })}
  </div>
);

const LaunchModal = ({ cfg, onConfirm, onCancel, busy }) => (
  <div style={{
    position: 'fixed', inset: 0, zIndex: 200,
    background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(8px)',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
  }}>
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border-medium)',
      borderRadius: 16, padding: '24px 26px', maxWidth: 400, width: '90%',
      boxShadow: '0 24px 60px rgba(0,0,0,0.6)',
      animation: 'modalIn 0.22s cubic-bezier(0.34,1.56,0.64,1)',
    }}>
      <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
        Launch Experiment
      </div>
      <div style={{ fontSize: '11.5px', color: 'var(--text-tertiary)', marginBottom: 18, lineHeight: 1.6 }}>
        Provisions{' '}
        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{cfg.instance_type}</span>
        {' '}in{' '}
        <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}>{cfg.region}</span>
        {cfg.spot ? ' · spot pricing' : ''}.
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {[
          ['Experiment', cfg.experiment_id],
          ['Backbone',   cfg.backbone],
          ['Input',      `${cfg.input_size}×${cfg.input_size}`],
          ['Optimizer',  cfg.optimizer],
          ['LR',         cfg.learning_rate.toExponential(1)],
          ['Batch / Ep', `${cfg.batch_size} / ${cfg.epochs}`],
          ['Instance',   cfg.instance_type],
          ['Spot',       cfg.spot ? 'Yes' : 'No'],
        ].map(([k, v]) => (
          <div key={k} style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '6px 10px', borderRadius: 8,
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.05)',
            marginBottom: 4,
          }}>
            <span style={{ fontSize: '10px', color: 'var(--text-tertiary)', minWidth: 52 }}>{k}</span>
            <span style={{ fontSize: '11.5px', color: 'var(--text-primary)', fontFamily: 'monospace', fontWeight: 600 }}>{v}</span>
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', gap: 8, marginTop: 18 }}>
        <PillButton onClick={onCancel} style={{ flex: 1, justifyContent: 'center' }}>Cancel</PillButton>
        <button onClick={onConfirm} disabled={busy} style={{
          flex: 2, padding: '8px 10px', borderRadius: 999, border: `1px solid ${'var(--accent)'}44`,
          background: busy ? 'rgba(0,212,160,0.06)' : 'rgba(0,212,160,0.14)',
          color: busy ? 'var(--text-tertiary)' : 'var(--accent)',
          fontSize: '12.5px', fontFamily: 'inherit', fontWeight: 700,
          cursor: busy ? 'not-allowed' : 'pointer', transition: 'all 0.2s',
        }}>
          {busy ? '…' : 'Confirm Launch'}
        </button>
      </div>
    </div>
  </div>
);

const LaunchBanner = ({ result, onDismiss }) => (
  <div style={{
    position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
    zIndex: 300, display: 'flex', alignItems: 'center', gap: 12,
    padding: '12px 18px', borderRadius: 12,
    background: 'var(--bg-elevated)',
    border: '1px solid rgba(101,193,106,0.35)',
    boxShadow: '0 0 24px rgba(101,193,106,0.12), 0 10px 32px rgba(0,0,0,0.5)',
    animation: 'modalIn 0.28s cubic-bezier(0.34,1.56,0.64,1)',
  }}>
    <ShieldCheck size={16}/>
    <div>
      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-primary)' }}>Training launched</div>
      <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>fp · {result.fingerprint}</div>
    </div>
    <button onClick={onDismiss} style={{
      background: 'none', border: 'none', color: 'var(--text-tertiary)',
      cursor: 'pointer', fontSize: '12px', padding: '4px 6px', marginLeft: 4,
    }}>✕</button>
  </div>
);

const toYAML = cfg => [
  `# sentinel> config builder`,
  `experiment_id: ${cfg.experiment_id}`,
  ``,
  `model:`,
  `  backbone: ${cfg.backbone}`,
  `  pretrained: ${cfg.pretrained}`,
  `  input_size: [${cfg.input_size}, ${cfg.input_size}, 3]`,
  `  num_classes: ${cfg.num_classes}`,
  ``,
  `training:`,
  `  optimizer: ${cfg.optimizer}`,
  `  learning_rate: ${cfg.learning_rate.toExponential(1)}`,
  `  lr_schedule: ${cfg.lr_schedule}`,
  `  warmup_epochs: ${cfg.warmup_epochs}`,
  `  batch_size: ${cfg.batch_size}`,
  `  epochs: ${cfg.epochs}`,
  `  use_amp: ${cfg.use_amp}`,
  `  grad_clip: ${cfg.grad_clip}`,
  ``,
  `loss:`,
  `  classification: ${cfg.cls_loss}`,
  `  localization: ${cfg.loc_loss}`,
  ...(cfg.cls_loss === 'focal' ? [`  focal:`, `    alpha: ${cfg.focal_alpha}`, `    gamma: ${cfg.focal_gamma}`] : []),
  `  neg_pos_ratio: ${cfg.neg_pos_ratio}`,
  ``,
  `data:`,
  `  dataset: ${cfg.dataset}`,
  `  augmentations:`,
  `    horizontal_flip: ${cfg.aug_flip}`,
  `    color_jitter: ${cfg.aug_color}`,
  `    random_crop: ${cfg.aug_crop}`,
  `    random_scale: ${cfg.aug_scale}`,
  ``,
  `deploy:`,
  `  instance_type: ${cfg.instance_type}`,
  `  region: ${cfg.region}`,
  `  spot_instance: ${cfg.spot}`,
].join('\n');

const DEFAULT_CFG = {
  experiment_id: 'exp005_custom',
  backbone: 'mobilenetv2', pretrained: true, input_size: 300, num_classes: 21,
  optimizer: 'adam', learning_rate: 1e-3, lr_schedule: 'cosine',
  warmup_epochs: 5, batch_size: 32, epochs: 200, use_amp: true, grad_clip: 1.0,
  cls_loss: 'focal', loc_loss: 'smooth_l1', focal_alpha: 0.25, focal_gamma: 2.0, neg_pos_ratio: 3,
  dataset: 'voc2012', aug_flip: true, aug_color: true, aug_crop: true, aug_scale: false,
  instance_type: 'p3.2xlarge', region: 'us-east-1', spot: false,
};

const ConfigView = () => {
  const [cfg, setCfg]       = React.useState({...DEFAULT_CFG});
  const [modal, setModal]   = React.useState(false);
  const [busy, setBusy]     = React.useState(false);
  const [launched, setLaunched] = React.useState(null);
  const [copied, setCopied] = React.useState(false);

  const set = key => val => setCfg(c => ({...c, [key]: val}));
  const yaml = React.useMemo(() => toYAML(cfg), [cfg]);

  const handleCopy = () => {
    navigator.clipboard?.writeText(yaml);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleLaunch = async () => {
    setBusy(true);
    try {
      const fp = Math.random().toString(16).slice(2, 10);
      await launchTraining({ experiment_id: cfg.experiment_id, fingerprint: fp, config_filename: `${cfg.experiment_id}.yaml` });
      setLaunched({ fingerprint: fp });
      setModal(false);
    } catch(e) { console.error(e); }
    finally { setBusy(false); }
  };

  return (
    <div style={{ height: '100%', display: 'flex', gap: 0, overflow: 'hidden' }}>

      <div style={{
        width: '54%', flexShrink: 0,
        display: 'flex', flexDirection: 'column', gap: 12,
        padding: '18px 14px 14px 20px', overflowY: 'auto',
        borderRight: '1px solid var(--border-subtle)',
      }}>

        {/* Experiment ID always at top */}
        <FadeIn delay={0.05} from="left">
          <CfgInput label="Experiment ID" value={cfg.experiment_id}
            onChange={set('experiment_id')} mono placeholder="exp005_my_run"/>
        </FadeIn>

        <div style={DIVIDER}/>

        <FadeIn delay={0.1} from="left">
          <Section title="Architecture">
            <ChipGroup label="Backbone" value={cfg.backbone} onChange={set('backbone')} options={[
              {label:'MobileNetV2', value:'mobilenetv2'},
              {label:'MobileNetV3', value:'mobilenetv3'},
              {label:'ResNet-50',   value:'resnet50'},
              {label:'EfficientB0', value:'efficientnet_b0'},
            ]}/>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <ChipGroup label="Input size" value={cfg.input_size} onChange={set('input_size')} options={[
                {label:'300', value:300}, {label:'416', value:416}, {label:'512', value:512},
              ]}/>
              <CfgInput label="Num classes" value={cfg.num_classes} onChange={set('num_classes')} type="number" mono/>
            </div>
            <CfgToggle label="ImageNet pretrained" sub="Freeze backbone for first 5 epochs"
              value={cfg.pretrained} onChange={set('pretrained')}/>
          </Section>
        </FadeIn>

        <FadeIn delay={0.14} from="left">
          <Section title="Training">
            <ChipGroup label="Optimizer" value={cfg.optimizer} onChange={set('optimizer')} options={[
              {label:'Adam',value:'adam'},{label:'AdamW',value:'adamw'},
              {label:'SGD',value:'sgd'},{label:'RMSprop',value:'rmsprop'},
            ]}/>
            <CfgSlider label="Learning rate" value={cfg.learning_rate} onChange={set('learning_rate')}
              min={1e-5} max={1e-2} step={1e-5} fmt={v => v.toExponential(1)}/>
            <ChipGroup label="LR schedule" value={cfg.lr_schedule} onChange={set('lr_schedule')} options={[
              {label:'Cosine',value:'cosine'},{label:'Step',value:'step'},
              {label:'Exponential',value:'exp'},{label:'None',value:'none'},
            ]}/>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <CfgSlider label="Warmup epochs" value={cfg.warmup_epochs}
                onChange={set('warmup_epochs')} min={0} max={20}/>
              <ChipGroup label="Batch size" value={cfg.batch_size} onChange={set('batch_size')} options={[
                {label:'8',value:8},{label:'16',value:16},{label:'32',value:32},{label:'64',value:64},
              ]}/>
            </div>
            <CfgSlider label="Total epochs" value={cfg.epochs} onChange={set('epochs')} min={50} max={500} step={10}/>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <CfgToggle label="Mixed precision" sub="fp16 + fp32 (AMP)" value={cfg.use_amp} onChange={set('use_amp')}/>
              <CfgInput  label="Grad clip norm" value={cfg.grad_clip} onChange={set('grad_clip')} type="number" mono/>
            </div>
          </Section>
        </FadeIn>

        <FadeIn delay={0.18} from="left">
          <Section title="Loss">
            <ChipGroup label="Classification" value={cfg.cls_loss} onChange={set('cls_loss')} options={[
              {label:'Cross-Entropy', value:'cross_entropy'},
              {label:'Focal Loss',   value:'focal'},
            ]}/>
            {cfg.cls_loss === 'focal' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                <CfgSlider label="Focal α" value={cfg.focal_alpha} onChange={set('focal_alpha')}
                  min={0.1} max={0.9} step={0.05} fmt={v => v.toFixed(2)}/>
                <CfgSlider label="Focal γ" value={cfg.focal_gamma} onChange={set('focal_gamma')}
                  min={0.5} max={5.0} step={0.5}  fmt={v => v.toFixed(1)}/>
              </div>
            )}
            <ChipGroup label="Localization" value={cfg.loc_loss} onChange={set('loc_loss')} options={[
              {label:'Smooth L1',value:'smooth_l1'},{label:'L1',value:'l1'},{label:'IoU',value:'iou'},
            ]}/>
            <CfgSlider label="Neg/Pos ratio" value={cfg.neg_pos_ratio}
              onChange={set('neg_pos_ratio')} min={1} max={6} fmt={v => `${v}:1`}/>
          </Section>
        </FadeIn>

        <FadeIn delay={0.22} from="left">
          <Section title="Dataset & Augmentation">
            <ChipGroup label="Dataset" value={cfg.dataset} onChange={set('dataset')} options={[
              {label:'VOC 2012',  value:'voc2012'},
              {label:'VOC 07+12', value:'voc0712'},
              {label:'VisDrone',  value:'visdrone'},
              {label:'Custom',    value:'custom'},
            ]}/>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <CfgToggle label="Horizontal flip"  value={cfg.aug_flip}   onChange={set('aug_flip')}/>
              <CfgToggle label="Color jitter"     value={cfg.aug_color}  onChange={set('aug_color')}/>
              <CfgToggle label="Random crop"      value={cfg.aug_crop}   onChange={set('aug_crop')}/>
              <CfgToggle label="Random scale"     value={cfg.aug_scale}  onChange={set('aug_scale')}/>
            </div>
          </Section>
        </FadeIn>

        <FadeIn delay={0.26} from="left">
          <Section title="Infrastructure">
            <ChipGroup label="Instance type" value={cfg.instance_type} onChange={set('instance_type')} options={[
              {label:'p2.xlarge',   value:'p2.xlarge'},
              {label:'p3.2xlarge',  value:'p3.2xlarge'},
              {label:'g4dn.xlarge', value:'g4dn.xlarge'},
              {label:'g5.xlarge',   value:'g5.xlarge'},
            ]}/>
            <CfgSelect label="Region" value={cfg.region} onChange={set('region')} options={[
              {label:'us-east-1 · N. Virginia',   value:'us-east-1'},
              {label:'us-west-2 · Oregon',         value:'us-west-2'},
              {label:'eu-west-1 · Ireland',        value:'eu-west-1'},
              {label:'eu-central-1 · Frankfurt',   value:'eu-central-1'},
              {label:'ap-southeast-1 · Singapore', value:'ap-southeast-1'},
              {label:'ap-northeast-1 · Tokyo',     value:'ap-northeast-1'},
            ]}/>
            <CfgToggle label="Spot instances" sub="Up to 70% cheaper · may interrupt"
              value={cfg.spot} onChange={set('spot')}/>
          </Section>
        </FadeIn>

        {/* Action buttons */}
        <FadeIn delay={0.3} from="left">
          <div style={{ display: 'flex', gap: 8, paddingBottom: 4 }}>
            <button onClick={() => setCfg({...DEFAULT_CFG})} style={{
              flex: 1, padding: '7px 10px', borderRadius: 8, fontSize: '11.5px', fontWeight: 600,
              cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.2s ease', outline: 'none',
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-pill)', color: 'var(--text-secondary)',
            }}>Reset</button>
            <button onClick={() => setModal(true)} style={{
              flex: 3, padding: '7px 10px', borderRadius: 8, fontSize: '11.5px', fontWeight: 600,
              cursor: 'pointer', fontFamily: 'inherit', transition: 'all 0.2s ease', outline: 'none',
              border: '1px solid rgba(0,212,160,0.44)',
              background: 'rgba(0,212,160,0.14)', color: 'var(--accent)',
            }}>Schedule & Launch</button>
          </div>
        </FadeIn>

      </div>

      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0,
      }}>
        {/* Header bar */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0 16px', height: 36, flexShrink: 0,
          borderBottom: '1px solid var(--border-subtle)',
          background: 'var(--bg-secondary)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ display: 'flex', gap: 5 }}>
              {['var(--danger)', 'var(--warning)', 'var(--success)'].map(c => (
                <div key={c} style={{ width: 9, height: 9, borderRadius: '50%', background: c, opacity: 0.65 }}/>
              ))}
            </div>
            <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>
              {cfg.experiment_id}.yaml
            </span>
          </div>
          <button onClick={handleCopy} style={{
            padding: '4px 12px', borderRadius: 999,
            border: '1px solid var(--border-subtle)',
            background: 'var(--bg-pill)',
            color: copied ? 'var(--accent)' : 'var(--text-secondary)',
            fontSize: '11px', fontFamily: 'inherit', fontWeight: 600,
            cursor: 'pointer', transition: 'color 0.2s',
          }}>
            {copied ? '✓ Copied' : 'Copy'}
          </button>
        </div>

        {/* Code area */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
          <YAMLView yaml={yaml}/>
        </div>
      </div>

      {modal   && <LaunchModal cfg={cfg} onConfirm={handleLaunch} onCancel={() => setModal(false)} busy={busy}/>}
      {launched && <LaunchBanner result={launched} onDismiss={() => setLaunched(null)}/>}
    </div>
  );
};

Object.assign(window, { ConfigView });
