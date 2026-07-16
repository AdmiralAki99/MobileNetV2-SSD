/* dash-app.jsx — ML Pipeline Dashboard */

const STATUS_COLOR = {
  pending:         'var(--warning)',
  running:         'var(--accent)',
  success:         'var(--success)',
  failed:          'var(--danger)',
  completed:       'var(--success)',
  processing:      'var(--accent)',
  queued:          'var(--warning)',
  skipped:         'var(--text-tertiary)',
  upstream_failed: 'var(--danger)',
  current:         'var(--success)',
  deprecated:      'var(--warning)',
  archived:        'var(--text-tertiary)',
};

const StatusBadge = ({ status }) => (
  <span style={{
    fontSize: '10px', fontWeight: 600, letterSpacing: '0.5px',
    padding: '3px 8px', borderRadius: 999, textTransform: 'uppercase',
    background: `${STATUS_COLOR[status] || '#555'}22`,
    color: STATUS_COLOR[status] || 'var(--text-tertiary)',
    border: `1px solid ${STATUS_COLOR[status] || '#555'}44`,
  }}>{status || 'unknown'}</span>
);

const Header = ({ statusFilter, setStatusFilter, viewMode, setViewMode }) => {
  const statuses = ['all', 'pending', 'running', 'success', 'failed'];
  return (
    <FadeIn delay={0.1} from="top" style={{
      display: 'flex', alignItems: 'center', gap: 16, padding: '0 20px',
      height: 52, borderBottom: '1px solid var(--border-subtle)',
      background: 'var(--bg-secondary)', flexShrink: 0,
    }}>
      <div style={{ fontSize: '16px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.3px', marginRight: 8, whiteSpace: 'nowrap' }}>
        sentinel<span style={{ color: 'var(--accent)' }}>&gt;</span>
      </div>
      <TabGroup
        tabs={['Pipeline', 'Metrics', 'ETL', 'Ops', 'Deploy', 'Config']}
        active={viewMode === 'metrics' ? 'Metrics' : viewMode === 'config' ? 'Config' : viewMode === 'etl' ? 'ETL' : viewMode === 'ops' ? 'Ops' : viewMode === 'deploy' ? 'Deploy' : 'Pipeline'}
        onChange={t => setViewMode(t === 'Metrics' ? 'metrics' : t === 'Config' ? 'config' : t === 'ETL' ? 'etl' : t === 'Ops' ? 'ops' : t === 'Deploy' ? 'deploy' : 'pipeline')}
      />
      {viewMode === 'pipeline' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, flex: 1, overflow: 'hidden', minWidth: 0 }}>
          {statuses.map(s => (
            <PillButton key={s} active={statusFilter === s} onClick={() => setStatusFilter(s)}>{s}</PillButton>
          ))}
        </div>
      )}
      {viewMode === 'metrics' && <div style={{ flex: 1 }}/>}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
        <SearchBar/>
        <button style={{
          width: 34, height: 34, borderRadius: '50%', border: '1px solid var(--border-subtle)',
          background: 'var(--bg-pill)', color: 'var(--text-secondary)',
          cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}><GearSvg size={15}/></button>
      </div>
    </FadeIn>
  );
};

const ExperimentRow = ({ exp, selected, onClick }) => (
  <div onClick={onClick} style={{
    padding: '10px 12px', borderRadius: 10, cursor: 'pointer', marginBottom: 6,
    border: `1px solid ${selected ? 'var(--accent)' : 'var(--border-subtle)'}`,
    background: selected ? 'rgba(0,212,160,0.06)' : 'var(--bg-surface)',
    transition: 'all 0.2s ease',
  }}
    onMouseEnter={e => { if (!selected) e.currentTarget.style.borderColor = 'var(--border-medium)'; }}
    onMouseLeave={e => { if (!selected) e.currentTarget.style.borderColor = 'var(--border-subtle)'; }}
  >
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
      <span style={{ fontSize: '12.5px', fontWeight: 600, color: 'var(--text-primary)' }}>{exp.experiment_id}</span>
      <StatusBadge status={exp.status}/>
    </div>
    <span style={{ fontSize: '10.5px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{exp.fingerprint}</span>
  </div>
);

const LeftPanel = ({ experiments, loading, selectedExp, setSelectedExp, statusFilter, refresh }) => {
  const filtered = (experiments || []).filter(e =>
    statusFilter === 'all' || e.status === statusFilter
  );
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 10,
      padding: '18px 12px 14px 18px', overflow: 'hidden', minWidth: 0,
    }}>
      <FadeIn delay={0.2} from="left">
        <div style={{ marginBottom: 8 }}>
          <h1 style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1.15, letterSpacing: '-0.4px', margin: 0 }}>
            ML<br/>Pipeline
          </h1>
          <p style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginTop: 4 }}>
            {loading ? 'Loading…' : `${(experiments || []).length} experiments`}
          </p>
        </div>
      </FadeIn>
      <FadeIn delay={0.3} from="left" style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {filtered.length === 0 && !loading && (
          <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '20px 0' }}>No experiments.</div>
        )}
        {filtered.map(exp => (
          <ExperimentRow
            key={`${exp.experiment_id}_${exp.fingerprint}`}
            exp={exp} selected={selectedExp?.experiment_id === exp.experiment_id && selectedExp?.fingerprint === exp.fingerprint}
            onClick={() => setSelectedExp(exp)}
          />
        ))}
      </FadeIn>
      <FadeIn delay={0.5} from="left">
        <PillButton onClick={refresh} style={{ width: '100%', justifyContent: 'center' }}>Refresh</PillButton>
      </FadeIn>
    </div>
  );
};

const ActionBtn = ({ label, color, disabled, busy, onClick }) => (
  <button onClick={!disabled && !busy ? onClick : undefined} style={{
    width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
    padding: '7px 10px', borderRadius: 8, fontSize: '11.5px', fontWeight: 600,
    cursor: (disabled || busy) ? 'not-allowed' : 'pointer', fontFamily: 'inherit',
    transition: 'all 0.2s ease', outline: 'none',
    border: `1px solid ${color}44`,
    background: `${color}14`,
    color: (disabled || busy) ? 'var(--text-tertiary)' : color,
    opacity: (disabled || busy) ? 0.45 : 1,
  }}>
    {busy ? '…' : label}
  </button>
);

const ArtifactRow = ({ label, ok }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0' }}>
    {ok ? <ShieldCheck size={13}/> : <AlertCircle size={13} color="var(--text-tertiary)"/>}
    <span style={{ fontSize: '11.5px', color: ok ? 'var(--text-primary)' : 'var(--text-tertiary)' }}>{label}</span>
  </div>
);

const RightPanel = ({ selectedExp, refreshExps }) => {
  const [instanceInfo, setInstanceInfo] = React.useState(null);
  const [launching,    setLaunching]    = React.useState(false);
  const [stopping,     setStopping]     = React.useState(false);
  const [jobId,        setJobId]        = React.useState(null);
  const [jobType,      setJobType]      = React.useState(null);
  const [jobStatus,    setJobStatus]    = React.useState(null);
  const [artifacts,    setArtifacts]    = React.useState(null);

  // Fetch EC2 instance info when a running experiment is selected
  React.useEffect(() => {
    if (!selectedExp?.ec2_instance) { setInstanceInfo(null); return; }
    fetchInstanceStatus(selectedExp.ec2_instance)
      .then(res => setInstanceInfo(res?.message || null))
      .catch(() => setInstanceInfo(null));
  }, [selectedExp?.ec2_instance]);

  // Fetch artifacts when experiment changes
  React.useEffect(() => {
    if (!selectedExp) { setArtifacts(null); return; }
    fetchArtifacts(selectedExp.experiment_id)
      .then(res => setArtifacts(res?.artifat_status || null))
      .catch(() => setArtifacts(null));
  }, [selectedExp?.experiment_id, selectedExp?.fingerprint]);

  // Poll export job while running
  const { data: jobData } = usePolling(
    () => fetchJobStatus(jobId),
    3000,
    !!jobId && jobStatus === 'running'
  );
  React.useEffect(() => {
    if (jobData?.job_status?.status) setJobStatus(jobData.job_status.status);
  }, [jobData]);

  const handleLaunch = async () => {
    setLaunching(true);
    try {
      await launchTraining({
        experiment_id:   selectedExp.experiment_id,
        fingerprint:     selectedExp.fingerprint,
        config_filename: selectedExp.config_filename || `${selectedExp.experiment_id}.yaml`,
      });
      refreshExps();
    } catch (e) { console.error('Launch failed', e); }
    finally { setLaunching(false); }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await stopTraining({
        instance_id:   selectedExp.ec2_instance,
        experiment_id: selectedExp.experiment_id,
        fingerprint:   selectedExp.fingerprint,
      });
      refreshExps();
    } catch (e) { console.error('Stop failed', e); }
    finally { setStopping(false); }
  };

  const handleExportModel = async () => {
    try {
      const res = await exportSavedModel({
        experiment_id:      selectedExp.experiment_id,
        fingerprint:        selectedExp.fingerprint,
        checkpoint_s3_path: selectedExp.checkpoint_s3_path,
        config_filename:    selectedExp.config_filename || `${selectedExp.experiment_id}.yaml`,
      });
      setJobId(res.job_id); setJobType('savedmodel'); setJobStatus('running');
    } catch (e) { console.error('Export SavedModel failed', e); }
  };

  const handleExportOnnx = async () => {
    try {
      const res = await exportOnnx({ experiment_id: selectedExp.experiment_id });
      setJobId(res.job_id); setJobType('onnx'); setJobStatus('running');
    } catch (e) { console.error('Export ONNX failed', e); }
  };

  const canLaunch      = selectedExp?.status === 'pending' || selectedExp?.status === 'failed';
  const canStop        = selectedExp?.status === 'running' && !!selectedExp?.ec2_instance;
  const canExportModel = !!selectedExp?.checkpoint_s3_path;
  const canExportOnnx  = !!artifacts?.saved_model;
  const mapPct         = selectedExp?.best_metric != null ? Math.round(selectedExp.best_metric * 100) : null;

  if (!selectedExp) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
        <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>Select an experiment</span>
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 12,
      padding: '18px 14px 14px 10px', overflowY: 'auto', minWidth: 0,
    }}>
      {/* Gauges */}
      <FadeIn delay={0.3} from="right">
        <StatCard
          label="Best Epoch"
          value={selectedExp.best_epoch != null ? String(selectedExp.best_epoch) : '—'}
          total="200" gauge={selectedExp.best_epoch != null} greenPct={0.7}
        />
      </FadeIn>
      <FadeIn delay={0.5} from="right">
        <StatCard
          label="Best mAP"
          value={mapPct != null ? `${mapPct}%` : '—'}
          total="100%" gauge={mapPct != null} greenPct={0.65}
        />
      </FadeIn>

      {/* TensorBoard link */}
      {instanceInfo?.tensorboard_url && (
        <FadeIn delay={0.55} from="right">
          <PillButton onClick={() => window.open(instanceInfo.tensorboard_url, '_blank')}
            style={{ width: '100%', justifyContent: 'center' }}>
            TensorBoard →
          </PillButton>
        </FadeIn>
      )}

      {/* Failure reason */}
      {selectedExp.failure_reason && (
        <FadeIn delay={0.6} from="right">
          <div style={{
            padding: '8px 12px', borderRadius: 10, wordBreak: 'break-word',
            background: 'rgba(232,72,85,0.08)', border: '1px solid rgba(232,72,85,0.25)',
            fontSize: '11px', color: 'var(--danger)',
          }}>{selectedExp.failure_reason}</div>
        </FadeIn>
      )}

      {/* Divider */}
      <div style={{ borderTop: '1px solid var(--border-subtle)', margin: '2px 0' }}/>

      {/* Action buttons */}
      <FadeIn delay={0.65} from="right">
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 6, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
          Actions
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <ActionBtn label="Launch Training"   color="var(--accent)"  disabled={!canLaunch}      busy={launching}                                         onClick={handleLaunch}/>
          <ActionBtn label="Stop Training"     color="var(--danger)"  disabled={!canStop}        busy={stopping}                                          onClick={handleStop}/>
          <ActionBtn label="Export SavedModel" color="var(--warning)" disabled={!canExportModel} busy={jobType === 'savedmodel' && jobStatus === 'running'} onClick={handleExportModel}/>
          <ActionBtn label="Export ONNX"       color="var(--warning)" disabled={!canExportOnnx}  busy={jobType === 'onnx'       && jobStatus === 'running'} onClick={handleExportOnnx}/>
        </div>
        {jobStatus && jobStatus !== 'running' && (
          <div style={{ marginTop: 8, fontSize: '11px', color: jobStatus === 'success' ? 'var(--success)' : 'var(--danger)' }}>
            {jobType} {'→'} {jobStatus}
          </div>
        )}
      </FadeIn>

      {/* Divider */}
      <div style={{ borderTop: '1px solid var(--border-subtle)', margin: '2px 0' }}/>

      {/* Artifact status */}
      <FadeIn delay={0.75} from="right">
        <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 6, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
          Artifacts
        </div>
        {!artifacts
          ? <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>—</span>
          : <div>
              <ArtifactRow label="SavedModel" ok={artifacts.saved_model}/>
              <ArtifactRow label="ONNX"       ok={artifacts.onnx}/>
              <ArtifactRow label="Priors"     ok={artifacts.priors}/>
            </div>
        }
      </FadeIn>
    </div>
  );
};

const RunDot = ({ exp, selected, onClick }) => {
  const cfg = {
    running: { bg: 'rgba(0,212,160,0.15)',  border: 'rgba(0,212,160,0.40)',  icon: <AlertCircle  size={12} color="var(--accent)"/>,   glow: 'rgba(0,212,160,0.30)'  },
    success: { bg: 'rgba(101,193,106,0.15)',border: 'rgba(101,193,106,0.40)',icon: <ShieldCheck  size={12}/>                      ,   glow: 'rgba(101,193,106,0.25)' },
    failed:  { bg: 'rgba(232,72,85,0.15)',  border: 'rgba(232,72,85,0.40)',  icon: <DangerTriangle size={12}/>                   ,   glow: 'rgba(232,72,85,0.25)'  },
    pending: { bg: 'rgba(255,255,255,0.04)',border: 'rgba(255,255,255,0.12)',icon: null                                          ,   glow: 'transparent'           },
  }[exp.status] || { bg: 'rgba(255,255,255,0.04)', border: 'rgba(255,255,255,0.10)', icon: null, glow: 'transparent' };

  return (
    <div
      onClick={onClick}
      title={`${exp.experiment_id} · ${exp.fingerprint}`}
      style={{
        width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: cfg.bg, cursor: 'pointer', transition: 'all 0.2s',
        border: selected ? '2px solid var(--text-primary)' : `1px solid ${cfg.border}`,
        boxShadow: selected
          ? `0 0 0 3px rgba(255,255,255,0.08), 0 0 10px ${cfg.glow}`
          : `0 0 8px ${cfg.glow}`,
      }}
      onMouseEnter={e => { e.currentTarget.style.transform = 'scale(1.15)'; }}
      onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)'; }}
    >
      {cfg.icon}
    </div>
  );
};

const RunMatrix = ({ experiments, selectedExp, setSelectedExp }) => {
  const groups = {
    running: (experiments || []).filter(e => e.status === 'running'),
    success: (experiments || []).filter(e => e.status === 'success'),
    pending: (experiments || []).filter(e => e.status === 'pending'),
    failed:  (experiments || []).filter(e => e.status === 'failed'),
  };
  const rows = [
    { label: 'Running', key: 'running' },
    { label: 'Success', key: 'success' },
    { label: 'Pending', key: 'pending' },
    { label: 'Failed',  key: 'failed'  },
  ];

  return (
    <div style={{ overflow: 'hidden', minWidth: 0 }}>
      <h3 style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 12px' }}>Run Matrix</h3>
      <div style={{ display: 'flex', gap: 14 }}>
        {/* Row labels */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18, paddingTop: 4, flexShrink: 0 }}>
          {rows.map(r => (
            <span key={r.key} style={{
              fontSize: '11.5px', color: 'var(--text-secondary)',
              whiteSpace: 'nowrap', height: 28, display: 'flex', alignItems: 'center',
            }}>{r.label}</span>
          ))}
        </div>

        {/* Dot rows */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {rows.map(r => (
            <div key={r.key} style={{ display: 'flex', gap: 8, flexWrap: 'wrap', minHeight: 28, alignItems: 'center' }}>
              {groups[r.key].length === 0
                ? <div style={{ width: 28, height: 28, borderRadius: '50%', border: '1px dashed rgba(255,255,255,0.07)' }}/>
                : groups[r.key].map(exp => (
                    <RunDot
                      key={`${exp.experiment_id}_${exp.fingerprint}`}
                      exp={exp}
                      selected={selectedExp?.experiment_id === exp.experiment_id && selectedExp?.fingerprint === exp.fingerprint}
                      onClick={() => setSelectedExp(exp)}
                    />
                  ))
              }
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const PipelineStats = ({ experiments }) => {
  const all     = experiments || [];
  const total   = all.length;
  const running = all.filter(e => e.status === 'running').length;
  const success = all.filter(e => e.status === 'success').length;
  const failed  = all.filter(e => e.status === 'failed').length;

  return (
    <div style={{ overflow: 'hidden', minWidth: 0 }}>
      <h3 style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)', margin: '0 0 12px' }}>Pipeline Overview</h3>

      {/* Legend — mirrors "Total Attacks by Alerts" legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px 16px', marginBottom: 14 }}>
        {[
          { icon: <ShieldCheck size={14}/>,     label: 'Success' },
          { icon: <AlertCircle size={14}/>,     label: 'Running' },
          { icon: <DangerTriangle size={14}/>,  label: 'Failed'  },
          { icon: <UnknownSquare size={14}/>,   label: 'Pending' },
        ].map(l => (
          <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            {l.icon}
            <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: 500 }}>{l.label}</span>
          </div>
        ))}
      </div>

      {/* Stat numbers — mirrors the "882k / 992k" style */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
        {[
          { label: 'Total',   val: total   },
          { label: 'Success', val: success  },
          { label: 'Running', val: running  },
          { label: 'Failed',  val: failed   },
        ].map((s, i) => (
          <FadeIn key={i} delay={0.9 + i * 0.1} from="bottom">
            <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 2 }}>{s.label}</div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 3 }}>
              <span style={{ fontSize: '22px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>{s.val}</span>
              <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>/ {total}</span>
            </div>
          </FadeIn>
        ))}
      </div>
    </div>
  );
};

const BottomSection = ({ experiments, selectedExp, setSelectedExp }) => (
  <FadeIn delay={0.7} from="bottom" style={{
    display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24,
    borderTop: '1px solid var(--border-subtle)', padding: '16px 20px 20px',
  }}>
    <RunMatrix experiments={experiments} selectedExp={selectedExp} setSelectedExp={setSelectedExp}/>
    <PipelineStats experiments={experiments}/>
  </FadeIn>
);

const ANNO_COLORS = ['#00d4a0', '#e88548', '#7b9ef0', '#e84855', '#65c16a', '#c67ef0', '#f0d07e'];
const annoColor = name => ANNO_COLORS[[...name].reduce((h, c) => Math.abs((h * 31 + c.charCodeAt(0)) | 0), 0) % ANNO_COLORS.length];
const MODELS = ['yolov8', 'rtdetr', 'grounding_dino'];

const BoxOverlay = ({ annotations }) => {
  const W = 400, H = Math.round(W * 9 / 16);
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}
      style={{ display: 'block', width: '100%', borderRadius: 8, background: 'rgba(255,255,255,0.025)', border: '1px solid var(--border-subtle)' }}>
      {[0.25, 0.5, 0.75].map(f => (
        <React.Fragment key={f}>
          <line x1={f*W} y1={0} x2={f*W} y2={H} stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
          <line x1={0} y1={f*H} x2={W} y2={f*H} stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
        </React.Fragment>
      ))}
      {annotations.map(ann => {
        const x = ann.x1 * W, y = ann.y1 * H;
        const w = (ann.x2 - ann.x1) * W, h = (ann.y2 - ann.y1) * H;
        const color = annoColor(ann.class_name);
        const above = y > 15;
        return (
          <g key={ann.id}>
            <rect x={x} y={y} width={w} height={h} rx="2"
              fill={`${color}22`} stroke={color} strokeWidth="1.5"/>
            <rect x={x} y={above ? y - 14 : y + h} width={64} height={13} rx="2"
              fill={color} opacity="0.9"/>
            <text x={x + 3} y={above ? y - 4 : y + h + 10}
              fontSize="8" fill="#0a0e0d" fontFamily="DM Sans, sans-serif" fontWeight="700">
              {ann.class_name} {Math.round(ann.consensus_score * 100)}%
            </text>
          </g>
        );
      })}
    </svg>
  );
};

const EtlView = ({ statsData, videosData }) => {
  const stats  = statsData  || {};
  const videos = videosData || [];
  const dist   = stats.class_distribution || [];
  const maxCount = dist.reduce((m, c) => Math.max(m, c.count), 1);

  const [selectedVideo, setSelectedVideo] = React.useState(null);
  const [selectedFrame, setSelectedFrame] = React.useState(null);
  const [frames,        setFrames]        = React.useState([]);
  const [annotations,   setAnnotations]   = React.useState([]);

  React.useEffect(() => {
    if (!selectedVideo) { setFrames([]); setSelectedFrame(null); setAnnotations([]); return; }
    fetchEtlFrames(selectedVideo.id).then(setFrames);
    setSelectedFrame(null);
    setAnnotations([]);
  }, [selectedVideo?.id]);

  React.useEffect(() => {
    if (!selectedFrame) { setAnnotations([]); return; }
    fetchEtlAnnotations(selectedFrame.id).then(setAnnotations);
  }, [selectedFrame?.id]);

  const summaryCards = [
    { label: 'Videos',      val: stats.total_videos      },
    { label: 'Frames',      val: stats.total_frames      },
    { label: 'Annotations', val: stats.total_annotations },
  ];

  return (
    <div style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Summary strip */}
      <FadeIn delay={0.1} from="top">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
          {summaryCards.map(({ label, val }, i) => (
            <FadeIn key={label} delay={0.15 + i * 0.08} from="bottom">
              <div style={{ padding: '14px 18px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 6, letterSpacing: '0.5px', textTransform: 'uppercase', fontWeight: 600 }}>{label}</div>
                <span style={{ fontSize: '28px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>
                  {val != null ? val.toLocaleString() : '—'}
                </span>
              </div>
            </FadeIn>
          ))}
        </div>
      </FadeIn>

      {/* Video table + class distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 260px', gap: 16 }}>

        {/* Video table */}
        <FadeIn delay={0.3} from="left">
          <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 12, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
              Videos — click a row to inspect
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 62px 46px 86px 58px 90px 76px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid var(--border-subtle)' }}>
              {['Filename', 'Duration', 'FPS', 'Resolution', 'Frames', 'Annotations', 'Status'].map(h => (
                <span key={h} style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontWeight: 600, letterSpacing: '0.4px', textTransform: 'uppercase' }}>{h}</span>
              ))}
            </div>
            <div style={{ maxHeight: 320, overflowY: 'auto' }}>
              {videos.length === 0
                ? <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '16px 4px' }}>No videos processed yet.</div>
                : videos.map((v, i) => {
                    const isSel = selectedVideo?.id === v.id;
                    return (
                      <FadeIn key={v.id} delay={0.35 + i * 0.06} from="bottom">
                        <div
                          onClick={() => setSelectedVideo(isSel ? null : v)}
                          style={{
                            display: 'grid', gridTemplateColumns: '1fr 62px 46px 86px 58px 90px 76px', gap: 8,
                            padding: '8px 4px', paddingLeft: 6, borderBottom: '1px solid var(--border-subtle)',
                            alignItems: 'center', cursor: 'pointer', transition: 'background 0.15s',
                            background: isSel ? 'rgba(0,212,160,0.06)' : 'transparent',
                            borderLeft: isSel ? '2px solid var(--accent)' : '2px solid transparent',
                          }}
                          onMouseEnter={e => { if (!isSel) e.currentTarget.style.background = 'rgba(255,255,255,0.025)'; }}
                          onMouseLeave={e => { if (!isSel) e.currentTarget.style.background = 'transparent'; }}
                        >
                          <span style={{ fontSize: '12px', color: isSel ? 'var(--accent)' : 'var(--text-primary)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v.filename}</span>
                          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.duration.toFixed(1)}s</span>
                          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.fps.toFixed(0)}</span>
                          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.width}×{v.height}</span>
                          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.frames}</span>
                          <span style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{v.annotations.toLocaleString()}</span>
                          <StatusBadge status={v.status}/>
                        </div>
                      </FadeIn>
                    );
                  })
              }
            </div>
          </div>
        </FadeIn>

        {/* Class distribution */}
        <FadeIn delay={0.3} from="right">
          <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 14, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
              Class Distribution
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {dist.map((c, i) => (
                <FadeIn key={c.class_name} delay={0.4 + i * 0.07} from="right">
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>{c.class_name}</span>
                      <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{c.count.toLocaleString()}</span>
                    </div>
                    <div style={{ height: 4, borderRadius: 999, background: 'var(--bg-pill)', overflow: 'hidden' }}>
                      <div style={{ height: '100%', borderRadius: 999, width: `${(c.count / maxCount) * 100}%`, background: 'var(--accent)', transition: 'width 0.9s cubic-bezier(0.34,1.56,0.64,1)' }}/>
                    </div>
                  </div>
                </FadeIn>
              ))}
              {dist.length === 0 && <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>No annotations yet.</span>}
            </div>
          </div>
        </FadeIn>
      </div>

      {/* Frame inspector — slides in when a video is selected */}
      {selectedVideo && (
        <FadeIn delay={0} from="bottom">
          <div style={{ borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid rgba(0,212,160,0.25)', overflow: 'hidden' }}>

            {/* Inspector header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '10px 16px', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(0,212,160,0.04)' }}>
              <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--accent)', fontFamily: 'monospace' }}>{selectedVideo.filename}</span>
              <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{frames.length} frames sampled</span>
              <button
                onClick={() => setSelectedVideo(null)}
                style={{ marginLeft: 'auto', fontSize: '11px', color: 'var(--text-tertiary)', background: 'none', border: 'none', cursor: 'pointer', fontFamily: 'inherit', padding: '2px 8px', borderRadius: 6, transition: 'color 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--text-tertiary)'}
              >✕ close</button>
            </div>

            {/* Inspector body: frame list + annotation detail */}
            <div style={{ display: 'grid', gridTemplateColumns: '190px 1fr', minHeight: 260 }}>

              {/* Frame list */}
              <div style={{ borderRight: '1px solid var(--border-subtle)', overflowY: 'auto', maxHeight: 440 }}>
                <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-tertiary)', padding: '10px 12px 6px', letterSpacing: '0.5px', textTransform: 'uppercase' }}>Frames</div>
                {frames.length === 0 && <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '12px' }}>No frames.</div>}
                {frames.map(f => {
                  const isSel = selectedFrame?.id === f.id;
                  return (
                    <div
                      key={f.id}
                      onClick={() => setSelectedFrame(isSel ? null : f)}
                      style={{
                        padding: '8px 12px', cursor: 'pointer', transition: 'background 0.12s',
                        background: isSel ? 'rgba(0,212,160,0.08)' : 'transparent',
                        borderLeft: isSel ? '2px solid var(--accent)' : '2px solid transparent',
                      }}
                      onMouseEnter={e => { if (!isSel) e.currentTarget.style.background = 'rgba(255,255,255,0.025)'; }}
                      onMouseLeave={e => { if (!isSel) e.currentTarget.style.background = 'transparent'; }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                        <span style={{ fontSize: '11.5px', fontWeight: 600, color: isSel ? 'var(--accent)' : 'var(--text-primary)', fontFamily: 'monospace' }}>#{f.frame_index}</span>
                        <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{f.annotation_count} ann</span>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>t={f.timestamp_s.toFixed(1)}s</span>
                        <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>Δ={f.scene_change_score.toFixed(2)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Annotation detail */}
              <div style={{ padding: '14px 16px', overflowY: 'auto', maxHeight: 440 }}>
                {!selectedFrame ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-tertiary)', fontSize: '12px' }}>
                    Select a frame to inspect annotations
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

                    {/* Bounding box overlay */}
                    <BoxOverlay annotations={annotations}/>

                    {/* Model vote table */}
                    <div>
                      <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 8, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
                        Model Votes — Frame #{selectedFrame.frame_index}
                      </div>
                      <div style={{ display: 'grid', gridTemplateColumns: '90px 44px 52px 68px 68px 110px', gap: 6, padding: '0 0 6px', borderBottom: '1px solid var(--border-subtle)', marginBottom: 4 }}>
                        {['Class', 'Votes', 'Score', 'YOLOv8', 'RT-DETR', 'Grounding DINO'].map(h => (
                          <span key={h} style={{ fontSize: '9px', color: 'var(--text-tertiary)', fontWeight: 600, letterSpacing: '0.4px', textTransform: 'uppercase' }}>{h}</span>
                        ))}
                      </div>
                      {annotations.map(ann => {
                        const color = annoColor(ann.class_name);
                        return (
                          <div key={ann.id} style={{ display: 'grid', gridTemplateColumns: '90px 44px 52px 68px 68px 110px', gap: 6, padding: '6px 0', borderBottom: '1px solid var(--border-subtle)', alignItems: 'center' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                              <div style={{ width: 6, height: 6, borderRadius: 1, background: color, flexShrink: 0 }}/>
                              <span style={{ fontSize: '11px', color: 'var(--text-primary)' }}>{ann.class_name}</span>
                            </div>
                            <span style={{ fontSize: '11px', fontWeight: 600, fontFamily: 'monospace', color: ann.votes === 3 ? 'var(--success)' : 'var(--warning)' }}>
                              {ann.votes}/3
                            </span>
                            <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                              {(ann.consensus_score * 100).toFixed(0)}%
                            </span>
                            {MODELS.map(m => (
                              <span key={m} style={{ fontSize: '11px', fontFamily: 'monospace', color: ann.model_confidences[m] != null ? 'var(--text-primary)' : 'var(--text-tertiary)' }}>
                                {ann.model_confidences[m] != null ? (ann.model_confidences[m] * 100).toFixed(0) + '%' : '—'}
                              </span>
                            ))}
                          </div>
                        );
                      })}
                      {annotations.length === 0 && (
                        <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '12px 0' }}>No annotation data for this frame.</div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </FadeIn>
      )}
    </div>
  );
};

const TASK_COLOR = {
  success:         '#65c16a',
  failed:          '#e84855',
  upstream_failed: '#e84855',
  running:         '#00d4a0',
  queued:          '#e88548',
  pending:         '#e88548',
  skipped:         '#494e4d',
};

const fmtOpsDur = s => s == null ? '—' : s >= 60 ? `${(s / 60).toFixed(1)}m` : `${s.toFixed(1)}s`;
const fmtOpsTime = iso => iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
const fmtOpsDate = iso => iso ? new Date(iso).toLocaleString('en-GB', { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' }) : '—';

// DAG graph — renders a horizontal chain of task boxes
const DagGraph = ({ tasks }) => {
  const boxW = 118, boxH = 54, arrowGap = 28, padX = 14, padY = 13;
  const svgW = Math.max(1, tasks.length * boxW + (tasks.length - 1) * arrowGap + padX * 2);
  const svgH = boxH + padY * 2;
  if (tasks.length === 0) {
    return <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '12px 4px' }}>No tasks for this run.</div>;
  }
  return (
    <div style={{ overflowX: 'auto' }}>
      <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} style={{ display: 'block' }}>
        {tasks.map((task, i) => {
          const x = padX + i * (boxW + arrowGap);
          const y = padY, cx = x + boxW / 2, cy = y + boxH / 2;
          const color = TASK_COLOR[task.state] || '#494e4d';
          return (
            <g key={task.task_id}>
              {i < tasks.length - 1 && (
                <g>
                  <line x1={x + boxW} y1={cy} x2={x + boxW + arrowGap - 6} y2={cy} stroke="rgba(255,255,255,0.12)" strokeWidth="1.5"/>
                  <polygon points={`${x+boxW+arrowGap-6},${cy-4} ${x+boxW+arrowGap-6},${cy+4} ${x+boxW+arrowGap},${cy}`} fill="rgba(255,255,255,0.12)"/>
                </g>
              )}
              <rect x={x} y={y} width={boxW} height={boxH} rx="8" fill={`${color}18`} stroke={`${color}55`} strokeWidth="1.5"/>
              <text x={cx} y={y + 17} textAnchor="middle" fill="rgba(232,234,233,0.85)" fontSize="9.5" fontFamily="monospace" fontWeight="600">{task.task_id}</text>
              <text x={cx} y={y + 31} textAnchor="middle" fill={color} fontSize="9" fontFamily="DM Sans, sans-serif" fontWeight="700" letterSpacing="0.3">{(task.state || '').toUpperCase()}</text>
              <text x={cx} y={y + 45} textAnchor="middle" fill="#494e4d" fontSize="9" fontFamily="monospace">{fmtOpsDur(task.duration)}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};

const OpsView = ({ airflowData, rayData, runsData }) => {
  const dag     = airflowData || {};
  const ray     = rayData       || {};
  const nodes   = ray.nodes     || [];
  const res     = ray.resources || {};
  const runs    = runsData      || [];

  // Selected historical run — null means "show the latest run from airflowData"
  const [selectedRun, setSelectedRun] = React.useState(null);
  const [runTasks,    setRunTasks]    = React.useState(null);

  React.useEffect(() => {
    if (!selectedRun) { setRunTasks(null); return; }
    fetchAirflowRunTasks(selectedRun.run_id).then(setRunTasks).catch(() => setRunTasks([]));
  }, [selectedRun?.run_id]);

  // Active run = selected historical run, or the latest run
  const activeRun   = selectedRun || dag.last_run || {};
  const activeTasks = selectedRun ? (runTasks || []) : (dag.tasks || []);
  const activeDur   = activeRun.start_date && activeRun.end_date
    ? (new Date(activeRun.end_date) - new Date(activeRun.start_date)) / 1000
    : (activeRun.duration ?? null);

  return (
    <div style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Airflow card */}
      <FadeIn delay={0.1} from="top">
        <div style={{ padding: '14px 18px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 18 }}>
            <div>
              <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-tertiary)', letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: 3 }}>
                Airflow DAG {selectedRun && <span style={{ color: 'var(--accent)' }}>· history</span>}
              </div>
              <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{dag.dag_id || '—'}</span>
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 20, alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>Schedule</div>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{dag.schedule || '—'}</span>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>{selectedRun ? 'Run' : 'Last Run'}</div>
                <StatusBadge status={activeRun.state || 'unknown'}/>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>Duration</div>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtOpsDur(activeDur)}</span>
              </div>
              {selectedRun && (
                <button onClick={() => setSelectedRun(null)}
                  style={{ fontSize: '11px', color: 'var(--text-tertiary)', background: 'none', border: '1px solid var(--border-subtle)', borderRadius: 6, padding: '4px 10px', cursor: 'pointer', fontFamily: 'inherit' }}
                  onMouseEnter={e => e.currentTarget.style.color = 'var(--text-primary)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--text-tertiary)'}
                >✕ latest</button>
              )}
            </div>
          </div>
          <DagGraph tasks={activeTasks}/>
        </div>
      </FadeIn>

      {/* Task table + Ray */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 260px', gap: 16 }}>

        {/* Task table */}
        <FadeIn delay={0.3} from="left">
          <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 12, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Task Runs</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 90px 64px 90px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid var(--border-subtle)' }}>
              {['Task', 'State', 'Duration', 'Start'].map(h => (
                <span key={h} style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontWeight: 600, letterSpacing: '0.4px', textTransform: 'uppercase' }}>{h}</span>
              ))}
            </div>
            {activeTasks.map((t, i) => (
              <FadeIn key={t.task_id} delay={0.35 + i * 0.07} from="bottom">
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 90px 64px 90px', gap: 8, padding: '8px 4px', borderBottom: '1px solid var(--border-subtle)', alignItems: 'center', transition: 'background 0.15s' }}
                  onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.025)'}
                  onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                >
                  <span style={{ fontSize: '12px', color: 'var(--text-primary)', fontFamily: 'monospace' }}>{t.task_id}</span>
                  <StatusBadge status={t.state}/>
                  <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtOpsDur(t.duration)}</span>
                  <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{fmtOpsTime(t.start_date)}</span>
                </div>
              </FadeIn>
            ))}
            {activeTasks.length === 0 && (
              <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '16px 4px' }}>No task runs.</div>
            )}
          </div>
        </FadeIn>

        {/* Ray cluster */}
        <FadeIn delay={0.3} from="right">
          <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', letterSpacing: '0.5px', textTransform: 'uppercase' }}>Ray Cluster</div>
              <StatusBadge status={ray.status || 'stopped'}/>
            </div>
            {[
              { label: 'CPU',    used: res.cpu_used,        total: res.cpu_total,        fmt: v => v.toFixed(1)         },
              { label: 'Memory', used: res.memory_used_gb,  total: res.memory_total_gb,  fmt: v => `${v.toFixed(1)} GB` },
            ].filter(r => r.total && r.used != null).map(r => (
              <div key={r.label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)' }}>{r.label}</span>
                  <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{r.fmt(r.used)} / {r.fmt(r.total)}</span>
                </div>
                <div style={{ height: 4, borderRadius: 999, background: 'var(--bg-pill)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', borderRadius: 999, width: `${(r.used / r.total) * 100}%`, background: 'var(--accent)', transition: 'width 0.9s ease' }}/>
                </div>
              </div>
            ))}
            <div style={{ borderTop: '1px solid var(--border-subtle)' }}/>
            <div>
              <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 10, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Nodes ({nodes.length})</div>
              {nodes.map((n, i) => (
                <FadeIn key={n.id} delay={0.5 + i * 0.07} from="right">
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 0', borderBottom: i < nodes.length - 1 ? '1px solid var(--border-subtle)' : 'none' }}>
                    <div style={{ width: 7, height: 7, borderRadius: '50%', flexShrink: 0, background: n.status === 'alive' ? '#65c16a' : '#e84855', boxShadow: n.status === 'alive' ? '0 0 6px rgba(101,193,106,0.5)' : 'none' }}/>
                    <span style={{ fontSize: '11.5px', color: 'var(--text-secondary)', fontFamily: 'monospace', flex: 1 }}>{n.ip || n.id}</span>
                    <span style={{ fontSize: '11px', color: 'var(--text-tertiary)' }}>{n.instance_type || (n.cpu_pct != null ? `${n.cpu_pct}% CPU` : '')}</span>
                  </div>
                </FadeIn>
              ))}
              {nodes.length === 0 && <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>No nodes active.</span>}
            </div>
            {ray.status === 'running' && ray.dashboard_url && (
              <PillButton onClick={() => window.open(ray.dashboard_url, '_blank')} style={{ width: '100%', justifyContent: 'center' }}>Ray Dashboard →</PillButton>
            )}
          </div>
        </FadeIn>
      </div>

      {/* Run history */}
      <FadeIn delay={0.4} from="bottom">
        <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 12, letterSpacing: '0.5px', textTransform: 'uppercase' }}>
            Run History — click a run to inspect
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 90px 80px 70px 120px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid var(--border-subtle)' }}>
            {['Run ID', 'State', 'Trigger', 'Duration', 'Started'].map(h => (
              <span key={h} style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontWeight: 600, letterSpacing: '0.4px', textTransform: 'uppercase' }}>{h}</span>
            ))}
          </div>
          <div style={{ maxHeight: 300, overflowY: 'auto' }}>
            {runs.length === 0 && <div style={{ fontSize: '12px', color: 'var(--text-tertiary)', padding: '16px 4px' }}>No runs yet.</div>}
            {runs.map((r, i) => {
              const isSel = selectedRun?.run_id === r.run_id;
              return (
                <FadeIn key={r.run_id} delay={0.45 + i * 0.04} from="bottom">
                  <div
                    onClick={() => setSelectedRun(isSel ? null : r)}
                    style={{
                      display: 'grid', gridTemplateColumns: '1fr 90px 80px 70px 120px', gap: 8,
                      padding: '8px 4px', paddingLeft: 6, borderBottom: '1px solid var(--border-subtle)', alignItems: 'center',
                      cursor: 'pointer', transition: 'background 0.15s',
                      background: isSel ? 'rgba(0,212,160,0.06)' : 'transparent',
                      borderLeft: isSel ? '2px solid var(--accent)' : '2px solid transparent',
                    }}
                    onMouseEnter={e => { if (!isSel) e.currentTarget.style.background = 'rgba(255,255,255,0.025)'; }}
                    onMouseLeave={e => { if (!isSel) e.currentTarget.style.background = 'transparent'; }}
                  >
                    <span style={{ fontSize: '11px', color: isSel ? 'var(--accent)' : 'var(--text-primary)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.run_id}</span>
                    <StatusBadge status={r.state}/>
                    <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{r.run_type || '—'}</span>
                    <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtOpsDur(r.duration)}</span>
                    <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{fmtOpsDate(r.start_date)}</span>
                  </div>
                </FadeIn>
              );
            })}
          </div>
        </div>
      </FadeIn>
    </div>
  );
};

const STAGE_LOGS = {
  lint: [
    '$ ruff check src/ --select E,W,F',
    'All checks passed.',
    '$ mypy src/ --ignore-missing-imports --strict',
    'Success: no issues found in 47 source files',
    '✓ Lint & type check complete (18.4s)',
  ],
  unit: [
    '$ pytest tests/unit/ -v --tb=short',
    'collected 32 items',
    '',
    'tests/unit/test_heads_tf.py::test_forward_pass PASSED',
    'tests/unit/test_heads_tf.py::test_output_shape PASSED',
    'tests/unit/test_priors_orch.py::test_prior_count PASSED',
    'tests/unit/test_priors_orch.py::test_prior_format PASSED',
    '',
    '32 passed in 42.1s',
  ],
  integration: [
    '$ pytest tests/integration/ -v --tb=short',
    'collected 8 items',
    '',
    'tests/integration/test_priors_orch.py::test_end_to_end PASSED',
    'tests/integration/test_etl_pipeline.py::test_frame_sampler PASSED',
    'tests/integration/test_etl_pipeline.py::test_consensus_engine RUNNING...',
  ],
};

const StagePipeline = ({ stages, selected, onSelect }) => {
  const fmtDur = s => s == null ? '' : s >= 60 ? `${(s/60).toFixed(1)}m` : `${s.toFixed(1)}s`;
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', overflowX: 'auto', padding: '6px 0 2px' }}>
      {stages.map((stage, i) => {
        const isLast = i === stages.length - 1;
        const isSel  = selected?.id === stage.id;
        const color  = TASK_COLOR[stage.status] || '#494e4d';
        const isPending = stage.status === 'pending';
        return (
          <div key={stage.id} style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
            <div onClick={() => onSelect(isSel ? null : stage)}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 82, cursor: 'pointer' }}>
              <div style={{
                width: 34, height: 34, borderRadius: '50%',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: isPending ? 'transparent' : `${color}1e`,
                border: `2px solid ${isPending ? 'rgba(255,255,255,0.10)' : color}`,
                boxShadow: isSel ? `0 0 0 3px ${color}33` : stage.status === 'running' ? `0 0 12px ${color}55` : 'none',
                animation: stage.status === 'running' ? 'pulse 2s ease-in-out infinite' : 'none',
                transition: 'all 0.2s',
              }}>
                {stage.status === 'success' && <span style={{ fontSize: 13, color }}>✓</span>}
                {stage.status === 'failed'  && <span style={{ fontSize: 13, color }}>✗</span>}
                {stage.status === 'running' && <span style={{ width: 8, height: 8, borderRadius: '50%', background: color, display: 'block' }}/>}
                {stage.status === 'pending' && <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'rgba(255,255,255,0.12)', display: 'block' }}/>}
              </div>
              <span style={{ fontSize: '10px', color: isSel ? 'var(--accent)' : 'var(--text-secondary)', marginTop: 6, textAlign: 'center', whiteSpace: 'nowrap', fontWeight: isSel ? 600 : 400 }}>
                {stage.name}
              </span>
              <span style={{ fontSize: '9px', color: 'var(--text-tertiary)', fontFamily: 'monospace', marginTop: 2, height: 12 }}>
                {fmtDur(stage.duration)}
              </span>
            </div>
            {!isLast && (
              <div style={{ width: 18, height: 2, flexShrink: 0, marginBottom: 22, background: stage.status === 'success' ? `${TASK_COLOR.success}50` : 'rgba(255,255,255,0.07)' }}/>
            )}
          </div>
        );
      })}
    </div>
  );
};

const DeployView = ({ cicdData, releasesData }) => {
  const run        = cicdData?.current_run || {};
  const recentRuns = cicdData?.recent_runs || [];
  const releases   = releasesData          || [];
  const [selectedStage, setSelectedStage] = React.useState(null);

  const fmtDur  = s => s >= 60 ? `${Math.floor(s/60)}m ${s%60|0}s` : `${s}s`;
  const fmtDate = iso => iso ? new Date(iso).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }) : '—';
  const fmtTime = iso => iso ? new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '—';

  return (
    <div style={{ padding: '20px 24px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* Current CI run */}
      <FadeIn delay={0.1} from="top">
        <div style={{ padding: '14px 18px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14, marginBottom: 18 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontSize: '10px', fontWeight: 600, color: 'var(--text-tertiary)', letterSpacing: '0.5px', textTransform: 'uppercase', marginBottom: 4 }}>Current Run</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                <span style={{ fontSize: '13px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{run.branch}</span>
                <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>@{run.commit}</span>
                <StatusBadge status={run.status}/>
              </div>
              <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{run.commit_message}</div>
            </div>
            <div style={{ display: 'flex', gap: 20, flexShrink: 0 }}>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>Trigger</div>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{run.trigger || '—'}</span>
              </div>
              <div>
                <div style={{ fontSize: '10px', color: 'var(--text-tertiary)', marginBottom: 3 }}>Started</div>
                <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{fmtTime(run.started_at)}</span>
              </div>
            </div>
          </div>
          <StagePipeline stages={run.stages || []} selected={selectedStage} onSelect={setSelectedStage}/>
        </div>
      </FadeIn>

      {/* Recent runs + stage log */}
      <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', gap: 16 }}>

        <FadeIn delay={0.3} from="left">
          <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 12, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Recent Runs</div>
            {recentRuns.map((r, i) => (
              <FadeIn key={r.id} delay={0.35 + i * 0.07} from="left">
                <div style={{ padding: '8px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3 }}>
                    <span style={{ fontSize: '11.5px', fontFamily: 'monospace', color: 'var(--text-secondary)' }}>{r.branch}</span>
                    <span style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>@{r.commit}</span>
                    <div style={{ marginLeft: 'auto' }}><StatusBadge status={r.status}/></div>
                  </div>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <span style={{ fontSize: '10px', color: 'var(--text-tertiary)' }}>{fmtDate(r.started_at)}</span>
                    <span style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{fmtDur(r.duration)}</span>
                  </div>
                </div>
              </FadeIn>
            ))}
          </div>
        </FadeIn>

        <FadeIn delay={0.3} from="right">
          <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', minHeight: 160, display: 'flex', flexDirection: 'column' }}>
            {!selectedStage ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-tertiary)', fontSize: '12px' }}>
                Click a stage above to view logs
              </div>
            ) : (
              <>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', letterSpacing: '0.5px', textTransform: 'uppercase' }}>{selectedStage.name}</div>
                  <StatusBadge status={selectedStage.status}/>
                </div>
                <div style={{ fontFamily: 'monospace', fontSize: '11px', lineHeight: 1.75, overflowY: 'auto', flex: 1 }}>
                  {(STAGE_LOGS[selectedStage.id] || []).length === 0
                    ? <span style={{ color: 'var(--text-tertiary)' }}>Stage not yet started.</span>
                    : (STAGE_LOGS[selectedStage.id]).map((line, i) => (
                        <div key={i} style={{
                          paddingLeft: line.startsWith('$') ? 0 : 10,
                          color: line.startsWith('$')  ? 'var(--accent)'
                               : line.startsWith('✓')  ? 'var(--success)'
                               : /PASSED$/.test(line)  ? 'var(--success)'
                               : /RUNNING/.test(line)  ? 'var(--warning)'
                               : line === ''            ? undefined
                               : 'var(--text-secondary)',
                        }}>{line || ' '}</div>
                      ))
                  }
                  {selectedStage.status === 'running' && (
                    <div style={{ color: 'var(--accent)', marginTop: 4 }}>▌</div>
                  )}
                </div>
              </>
            )}
          </div>
        </FadeIn>
      </div>

      {/* Release history */}
      <FadeIn delay={0.4} from="bottom">
        <div style={{ padding: '14px 16px', borderRadius: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-tertiary)', marginBottom: 14, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Release History</div>
          <div style={{ display: 'grid', gridTemplateColumns: '68px 82px 62px 1fr 90px 150px 100px', gap: 8, padding: '0 4px 8px', borderBottom: '1px solid var(--border-subtle)' }}>
            {['Version', 'Status', 'mAP', 'Experiment', 'Targets', 'Artifacts', 'Released'].map(h => (
              <span key={h} style={{ fontSize: '10px', color: 'var(--text-tertiary)', fontWeight: 600, letterSpacing: '0.4px', textTransform: 'uppercase' }}>{h}</span>
            ))}
          </div>
          {releases.map((rel, i) => (
            <FadeIn key={rel.version} delay={0.45 + i * 0.07} from="bottom">
              <div
                style={{ display: 'grid', gridTemplateColumns: '68px 82px 62px 1fr 90px 150px 100px', gap: 8, padding: '10px 4px', borderBottom: '1px solid var(--border-subtle)', alignItems: 'center', transition: 'background 0.15s' }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.025)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'monospace' }}>{rel.version}</span>
                <StatusBadge status={rel.status}/>
                <span style={{ fontSize: '12px', fontWeight: 600, fontFamily: 'monospace', color: rel.map_score >= 0.75 ? 'var(--success)' : 'var(--warning)' }}>
                  {(rel.map_score * 100).toFixed(1)}%
                </span>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{rel.experiment}</span>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                  {rel.targets.map(t => (
                    <span key={t} style={{ fontSize: '9px', fontWeight: 600, padding: '2px 6px', borderRadius: 999, background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.10)', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.3px' }}>{t}</span>
                  ))}
                </div>
                <div style={{ display: 'flex', gap: 10 }}>
                  {[['SM', rel.artifacts.saved_model], ['ONNX', rel.artifacts.onnx], ['TRT', rel.artifacts.tensorrt]].map(([label, ok]) => (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                      {ok ? <ShieldCheck size={11}/> : <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.15)' }}>✗</span>}
                      <span style={{ fontSize: '9px', color: ok ? 'var(--text-secondary)' : 'var(--text-tertiary)' }}>{label}</span>
                    </div>
                  ))}
                </div>
                <span style={{ fontSize: '11px', color: 'var(--text-tertiary)', fontFamily: 'monospace' }}>{fmtDate(rel.released_at)}</span>
              </div>
            </FadeIn>
          ))}
        </div>
      </FadeIn>
    </div>
  );
};

const Dashboard = () => {
  const [selectedExp, setSelectedExp]   = React.useState(null);
  const [statusFilter, setStatusFilter] = React.useState('all');
  const [viewMode, setViewMode]         = React.useState('pipeline');

  const { data: experiments, loading, refresh } = usePolling(fetchExperiments, 10000);

  const { data: metricsData } = usePolling(
    () => fetchMetrics(selectedExp?.experiment_id),
    30000,
    viewMode === 'metrics' && !!selectedExp
  );

  const { data: etlStats  } = usePolling(fetchEtlStats,  30000, viewMode === 'etl');
  const { data: etlVideos } = usePolling(fetchEtlVideos, 30000, viewMode === 'etl');

  const { data: airflowData } = usePolling(fetchAirflow,     15000, viewMode === 'ops');
  const { data: rayData     } = usePolling(fetchRay,         15000, viewMode === 'ops');
  const { data: airflowRuns } = usePolling(fetchAirflowRuns, 15000, viewMode === 'ops');

  const { data: cicdData     } = usePolling(fetchCicd,     30000, viewMode === 'deploy');
  const { data: releasesData } = usePolling(fetchReleases, 60000, viewMode === 'deploy');

  // Keep selectedExp synced with latest polled data
  React.useEffect(() => {
    if (!selectedExp || !experiments) return;
    const updated = experiments.find(
      e => e.experiment_id === selectedExp.experiment_id && e.fingerprint === selectedExp.fingerprint
    );
    if (updated) setSelectedExp(updated);
  }, [experiments]);

  const pipelineView = (
    <>
      <div style={{
        display: 'grid', gridTemplateColumns: '210px minmax(0,1fr) 210px',
        minHeight: 0, overflow: 'hidden',
      }}>
        <LeftPanel
          experiments={experiments} loading={loading}
          selectedExp={selectedExp} setSelectedExp={setSelectedExp}
          statusFilter={statusFilter} refresh={refresh}
        />
        <FadeIn delay={0.25} from="scale" style={{ padding: '10px 0', minHeight: 0, overflow: 'hidden' }}>
          <VisualizationPanel selectedExp={selectedExp}/>
        </FadeIn>
        <RightPanel selectedExp={selectedExp} refreshExps={refresh}/>
      </div>
      <BottomSection
        experiments={experiments}
        selectedExp={selectedExp}
        setSelectedExp={setSelectedExp}
      />
    </>
  );

  const metricsView = (
    <div style={{ minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      {!selectedExp
        ? <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1, color: 'var(--text-tertiary)', fontSize: '13px' }}>
            Select an experiment from the Pipeline tab first
          </div>
        : <MetricsView metricsData={metricsData} selectedExp={selectedExp}/>
      }
    </div>
  );

  const configView = (
    <div style={{ minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
      <ConfigView/>
    </div>
  );

  const etlView = (
    <div style={{ minHeight: 0, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      <EtlView statsData={etlStats} videosData={etlVideos}/>
    </div>
  );

  const opsView = (
    <div style={{ minHeight: 0, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      <OpsView airflowData={airflowData} rayData={rayData} runsData={airflowRuns}/>
    </div>
  );

  const deployView = (
    <div style={{ minHeight: 0, overflow: 'auto', display: 'flex', flexDirection: 'column' }}>
      <DeployView cicdData={cicdData} releasesData={releasesData}/>
    </div>
  );

  return (
    <div style={{
      display: 'grid',
      gridTemplateRows: viewMode === 'pipeline' ? 'auto 1fr auto' : 'auto 1fr',
      height: '100vh', width: '100vw', background: 'var(--bg-primary)',
      color: 'var(--text-primary)', overflow: 'hidden',
    }}>
      <Header
        statusFilter={statusFilter} setStatusFilter={setStatusFilter}
        viewMode={viewMode} setViewMode={setViewMode}
      />
      {viewMode === 'pipeline' ? pipelineView : viewMode === 'metrics' ? metricsView : viewMode === 'etl' ? etlView : viewMode === 'ops' ? opsView : viewMode === 'deploy' ? deployView : configView}
    </div>
  );
};

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accentHue": 165,
  "animSpeed": 1,
  "darkLevel": "deep"
}/*EDITMODE-END*/;

const App = () => {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);

  React.useEffect(() => {
    document.documentElement.style.setProperty('--accent', `oklch(0.72 0.17 ${t.accentHue})`);
  }, [t.accentHue]);

  React.useEffect(() => {
    const bg = t.darkLevel === 'deep' ? '#060a09' : t.darkLevel === 'darker' ? '#080c0b' : '#0c1110';
    document.documentElement.style.setProperty('--bg-primary', bg);
  }, [t.darkLevel]);

  return (
    <React.Fragment>
      <Dashboard/>
      <TweaksPanel title="Tweaks">
        <TweakSection label="Theme">
          <TweakRadio  label="Darkness"   value={t.darkLevel} onChange={v => setTweak('darkLevel', v)} options={['deep','darker','dark']}/>
          <TweakSlider label="Accent Hue" value={t.accentHue} onChange={v => setTweak('accentHue', v)} min={0} max={360} step={5}/>
        </TweakSection>
      </TweaksPanel>
    </React.Fragment>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App/>);
