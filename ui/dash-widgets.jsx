/* dash-widgets.jsx — Icons, pills, gauges, charts, badges, stat cards */
const { useState, useRef, useEffect, useMemo, useCallback } = React;

const Icon = ({ children, size = 16, ...props }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" {...props}>{children}</svg>
);

const ChevronDown = ({ size = 12 }) => <Icon size={size}><path d="M4 6l4 4 4-4"/></Icon>;
const ChevronUp = ({ size = 12 }) => <Icon size={size}><path d="M4 10l4-4 4 4"/></Icon>;
const ArrowLeft = ({ size = 16 }) => <Icon size={size}><path d="M10 3L5 8l5 5"/></Icon>;
const ArrowUp = ({ size = 14 }) => <Icon size={size}><path d="M8 12V4M4 7l4-3 4 3"/></Icon>;
const SearchSvg = ({ size = 15 }) => <Icon size={size}><circle cx="6.8" cy="6.8" r="4.2"/><path d="M10 10l3.5 3.5"/></Icon>;
const GearSvg = ({ size = 17 }) => <Icon size={size} strokeWidth="1.2"><circle cx="8" cy="8" r="2.2"/><path d="M8 2v1.5M8 12.5V14M2 8h1.5M12.5 8H14M3.8 3.8l1 1M11.2 11.2l1 1M3.8 12.2l1-1M11.2 4.8l1-1"/></Icon>;
const RefreshSvg = ({ size = 14 }) => <Icon size={size}><path d="M2.5 8a5.5 5.5 0 019.2-3M13.5 8a5.5 5.5 0 01-9.2 3"/><path d="M11.7 2v3h-3M4.3 14v-3h3" strokeWidth="1.2"/></Icon>;
const MoreSvg = ({ size = 14 }) => <Icon size={size} fill="currentColor" stroke="none"><circle cx="3" cy="8" r="1.2"/><circle cx="8" cy="8" r="1.2"/><circle cx="13" cy="8" r="1.2"/></Icon>;
const PlusSvg = ({ size = 14 }) => <Icon size={size}><path d="M8 3v10M3 8h10"/></Icon>;
const CloseSvg = ({ size = 12 }) => <Icon size={size}><path d="M3 3l10 10M13 3L3 13"/></Icon>;
const FilterSvg = ({ size = 14 }) => <Icon size={size}><path d="M2 4h12M4 8h8M6 12h4"/></Icon>;
const CopySvg = ({ size = 14 }) => <Icon size={size}><rect x="5" y="5" width="8" height="8" rx="1.5"/><path d="M3 11V3h8"/></Icon>;
const UserSvg = ({ size = 14 }) => <Icon size={size}><circle cx="8" cy="5" r="2.5"/><path d="M3 14c0-2.8 2.2-5 5-5s5 2.2 5 5"/></Icon>;
const LayersSvg = ({ size = 14 }) => <Icon size={size}><path d="M8 2L2 5.5 8 9l6-3.5L8 2z"/><path d="M2 8.5L8 12l6-3.5" strokeWidth="1.2"/><path d="M2 11.5L8 15l6-3.5" strokeWidth="1.2"/></Icon>;

// Alert icons (filled)
const ShieldCheck = ({ size = 16, color = '#65c16a' }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <path d="M8 1.5L2.5 4v3.5c0 3.8 2.3 6.2 5.5 7 3.2-.8 5.5-3.2 5.5-7V4L8 1.5z" fill={color} opacity="0.9"/>
    <path d="M6 8.5l1.5 1.5L10 7" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const DangerTriangle = ({ size = 16, color = '#e84855' }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <path d="M8 2L1.5 13.5h13L8 2z" fill={color} opacity="0.9"/>
    <path d="M8 6.5v3" stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
    <circle cx="8" cy="11.2" r="0.7" fill="#fff"/>
  </svg>
);

const AlertCircle = ({ size = 16, color = '#e88548' }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <circle cx="8" cy="8" r="6.5" fill={color} opacity="0.9"/>
    <path d="M8 5v3.5" stroke="#fff" strokeWidth="1.4" strokeLinecap="round"/>
    <circle cx="8" cy="10.8" r="0.7" fill="#fff"/>
  </svg>
);

const UnknownSquare = ({ size = 16 }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none">
    <rect x="2.5" y="2.5" width="11" height="11" rx="2" fill="#555" opacity="0.7"/>
    <path d="M6.5 6a1.5 1.5 0 013 0c0 1-1.5 1-1.5 2" stroke="#fff" strokeWidth="1.2" strokeLinecap="round"/>
    <circle cx="8" cy="10.5" r="0.6" fill="#fff"/>
  </svg>
);

const pillBase = {
  display: 'inline-flex', alignItems: 'center', gap: '6px',
  padding: '7px 14px', borderRadius: '999px', fontSize: '12.5px',
  fontWeight: 500, cursor: 'pointer', transition: 'all 0.25s ease',
  whiteSpace: 'nowrap', fontFamily: 'inherit', outline: 'none',
  lineHeight: 1.2,
};

const PillButton = ({ children, active, onClick, style: s }) => (
  <button onClick={onClick} style={{
    ...pillBase,
    border: active ? '1px solid rgba(255,255,255,0.15)' : '1px solid var(--border-subtle)',
    background: active ? '#fff' : 'var(--bg-pill)',
    color: active ? '#0a0e0d' : 'var(--text-secondary)',
    ...s,
  }}>{children}</button>
);

const PillDropdown = ({ label, value, options = [], style: s }) => {
  const [open, setOpen] = useState(false);
  const [val, setVal] = useState(value);
  const ref = useRef(null);

  useEffect(() => {
    const close = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener('mousedown', close);
    return () => document.removeEventListener('mousedown', close);
  }, []);

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
        <div style={{
          position: 'absolute', top: '110%', left: 0, zIndex: 50,
          background: 'var(--bg-elevated)', border: '1px solid var(--border-medium)',
          borderRadius: 12, padding: '4px 0', minWidth: 140,
          boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
          animation: 'fadeIn 0.15s ease',
        }}>
          {options.map(o => (
            <div key={o} onClick={() => { setVal(o); setOpen(false); }} style={{
              padding: '8px 14px', fontSize: '12.5px', cursor: 'pointer',
              color: o === val ? 'var(--accent)' : 'var(--text-secondary)',
              background: o === val ? 'rgba(0,212,160,0.06)' : 'transparent',
              transition: 'background 0.15s',
            }} onMouseEnter={e => e.target.style.background = 'rgba(255,255,255,0.04)'}
               onMouseLeave={e => e.target.style.background = o === val ? 'rgba(0,212,160,0.06)' : 'transparent'}>
              {o}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const SearchBar = () => {
  const [focused, setFocused] = useState(false);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '7px 14px', borderRadius: 999, minWidth: 180,
      background: 'var(--bg-pill)',
      border: `1px solid ${focused ? 'var(--accent)' : 'var(--border-subtle)'}`,
      transition: 'border-color 0.25s',
    }}>
      <SearchSvg />
      <input
        placeholder="Search by Name etc"
        onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}
        style={{
          background: 'none', border: 'none', outline: 'none', color: 'var(--text-primary)',
          fontSize: '12.5px', fontFamily: 'inherit', width: '100%',
        }}
      />
    </div>
  );
};

const TabGroup = ({ tabs, active, onChange }) => (
  <div style={{
    display: 'flex', gap: 2, padding: 3, borderRadius: 999,
    background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)',
  }}>
    {tabs.map(t => (
      <button key={t} onClick={() => onChange(t)} style={{
        ...pillBase, padding: '6px 16px', fontSize: '12px',
        border: 'none',
        background: t === active ? '#fff' : 'transparent',
        color: t === active ? '#0a0e0d' : 'var(--text-secondary)',
      }}>{t}</button>
    ))}
  </div>
);

const GaugeChart = ({ value, max, size = 90, greenPct = 0.7, label, color = 'green' }) => {
  const [animVal, setAnimVal] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimVal(value), 400);
    return () => clearTimeout(t);
  }, [value]);

  const cx = 55, cy = 55, r = 42;
  const startA = 135, totalA = 270;
  const pct = Math.min(animVal / max, 1);
  const toRad = d => (d - 90) * Math.PI / 180;
  const ptOn = (a) => ({ x: cx + r * Math.cos(toRad(a)), y: cy + r * Math.sin(toRad(a)) });

  const makeArc = (from, to) => {
    const s = ptOn(to), e = ptOn(from);
    const large = to - from > 180 ? 1 : 0;
    return `M${s.x} ${s.y} A${r} ${r} 0 ${large} 0 ${e.x} ${e.y}`;
  };

  const needleA = startA + pct * totalA;
  const nTip = ptOn(needleA);
  const greenEnd = startA + greenPct * totalA;

  return (
    <div style={{ textAlign: 'center' }}>
      <svg width={size} height={size * 0.75} viewBox="0 0 110 82">
        {/* Background track */}
        <path d={makeArc(startA, startA + totalA)} fill="none" stroke="var(--bg-pill)" strokeWidth="5" strokeLinecap="round"/>
        {/* Green arc */}
        <path d={makeArc(startA, greenEnd)} fill="none" stroke="#65c16a" strokeWidth="5" strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 1s ease' }}/>
        {/* Red arc */}
        <path d={makeArc(greenEnd, startA + totalA)} fill="none" stroke="#e84855" strokeWidth="5" strokeLinecap="round"/>
        {/* Needle line */}
        <line x1={cx} y1={cy} x2={nTip.x} y2={nTip.y}
          stroke="var(--text-primary)" strokeWidth="1.5" strokeLinecap="round"
          style={{ transition: 'all 1.2s cubic-bezier(0.34,1.56,0.64,1)' }}/>
        <circle cx={cx} cy={cy} r="3" fill="var(--bg-surface)" stroke="var(--text-tertiary)" strokeWidth="1"/>
      </svg>
    </div>
  );
};

const StatCard = ({ label, value, total, gauge, greenPct = 0.7 }) => {
  const [show, setShow] = useState(false);
  useEffect(() => { setTimeout(() => setShow(true), 600); }, []);
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10, padding: '8px 0',
      opacity: show ? 1 : 0, transform: show ? 'translateX(0)' : 'translateX(12px)',
      transition: 'all 0.6s ease',
    }}>
      {gauge && <GaugeChart value={parseInt(value.replace(/[^0-9]/g,''))} max={parseInt(total.replace(/[^0-9]/g,''))} size={80} greenPct={greenPct}/>}
      <div>
        <div style={{ fontSize: '11px', color: 'var(--text-tertiary)', marginBottom: 2 }}>{label}</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 4 }}>
          <span style={{ fontSize: '28px', fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.5px' }}>{value}</span>
          <span style={{ fontSize: '12px', color: 'var(--text-tertiary)' }}>/{total}</span>
        </div>
      </div>
    </div>
  );
};

const BarChart = ({ width = 260, height = 80 }) => {
  const bars = useMemo(() => {
    const b = [];
    for (let i = 0; i < 40; i++) {
      const h = 5 + Math.random() * 55 + (i > 15 && i < 30 ? Math.random() * 20 : 0);
      b.push(h);
    }
    return b;
  }, []);
  const bw = width / bars.length - 1.5;

  return (
    <div style={{ marginBottom: 8 }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {bars.map((h, i) => (
          <rect key={i} x={i * (bw + 1.5)} y={height - h} width={bw} height={h}
            fill="rgba(255,255,255,0.7)" rx="0.5"
            style={{ animation: `barGrow 0.8s ease ${i * 0.02}s both` }}/>
        ))}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', color: 'var(--text-tertiary)', marginTop: 4 }}>
        <span>316</span><span>564</span>
      </div>
    </div>
  );
};

const TriangleChart = ({ width = 260, height = 55 }) => {
  const tris = useMemo(() => {
    const sizes = [12,8,18,6,10,25,5,14,7,22,9,30,8,11,16];
    let x = 5;
    return sizes.map(s => {
      const t = { x, size: s };
      x += s * 0.9 + 3;
      return t;
    });
  }, []);

  return (
    <div style={{ marginBottom: 8 }}>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {/* Base line */}
        <line x1="0" y1={height - 5} x2={width} y2={height - 5} stroke="rgba(255,255,255,0.1)" strokeWidth="1"/>
        {/* Triangles */}
        {tris.map((t, i) => {
          const bx = t.x, by = height - 5;
          return (
            <polygon key={i}
              points={`${bx},${by} ${bx + t.size / 2},${by - t.size * 1.2} ${bx + t.size},${by}`}
              fill="rgba(255,255,255,0.75)"
              style={{ animation: `fadeIn 0.5s ease ${i * 0.05}s both` }}/>
          );
        })}
        {/* Data points on line */}
        {[40, 100, 160, 220].map((x, i) => (
          <circle key={i} cx={x} cy={height - 5} r="2" fill="var(--text-primary)"/>
        ))}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10.5px', color: 'var(--text-tertiary)', marginTop: 4 }}>
        <span>72</span><span>50</span><span>100</span>
      </div>
    </div>
  );
};

const AlertBadge = ({ type = 'protected', style: s, animate = true }) => {
  const config = {
    protected: { icon: <ShieldCheck size={18}/>, bg: 'rgba(101,193,106,0.15)', border: 'rgba(101,193,106,0.3)', glow: 'rgba(101,193,106,0.2)' },
    danger:    { icon: <DangerTriangle size={18}/>, bg: 'rgba(232,72,85,0.15)', border: 'rgba(232,72,85,0.3)', glow: 'rgba(232,72,85,0.2)' },
    alert:     { icon: <AlertCircle size={18}/>, bg: 'rgba(232,133,72,0.15)', border: 'rgba(232,133,72,0.3)', glow: 'rgba(232,133,72,0.2)' },
  }[type];

  return (
    <div style={{
      width: 34, height: 34, borderRadius: '50%',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: config.bg, border: `1px solid ${config.border}`,
      boxShadow: `0 0 12px ${config.glow}`,
      animation: animate ? `pulse 3s ease-in-out infinite` : 'none',
      cursor: 'pointer', transition: 'transform 0.2s',
      ...s,
    }}>{config.icon}</div>
  );
};

const UserRow = ({ name }) => (
  <div style={{
    display: 'flex', alignItems: 'center', gap: 8, padding: '7px 14px',
    background: 'var(--bg-surface)', borderRadius: 10,
    border: '1px solid var(--border-subtle)', cursor: 'pointer',
    transition: 'border-color 0.2s',
  }} onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--border-medium)'}
     onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border-subtle)'}>
    <UserSvg size={13}/>
    <span style={{ flex: 1, fontSize: '12.5px', color: 'var(--text-secondary)' }}>{name}</span>
    <ChevronDown/>
  </div>
);

const DotGrid = ({ cols = 16, rows = 6, width = 400, height = 160 }) => {
  const gapX = width / cols, gapY = height / rows;
  const highlights = useMemo(() => [
    { r: 1, c: 2, type: 'protected' }, { r: 0, c: 6, type: 'protected' },
    { r: 2, c: 4, type: 'alert' }, { r: 3, c: 8, type: 'danger' },
    { r: 1, c: 10, type: 'alert' }, { r: 4, c: 3, type: 'danger' },
    { r: 2, c: 12, type: 'protected' }, { r: 5, c: 7, type: 'alert' },
  ], []);

  return (
    <div style={{ position: 'relative', width, height }}>
      <svg width={width} height={height} style={{ position: 'absolute', top: 0, left: 0 }}>
        {Array.from({ length: rows }).map((_, r) =>
          Array.from({ length: cols }).map((_, c) => (
            <circle key={`${r}-${c}`} cx={gapX * (c + 0.5)} cy={gapY * (r + 0.5)} r="2"
              fill="rgba(255,255,255,0.07)" style={{ animation: `fadeIn 0.3s ease ${(r * cols + c) * 0.005}s both` }}/>
          ))
        )}
        {/* Selection rectangle */}
        <rect x={gapX * 1} y={gapY * 0.5} width={gapX * 8} height={gapY * 4.5} rx="8"
          fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="1" strokeDasharray="4 3"/>
      </svg>
      {highlights.map((h, i) => (
        <div key={i} style={{
          position: 'absolute',
          left: gapX * (h.c + 0.5) - 12,
          top: gapY * (h.r + 0.5) - 12,
        }}>
          <AlertBadge type={h.type} animate={false} style={{ width: 24, height: 24 }}/>
        </div>
      ))}
    </div>
  );
};

// Export to window
Object.assign(window, {
  ChevronDown, ChevronUp, ArrowLeft, ArrowUp, SearchSvg, GearSvg,
  RefreshSvg, MoreSvg, PlusSvg, CloseSvg, FilterSvg, CopySvg, UserSvg, LayersSvg,
  ShieldCheck, DangerTriangle, AlertCircle, UnknownSquare,
  PillButton, PillDropdown, SearchBar, TabGroup,
  GaugeChart, StatCard, BarChart, TriangleChart,
  AlertBadge, UserRow, DotGrid,
});
