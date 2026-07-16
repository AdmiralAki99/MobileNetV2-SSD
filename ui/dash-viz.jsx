/* dash-viz.jsx — World map visualization with AWS region markers */

const AWS_REGIONS = [
  { id: 'us-east-1',      name: 'N. Virginia',   lon: -77.0, lat:  39.0 },
  { id: 'us-east-2',      name: 'Ohio',           lon: -82.0, lat:  40.0 },
  { id: 'us-west-1',      name: 'N. California',  lon:-121.0, lat:  37.0 },
  { id: 'us-west-2',      name: 'Oregon',         lon:-123.0, lat:  44.0 },
  { id: 'ca-central-1',   name: 'Canada',         lon: -73.0, lat:  45.0 },
  { id: 'eu-west-1',      name: 'Ireland',        lon:  -8.0, lat:  53.0 },
  { id: 'eu-west-2',      name: 'London',         lon:  -0.1, lat:  51.5 },
  { id: 'eu-central-1',   name: 'Frankfurt',      lon:   8.0, lat:  50.0 },
  { id: 'eu-west-3',      name: 'Paris',          lon:   2.0, lat:  48.9 },
  { id: 'eu-north-1',     name: 'Stockholm',      lon:  18.0, lat:  59.3 },
  { id: 'ap-southeast-1', name: 'Singapore',      lon: 104.0, lat:   1.0 },
  { id: 'ap-northeast-1', name: 'Tokyo',          lon: 139.7, lat:  35.7 },
  { id: 'ap-south-1',     name: 'Mumbai',         lon:  72.9, lat:  19.1 },
  { id: 'ap-northeast-2', name: 'Seoul',          lon: 126.9, lat:  37.6 },
  { id: 'ap-southeast-2', name: 'Sydney',         lon: 151.2, lat: -33.9 },
  { id: 'sa-east-1',      name: 'São Paulo',      lon: -46.6, lat: -23.5 },
  { id: 'me-south-1',     name: 'Bahrain',        lon:  50.5, lat:  26.0 },
  { id: 'af-south-1',     name: 'Cape Town',      lon:  18.4, lat: -33.9 },
];

const WorldMapVisualization = ({ activeRegionId }) => {
  const [mapPaths, setMapPaths]       = React.useState({ countries: [], borders: '', graticule: '' });
  const [hovered, setHovered]         = React.useState(null);
  const projRef                       = React.useRef(null);
  const [projReady, setProjReady]     = React.useState(false);

  React.useEffect(() => {
    // Build the D3 Natural Earth projection fitted to a 1000×500 viewBox
    const projection = d3.geoNaturalEarth1()
      .scale(153)
      .translate([500, 250]);
    projRef.current = projection;

    const path = d3.geoPath().projection(projection);

    // Graticule (grid lines)
    const graticule = d3.geoGraticule().step([30, 30])();

    // Fetch Natural Earth 110m country data (public domain, ~100 KB)
    fetch('https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json')
      .then(r => r.json())
      .then(world => {
        const countries = topojson.feature(world, world.objects.countries);
        const borders   = topojson.mesh(world, world.objects.countries, (a, b) => a !== b);
        setMapPaths({
          countries: countries.features.map(f => path(f)),
          borders:   path(borders),
          graticule: path(graticule),
        });
        setProjReady(true);
      })
      .catch(err => console.error('Failed to load world atlas:', err));
  }, []);

  // Project a [lon, lat] pair to SVG [x, y] — only valid after projReady
  const project = (lon, lat) => {
    const pt = projRef.current ? projRef.current([lon, lat]) : null;
    return pt || [0, 0];
  };

  return (
    <svg
      viewBox="0 0 1000 500"
      width="100%" height="100%"
      style={{ display: 'block' }}
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Graticule */}
      <path
        d={mapPaths.graticule}
        fill="none"
        stroke="rgba(255,255,255,0.04)"
        strokeWidth="0.5"
      />

      {/* Country fills */}
      {mapPaths.countries.map((d, i) => (
        <path key={i} d={d} fill="rgba(255,255,255,0.035)" stroke="none"/>
      ))}

      {/* Country border mesh */}
      <path
        d={mapPaths.borders}
        fill="none"
        stroke="rgba(255,255,255,0.09)"
        strokeWidth="0.4"
      />

      {/* AWS region markers — only rendered once projection is ready */}
      {projReady && AWS_REGIONS.map(region => {
        const [x, y]  = project(region.lon, region.lat);
        const isActive  = region.id === activeRegionId;
        const isHovered = hovered === region.id;

        return (
          <g
            key={region.id}
            onMouseEnter={() => setHovered(region.id)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'default' }}
          >
            {/* Expanding pulse rings — active only */}
            {isActive && (
              <>
                <circle cx={x} cy={y} r="4" fill="none" stroke="var(--accent)" strokeWidth="0.8" opacity="0">
                  <animate attributeName="r"       from="4"  to="24" dur="2.4s" repeatCount="indefinite"/>
                  <animate attributeName="opacity" from="0.8" to="0"  dur="2.4s" repeatCount="indefinite"/>
                </circle>
                <circle cx={x} cy={y} r="4" fill="none" stroke="var(--accent)" strokeWidth="0.8" opacity="0">
                  <animate attributeName="r"       from="4"  to="24" dur="2.4s" begin="0.9s" repeatCount="indefinite"/>
                  <animate attributeName="opacity" from="0.8" to="0"  dur="2.4s" begin="0.9s" repeatCount="indefinite"/>
                </circle>
              </>
            )}

            {/* Marker dot */}
            <circle
              cx={x} cy={y}
              r={isActive ? 5 : isHovered ? 3.5 : 2.2}
              fill={isActive ? 'var(--accent)' : isHovered ? 'rgba(255,255,255,0.7)' : 'rgba(255,255,255,0.2)'}
              style={{
                transition: 'r 0.2s, fill 0.2s',
                filter: isActive ? 'drop-shadow(0 0 5px var(--accent))' : 'none',
              }}
            />

            {/* Hover tooltip */}
            {isHovered && !isActive && (
              <g>
                <rect
                  x={x + 8} y={y - 15}
                  width={region.id.length * 6.8 + 12} height={18}
                  rx="4"
                  fill="var(--bg-elevated)" stroke="var(--border-medium)" strokeWidth="0.7"
                />
                <text
                  x={x + 14} y={y - 2}
                  fontSize="9" fill="var(--text-secondary)"
                  fontFamily="'DM Sans', sans-serif" fontWeight="500"
                >{region.id}</text>
              </g>
            )}
          </g>
        );
      })}

      {/* Active region label — always visible */}
      {projReady && activeRegionId && (() => {
        const r = AWS_REGIONS.find(r => r.id === activeRegionId);
        if (!r) return null;
        const [x, y] = project(r.lon, r.lat);
        const labelW = r.name.length * 7 + r.id.length * 6.5 + 28;
        return (
          <g>
            <rect
              x={x + 10} y={y - 20}
              width={labelW} height={22} rx="5"
              fill="rgba(0,212,160,0.10)" stroke="rgba(0,212,160,0.30)" strokeWidth="0.8"
            />
            <text
              x={x + 16} y={y - 5}
              fontSize="10" fill="var(--accent)"
              fontFamily="'DM Sans', sans-serif" fontWeight="600"
            >{r.name} · {r.id}</text>
          </g>
        );
      })()}
    </svg>
  );
};

const FadeIn = ({ delay = 0, from = 'bottom', children, style: s }) => {
  const [visible, setVisible] = React.useState(false);
  React.useEffect(() => {
    const t = setTimeout(() => setVisible(true), delay * 1000);
    return () => clearTimeout(t);
  }, [delay]);
  const offsets = { bottom: [0, 14], top: [0, -14], left: [-14, 0], right: [14, 0], scale: [0, 0] };
  const [ox, oy] = offsets[from] || [0, 14];
  const tx = from === 'scale'
    ? (visible ? 'scale(1)' : 'scale(0.96)')
    : `translate(${visible ? 0 : ox}px, ${visible ? 0 : oy}px)`;
  return (
    <div style={{
      opacity: visible ? 1 : 0, transform: tx,
      transition: 'opacity 0.55s ease, transform 0.55s ease', ...s,
    }}>{children}</div>
  );
};

const DetailRow = ({ label, value }) => (
  <div style={{
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '6px 10px', borderRadius: 8,
    background: 'rgba(255,255,255,0.04)',
    border: '1px solid rgba(255,255,255,0.06)',
  }}>
    <span style={{ fontSize: '10px', color: 'var(--text-tertiary)', minWidth: 52, flexShrink: 0 }}>{label}</span>
    <span style={{
      flex: 1, fontSize: '11.5px', color: 'var(--text-secondary)',
      fontFamily: 'monospace', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
    }}>{value || '—'}</span>
  </div>
);

const TrainingStatusPill = ({ status }) => {
  const color = {
    running: 'var(--accent)', success: 'var(--success)',
    failed: 'var(--danger)', pending: 'var(--warning)',
  }[status] || 'var(--text-tertiary)';
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 8, padding: '8px 16px',
      borderRadius: 999, background: 'rgba(20,25,23,0.9)', backdropFilter: 'blur(12px)',
      border: `1px solid ${color}44`,
    }}>
      <div style={{
        width: 6, height: 6, borderRadius: '50%', background: color,
        boxShadow: `0 0 6px ${color}`,
      }}/>
      <span style={{
        fontSize: '12px', color, textTransform: 'uppercase',
        letterSpacing: '0.5px', fontWeight: 600,
      }}>
        {status || 'no selection'}
      </span>
    </div>
  );
};

const MapToolbar = () => {
  const [active, setActive] = React.useState(0);
  const labels = ['Regions', 'Graticule', '360°'];
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 2, padding: '5px 8px',
      borderRadius: 999, background: 'rgba(20,25,23,0.85)', backdropFilter: 'blur(12px)',
      border: '1px solid var(--border-subtle)',
    }}>
      {labels.map((l, i) => (
        <button key={l} onClick={() => setActive(i)} style={{
          padding: '4px 14px', borderRadius: 999, border: 'none',
          background: i === active ? '#fff' : 'transparent',
          color: i === active ? '#0a0e0d' : 'var(--text-secondary)',
          fontSize: '11px', fontFamily: 'inherit', fontWeight: 600,
          cursor: 'pointer', transition: 'all 0.2s',
        }}>{l}</button>
      ))}
    </div>
  );
};

const VisualizationPanel = ({ selectedExp }) => (
  <div style={{
    position: 'relative', width: '100%', height: '100%',
    display: 'flex', flexDirection: 'column', overflow: 'hidden', minWidth: 0,
  }}>
    {/* Map area */}
    <div style={{
      flex: 1, position: 'relative', minHeight: 0, overflow: 'hidden',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '12px',
    }}>
      <WorldMapVisualization activeRegionId={selectedExp?.region || null}/>

      {/* Glass card overlay — top right */}
      <FadeIn delay={0.6} from="right" style={{
        position: 'absolute', top: 14, right: 12, maxWidth: 248,
      }}>
        <div style={{
          background: 'rgba(8,13,12,0.78)',
          backdropFilter: 'blur(18px)',
          WebkitBackdropFilter: 'blur(18px)',
          border: '1px solid rgba(255,255,255,0.08)',
          borderRadius: 14,
          padding: '12px 14px',
          display: 'flex', flexDirection: 'column', gap: 8,
          boxShadow: '0 8px 32px rgba(0,0,0,0.45)',
        }}>
          {/* Status header */}
          <TrainingStatusPill status={selectedExp?.status}/>

          {/* Detail rows — only when an experiment is selected */}
          {selectedExp && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              <DetailRow label="id"       value={selectedExp.experiment_id}/>
              <DetailRow label="run"      value={selectedExp.fingerprint}/>
              <DetailRow label="config"   value={selectedExp.config_filename}/>
              <DetailRow label="instance" value={selectedExp.ec2_instance}/>
              <DetailRow label="region"   value={selectedExp.region}/>
              <DetailRow label="steps"    value={selectedExp.total_steps != null ? String(selectedExp.total_steps) : null}/>
            </div>
          )}
        </div>
      </FadeIn>
    </div>

    {/* Toolbar */}
    <FadeIn delay={0.7} from="bottom" style={{ display: 'flex', justifyContent: 'center', padding: '10px 0 6px' }}>
      <MapToolbar/>
    </FadeIn>
  </div>
);

Object.assign(window, { VisualizationPanel, FadeIn });
