/* dash-charts.jsx — Metrics bento grid */

const _rng = (() => { let s=42; return () => { s=(s*1664525+1013904223)&0xffffffff; return (s>>>0)/0xffffffff; }; })();
const _curve = (fn,noise,n=200) => Array.from({length:n},(_,i)=>Math.max(0,fn(i+1)+(_rng()-0.5)*noise));

const MOCK_TRAIN_LOSS = _curve(e=>4.5*Math.exp(-e/45)+0.82,0.18);
const MOCK_VAL_LOSS   = _curve(e=>4.8*Math.exp(-e/50)+0.94,0.28);
const MOCK_MAP_CURVE  = _curve(e=>0.766/(1+Math.exp(-(e-80)/22)),0.018);

const VOC_CLASSES = [
  'aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
  'chair','cow','diningtable','dog','horse','motorbike','person',
  'pottedplant','sheep','sofa','train','tvmonitor',
];
const MOCK_CLASS_AP = {
  aeroplane:0.812,bicycle:0.791,bird:0.734,boat:0.623,bottle:0.423,
  bus:0.847,car:0.812,cat:0.883,chair:0.512,cow:0.741,
  diningtable:0.688,dog:0.871,horse:0.856,motorbike:0.803,person:0.847,
  pottedplant:0.489,sheep:0.762,sofa:0.692,train:0.851,tvmonitor:0.735,
};
const MOCK_CONF = (() => {
  const N=21,m=Array.from({length:N},()=>Array(N).fill(0));
  for(let r=0;r<N;r++){m[r][r]=80+(_rng()*40|0);for(let c=0;c<N;c++)if(c!==r&&_rng()<0.16)m[r][c]=(_rng()*12|0);}
  return m;
})();
const BOX_COLORS=['#00d4a0','#e88548','#7c9ef5','#e84855','#65c16a','#c97cf5','#f5ea7c'];
const MOCK_IMAGES=Array.from({length:8},(_,i)=>({
  id:i,label:VOC_CLASSES[i%20],
  boxes:Array.from({length:1+i%3},(_,b)=>({
    x:0.06+_rng()*0.40,y:0.10+_rng()*0.30,w:0.20+_rng()*0.26,h:0.20+_rng()*0.24,
    cls:VOC_CLASSES[(i+b)%20],score:0.56+_rng()*0.43,
  })),
}));

const card = {
  background:'#0f1714',
  border:'1px solid rgba(255,255,255,0.07)',
  borderRadius:16,
  overflow:'hidden',
  transition:'border-color 0.25s',
};
const cardHover = e => { e.currentTarget.style.borderColor='rgba(255,255,255,0.14)'; };
const cardLeave = e => { e.currentTarget.style.borderColor='rgba(255,255,255,0.07)'; };

const CardLabel = ({children}) => (
  <span style={{fontSize:'10px',fontWeight:600,letterSpacing:'0.7px',textTransform:'uppercase',color:'var(--text-tertiary)'}}>
    {children}
  </span>
);

const StatTile = ({label,value,sub,accentColor,sparkData,inverted=false}) => {
  const W=80,H=28,pad=2;
  const vals = sparkData ? sparkData.slice(-30) : [];
  let spark=null;
  if(vals.length>1){
    const mn=Math.min(...vals),mx=Math.max(...vals);
    const xSc=i=>(i/(vals.length-1))*(W-pad*2)+pad;
    const ySc=v=>H-pad-(mx===mn?H/2:(v-mn)/(mx-mn)*(H-pad*2));
    const pts=vals.map((v,i)=>`${xSc(i)},${ySc(v)}`).join(' ');
    const area=`M${xSc(0)},${H} `+vals.map((v,i)=>`L${xSc(i)},${ySc(v)}`).join(' ')+` L${xSc(vals.length-1)},${H} Z`;
    spark=(
      <svg width={W} height={H} style={{position:'absolute',bottom:0,right:0,opacity:0.25}} viewBox={`0 0 ${W} ${H}`}>
        <defs>
          <linearGradient id={`sg${label}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={accentColor||'var(--accent)'} stopOpacity="1"/>
            <stop offset="100%" stopColor={accentColor||'var(--accent)'} stopOpacity="0"/>
          </linearGradient>
        </defs>
        <path d={area} fill={`url(#sg${label})`}/>
        <polyline points={pts} fill="none" stroke={accentColor||'var(--accent)'} strokeWidth="1.2"/>
      </svg>
    );
  }
  return (
    <div style={{...card,position:'relative',padding:'16px 18px',display:'flex',flexDirection:'column',gap:4}}
      onMouseEnter={cardHover} onMouseLeave={cardLeave}>
      {spark}
      <CardLabel>{label}</CardLabel>
      <span style={{fontSize:'28px',fontWeight:700,color:accentColor||'var(--text-primary)',letterSpacing:'-1px',lineHeight:1.1,fontVariantNumeric:'tabular-nums'}}>
        {value}
      </span>
      {sub && <span style={{fontSize:'10px',color:'var(--text-tertiary)',marginTop:1}}>{sub}</span>}
    </div>
  );
};

const VW=640,VH=240,LP={top:24,right:18,bottom:38,left:52};

const LineChart = ({title,sub,series}) => {
  const [hover,setHover] = React.useState(null);
  const COLS = ['#00d4a0','#e88548','#7c9ef5'];

  const allVals = series.flatMap(s=>s.data);
  const xSc = d3.scaleLinear().domain([1,200]).range([LP.left,VW-LP.right]);
  const pad = (Math.max(...allVals)-Math.min(...allVals))*0.07;
  const ySc = d3.scaleLinear().domain([Math.min(...allVals)-pad,Math.max(...allVals)+pad]).range([VH-LP.bottom,LP.top]);

  const mkLine = d3.line().x((_,i)=>xSc(i+1)).y(d=>ySc(d)).curve(d3.curveCatmullRom.alpha(0.5));
  const mkArea = d3.area().x((_,i)=>xSc(i+1)).y0(VH-LP.bottom).y1(d=>ySc(d)).curve(d3.curveCatmullRom.alpha(0.5));

  const xTicks=[1,40,80,120,160,200];
  const yTicks=ySc.ticks(5);

  const onMove=e=>{
    const r=e.currentTarget.getBoundingClientRect();
    const svgX=(e.clientX-r.left)/r.width*VW;
    if(svgX<LP.left||svgX>VW-LP.right){setHover(null);return;}
    const epoch=Math.max(1,Math.min(200,Math.round(xSc.invert(svgX))));
    setHover({epoch,cx:xSc(epoch),pct:xSc(epoch)/VW,vals:series.map((s,i)=>({label:s.label,v:s.data[epoch-1],col:COLS[i]}))});
  };

  return (
    <div style={{...card,padding:'20px 22px 14px',display:'flex',flexDirection:'column',gap:14}}
      onMouseEnter={cardHover} onMouseLeave={cardLeave}>
      <div style={{display:'flex',alignItems:'baseline',justifyContent:'space-between'}}>
        <div style={{display:'flex',flexDirection:'column',gap:2}}>
          <span style={{fontSize:'13px',fontWeight:600,color:'var(--text-primary)'}}>{title}</span>
          {sub&&<span style={{fontSize:'10px',color:'var(--text-tertiary)'}}>{sub}</span>}
        </div>
        <div style={{display:'flex',gap:14}}>
          {series.map((s,i)=>(
            <div key={i} style={{display:'flex',alignItems:'center',gap:6}}>
              <div style={{width:16,height:2,background:COLS[i],borderRadius:2,boxShadow:`0 0 6px ${COLS[i]}88`}}/>
              <span style={{fontSize:'10px',color:'var(--text-tertiary)'}}>{s.label}</span>
            </div>
          ))}
        </div>
      </div>
      <div style={{position:'relative'}}>
        <svg viewBox={`0 0 ${VW} ${VH}`} width="100%" style={{display:'block',cursor:'crosshair',overflow:'visible'}}
          onMouseMove={onMove} onMouseLeave={()=>setHover(null)}>
          <defs>
            {series.map((_,i)=>(
              <React.Fragment key={i}>
                <linearGradient id={`lcg${i}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={COLS[i]} stopOpacity="0.30"/>
                  <stop offset="80%" stopColor={COLS[i]} stopOpacity="0.02"/>
                </linearGradient>
                <filter id={`glow${i}`} x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="2.5" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </React.Fragment>
            ))}
            <clipPath id="cc">
              <rect x={LP.left} y={LP.top} width={VW-LP.left-LP.right} height={VH-LP.top-LP.bottom}/>
            </clipPath>
          </defs>

          {/* Grid */}
          {yTicks.map((t,i)=>(
            <line key={i} x1={LP.left} x2={VW-LP.right} y1={ySc(t)} y2={ySc(t)}
              stroke="rgba(255,255,255,0.045)" strokeWidth="1" strokeDasharray="4 3"/>
          ))}

          {/* Area fills */}
          {series.map((_,i)=>(
            <path key={i} d={mkArea(series[i].data)} fill={`url(#lcg${i})`} clipPath="url(#cc)"/>
          ))}

          {/* Glow line layer */}
          {series.map((_,i)=>(
            <path key={i} d={mkLine(series[i].data)} fill="none"
              stroke={COLS[i]} strokeWidth="3.5" strokeOpacity="0.35" clipPath="url(#cc)"
              style={{filter:`blur(3px)`}}/>
          ))}
          {/* Crisp line layer */}
          {series.map((_,i)=>(
            <path key={i} d={mkLine(series[i].data)} fill="none"
              stroke={COLS[i]} strokeWidth="1.8" clipPath="url(#cc)" strokeLinejoin="round"/>
          ))}

          {/* Axes */}
          <line x1={LP.left} x2={VW-LP.right} y1={VH-LP.bottom} y2={VH-LP.bottom}
            stroke="rgba(255,255,255,0.08)" strokeWidth="1"/>

          {/* Tick labels */}
          {xTicks.map(t=>(
            <text key={t} x={xSc(t)} y={VH-LP.bottom+14} textAnchor="middle"
              fontSize="9" fill="var(--text-tertiary)" fontFamily="'DM Sans',sans-serif">{t}</text>
          ))}
          {yTicks.map((t,i)=>(
            <text key={i} x={LP.left-8} y={ySc(t)+3.5} textAnchor="end"
              fontSize="9" fill="var(--text-tertiary)" fontFamily="'DM Sans',sans-serif">{t.toFixed(2)}</text>
          ))}
          <text x={(LP.left+VW-LP.right)/2} y={VH-1}
            textAnchor="middle" fontSize="8.5" fill="var(--text-tertiary)" fontFamily="'DM Sans',sans-serif">epoch</text>

          {/* Crosshair */}
          {hover&&<>
            <line x1={hover.cx} x2={hover.cx} y1={LP.top} y2={VH-LP.bottom}
              stroke="rgba(255,255,255,0.18)" strokeWidth="1" strokeDasharray="3 2"/>
            {series.map((_,i)=>{
              const cy=ySc(series[i].data[hover.epoch-1]);
              return <React.Fragment key={i}>
                <circle cx={hover.cx} cy={cy} r="5" fill={COLS[i]} opacity="0.25"/>
                <circle cx={hover.cx} cy={cy} r="3.5" fill={COLS[i]} stroke="#0f1714" strokeWidth="1.5"/>
              </React.Fragment>;
            })}
          </>}

          <rect x={LP.left} y={LP.top} width={VW-LP.left-LP.right} height={VH-LP.top-LP.bottom} fill="transparent"/>
        </svg>

        {/* Popover */}
        {hover&&(
          <div style={{
            position:'absolute',
            left:`calc(${hover.pct*100}% + ${hover.pct>0.68?-144:14}px)`,
            top:'18px',
            pointerEvents:'none',
            background:'rgba(8,14,12,0.96)',
            backdropFilter:'blur(16px)',
            WebkitBackdropFilter:'blur(16px)',
            border:'1px solid rgba(255,255,255,0.13)',
            borderRadius:10,
            padding:'9px 13px',
            minWidth:128,
            boxShadow:'0 12px 36px rgba(0,0,0,0.55)',
            zIndex:20,
          }}>
            <div style={{fontSize:'9px',fontWeight:600,letterSpacing:'0.6px',textTransform:'uppercase',color:'var(--text-tertiary)',marginBottom:7}}>
              Epoch {hover.epoch}
            </div>
            {hover.vals.map(v=>(
              <div key={v.label} style={{display:'flex',justifyContent:'space-between',alignItems:'center',gap:18,marginBottom:4}}>
                <div style={{display:'flex',alignItems:'center',gap:5}}>
                  <div style={{width:6,height:6,borderRadius:'50%',background:v.col,boxShadow:`0 0 5px ${v.col}`}}/>
                  <span style={{fontSize:'10px',color:'var(--text-secondary)'}}>{v.label}</span>
                </div>
                <span style={{fontSize:'11px',fontWeight:600,color:'var(--text-primary)',fontFamily:'monospace'}}>{v.v.toFixed(4)}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

const ClassAPChart = ({data}) => {
  const sorted=[...VOC_CLASSES].map(cls=>({cls,ap:data[cls]||0})).sort((a,b)=>b.ap-a.ap);
  const mAP=sorted.reduce((s,x)=>s+x.ap,0)/sorted.length;
  const [hov,setHov]=React.useState(null);
  const [ready,setReady]=React.useState(false);
  React.useEffect(()=>{const t=setTimeout(()=>setReady(true),80);return()=>clearTimeout(t);},[]);

  return (
    <div style={{...card,padding:'20px 22px',display:'flex',flexDirection:'column',gap:14,height:'100%'}}
      onMouseEnter={cardHover} onMouseLeave={cardLeave}>
      <div style={{display:'flex',alignItems:'baseline',justifyContent:'space-between'}}>
        <span style={{fontSize:'13px',fontWeight:600,color:'var(--text-primary)'}}>Per-Class AP</span>
        <div style={{display:'flex',alignItems:'center',gap:6}}>
          <div style={{width:1,height:12,background:'rgba(255,255,255,0.2)'}}/>
          <span style={{fontSize:'11px',color:'var(--accent)',fontWeight:600}}>mAP {(mAP*100).toFixed(1)}%</span>
        </div>
      </div>
      <div style={{flex:1,overflowY:'auto',display:'flex',flexDirection:'column',gap:5}}>
        {sorted.map(({cls,ap},idx)=>{
          const above=ap>=mAP;
          const isH=hov===cls;
          return (
            <div key={cls} style={{display:'flex',alignItems:'center',gap:10}}
              onMouseEnter={()=>setHov(cls)} onMouseLeave={()=>setHov(null)}>
              <span style={{
                width:78,textAlign:'right',fontSize:'10px',flexShrink:0,
                color:isH?'var(--text-primary)':'var(--text-secondary)',
                transition:'color 0.15s',fontWeight:isH?600:400,
              }}>{cls}</span>
              <div style={{flex:1,height:9,background:'rgba(255,255,255,0.05)',borderRadius:6,position:'relative',overflow:'hidden'}}>
                <div style={{
                  position:'absolute',left:0,top:0,bottom:0,
                  width: ready?`${ap*100}%`:'0%',
                  background: above
                    ? (isH?'linear-gradient(90deg,#00d4a0,#00d4a088)':'linear-gradient(90deg,rgba(0,212,160,0.7),rgba(0,212,160,0.35))')
                    : (isH?'linear-gradient(90deg,#e88548,#e8854888)':'linear-gradient(90deg,rgba(232,133,72,0.65),rgba(232,133,72,0.3))'),
                  borderRadius:6,
                  transition:`width 0.7s cubic-bezier(0.22,1,0.36,1) ${idx*0.022}s, background 0.15s`,
                }}/>
                {/* mAP reference */}
                <div style={{position:'absolute',top:0,bottom:0,left:`${mAP*100}%`,width:1,background:'rgba(255,255,255,0.2)'}}/>
              </div>
              <span style={{
                width:34,fontSize:'9.5px',fontFamily:'monospace',flexShrink:0,textAlign:'right',
                color:above?'var(--accent)':'var(--text-tertiary)',
                fontWeight:isH?600:400,transition:'color 0.15s',
              }}>{(ap*100).toFixed(1)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const ConfusionMatrix = ({matrix}) => {
  const [tip,setTip]=React.useState(null);
  const N=21,CELL=16;
  const labels=['bg',...VOC_CLASSES.map(c=>c.slice(0,4))];
  const flat=matrix.flat(),mx=Math.max(...flat);
  const colSc=d3.scaleSequential(d3.interpolate('#0a1a14','#00d4a0')).domain([0,mx]);

  return (
    <div style={{...card,padding:'20px 22px',display:'flex',flexDirection:'column',gap:14}}
      onMouseEnter={cardHover} onMouseLeave={cardLeave}>
      <div style={{display:'flex',alignItems:'baseline',justifyContent:'space-between'}}>
        <span style={{fontSize:'13px',fontWeight:600,color:'var(--text-primary)'}}>Confusion Matrix</span>
        <span style={{fontSize:'10px',color:'var(--text-tertiary)'}}>rows actual · cols predicted</span>
      </div>
      <div style={{overflowX:'auto',position:'relative'}}>
        <svg width={N*CELL+2} height={N*CELL+2} style={{display:'block'}}>
          {matrix.map((row,r)=>row.map((val,c)=>{
            const x=1+c*CELL,y=1+r*CELL;
            return (
              <rect key={`${r}${c}`} x={x} y={y} width={CELL-1} height={CELL-1} rx="2"
                fill={val>0?colSc(val):'#0a1410'}
                onMouseEnter={e=>{
                  const sr=e.currentTarget.closest('svg').getBoundingClientRect();
                  const px=e.clientX-sr.left,py=e.clientY-sr.top;
                  setTip({r,c,val,x:px,y:py});
                }}
                onMouseLeave={()=>setTip(null)}
                style={{cursor:'default'}}/>
            );
          }))}
          {tip&&tip.val>0&&(()=>{
            const tx=Math.min(tip.x+8,N*CELL-118),ty=Math.max(tip.y-38,2);
            return(
              <g>
                <rect x={tx} y={ty} width={114} height={36} rx="6"
                  fill="rgba(8,14,12,0.97)" stroke="rgba(255,255,255,0.14)" strokeWidth="0.8"/>
                <text x={tx+9} y={ty+13} fontSize="8.5" fill="var(--text-secondary)" fontFamily="'DM Sans',sans-serif">
                  {labels[tip.r]} → {labels[tip.c]}
                </text>
                <text x={tx+9} y={ty+26} fontSize="10" fontWeight="700" fill="var(--text-primary)" fontFamily="'DM Sans',sans-serif">
                  {tip.val} samples
                </text>
              </g>
            );
          })()}
        </svg>
      </div>
      {/* Color scale legend */}
      <div style={{display:'flex',alignItems:'center',gap:8,marginTop:2}}>
        <span style={{fontSize:'9px',color:'var(--text-tertiary)'}}>0</span>
        <div style={{flex:1,height:4,borderRadius:3,background:'linear-gradient(90deg,#0a1a14,#00d4a0)'}}/>
        <span style={{fontSize:'9px',color:'var(--text-tertiary)'}}>{mx}</span>
      </div>
    </div>
  );
};

const DetectionCard=({img})=>{
  const [hov,setHov]=React.useState(null);
  const W=200,H=136;
  return(
    <div style={{flexShrink:0,width:W,display:'flex',flexDirection:'column',gap:6}}>
      <svg width={W} height={H} style={{display:'block',borderRadius:10,overflow:'hidden',border:'1px solid rgba(255,255,255,0.07)'}}>
        <rect width={W} height={H} fill="#0b1812"/>
        {/* Subtle grid lines */}
        {[1,2,3,4].map(i=><line key={`v${i}`} x1={i*(W/5)} y1={0} x2={i*(W/5)} y2={H} stroke="rgba(255,255,255,0.025)" strokeWidth="1"/>)}
        {[1,2,3].map(i=><line key={`h${i}`} x1={0} y1={i*(H/4)} x2={W} y2={i*(H/4)} stroke="rgba(255,255,255,0.025)" strokeWidth="1"/>)}
        <text x={W/2} y={H/2} textAnchor="middle" fontSize="11" fill="rgba(255,255,255,0.06)"
          fontFamily="'DM Sans',sans-serif" dy=".35em">{img.label}</text>
        {img.boxes.map((box,b)=>{
          const bx=box.x*W,by=box.y*H,bw=box.w*W,bh=box.h*H;
          const col=BOX_COLORS[b%BOX_COLORS.length];
          const isH=hov===b;
          const lw=box.cls.length*5.5+32;
          return(
            <g key={b} onMouseEnter={()=>setHov(b)} onMouseLeave={()=>setHov(null)}>
              <rect x={bx} y={by} width={bw} height={bh} rx="3"
                fill={`${col}${isH?'25':'10'}`} stroke={col} strokeWidth={isH?2:1.3}
                style={{transition:'all 0.15s'}}/>
              <rect x={bx} y={Math.max(by-14,0)} width={lw} height={14} rx="3" fill={col}/>
              <text x={bx+4} y={Math.max(by-3,10)} fontSize="7.5" fill="#000" fontWeight="700"
                fontFamily="'DM Sans',sans-serif">{box.cls} {(box.score*100).toFixed(0)}%</text>
            </g>
          );
        })}
      </svg>
      <div style={{display:'flex',justifyContent:'space-between',padding:'0 2px'}}>
        <span style={{fontSize:'9.5px',color:'var(--text-tertiary)',fontFamily:'monospace'}}>{img.label}</span>
        <span style={{fontSize:'9.5px',color:'var(--text-tertiary)'}}>{img.boxes.length} det</span>
      </div>
    </div>
  );
};

const MetricsView=({metricsData,selectedExp})=>{
  const d=metricsData||{};
  const trainLoss=d.train_loss||MOCK_TRAIN_LOSS;
  const valLoss  =d.val_loss  ||MOCK_VAL_LOSS;
  const mapCurve =d.map_curve ||MOCK_MAP_CURVE;
  const classAP  =d.class_ap  ||MOCK_CLASS_AP;
  const confMat  =d.conf_mat  ||MOCK_CONF;
  const images   =d.images    ||MOCK_IMAGES;

  const bestMAP   =Math.max(...mapCurve);
  const bestEpoch =mapCurve.indexOf(bestMAP)+1;
  const finalTrain=trainLoss[trainLoss.length-1];
  const finalVal  =valLoss[valLoss.length-1];
  const mAP       =Object.values(classAP).reduce((s,v)=>s+v,0)/Object.values(classAP).length;

  return(
    <div style={{
      height:'100%',overflowY:'auto',
      padding:'18px 20px 32px',
      display:'flex',flexDirection:'column',gap:14,
    }}>

      <FadeIn delay={0.04} from="top">
        <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:10}}>
          <StatTile label="Best mAP@0.5"   value={`${(bestMAP*100).toFixed(1)}%`}  sub={`epoch ${bestEpoch}`}   accentColor="var(--accent)" sparkData={mapCurve}/>
          <StatTile label="Mean AP"         value={`${(mAP*100).toFixed(1)}%`}       sub="20 VOC classes"         accentColor="#7c9ef5"       sparkData={mapCurve}/>
          <StatTile label="Final Train Loss" value={finalTrain.toFixed(3)}            sub="epoch 200"              accentColor="var(--text-secondary)" sparkData={trainLoss}/>
          <StatTile label="Final Val Loss"   value={finalVal.toFixed(3)}              sub="epoch 200"              accentColor="#e88548"       sparkData={valLoss}/>
          <StatTile label="Total Steps"      value={selectedExp?.total_steps!=null?selectedExp.total_steps.toLocaleString():'—'} sub={selectedExp?.experiment_id||'no experiment'} accentColor="var(--text-tertiary)"/>
        </div>
      </FadeIn>

      <FadeIn delay={0.1} from="bottom">
        <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:14}}>
          <LineChart title="Loss Curves" sub="train & validation · lower is better"
            series={[{label:'Train',data:trainLoss},{label:'Val',data:valLoss}]}/>
          <LineChart title="mAP@0.5" sub="VOC2012 validation · higher is better"
            series={[{label:'mAP',data:mapCurve}]}/>
        </div>
      </FadeIn>

      <FadeIn delay={0.16} from="bottom">
        <div style={{display:'grid',gridTemplateColumns:'1fr auto',gap:14,alignItems:'start'}}>
          <ClassAPChart data={classAP}/>
          <ConfusionMatrix matrix={confMat}/>
        </div>
      </FadeIn>

      <FadeIn delay={0.22} from="bottom">
        <div style={{...card,padding:'20px 22px'}}
          onMouseEnter={cardHover} onMouseLeave={cardLeave}>
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:14}}>
            <span style={{fontSize:'13px',fontWeight:600,color:'var(--text-primary)'}}>Detection Samples</span>
            <span style={{fontSize:'10px',color:'var(--text-tertiary)'}}>epoch 200 · val set · hover boxes for score</span>
          </div>
          <div style={{display:'flex',gap:12,overflowX:'auto',paddingBottom:4}}>
            {images.map(img=><DetectionCard key={img.id} img={img}/>)}
          </div>
        </div>
      </FadeIn>

    </div>
  );
};

Object.assign(window,{MetricsView});
