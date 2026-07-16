import { useEffect, useRef, useState } from 'react'
import { type Experiment } from '../../types/experiment'

const STATUS_COLOR: Record<string, string> = {
  running:  '#00d4a0',
  complete: '#8a9a6a',
  failed:   '#c87070',
  pending:  'rgba(255,255,255,0.4)',
  stopped:  'rgba(255,255,255,0.25)',
}

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))

interface Pt3 { x: number; y: number; z: number }

interface ProjectedPoint {
  exp: Experiment
  sx: number; sy: number; k: number; r: number; color: string; hasMetric: boolean
}

function rotateY(p: Pt3, a: number): Pt3 {
  const cos = Math.cos(a), sin = Math.sin(a)
  return { x: p.x * cos + p.z * sin, y: p.y, z: -p.x * sin + p.z * cos }
}
function rotateX(p: Pt3, a: number): Pt3 {
  const cos = Math.cos(a), sin = Math.sin(a)
  return { x: p.x, y: p.y * cos - p.z * sin, z: p.y * sin + p.z * cos }
}

interface Props {
  experiments: Experiment[]
  selectedId?: string
  onSelect: (exp: Experiment) => void
}

export const ExperimentOrbit = ({ experiments, selectedId, onSelect }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const stateRef = useRef({
    yaw: 0.6, pitch: -0.3,
    dragging: false, dragged: false,
    lastMouse: [0, 0] as [number, number],
    W: 0, H: 0, dpr: 1,
    animId: 0,
    lastProjected: [] as ProjectedPoint[],
    mouse: null as [number, number] | null,
  })

  const dataRef = useRef({ experiments, selectedId, hoveredId })
  dataRef.current = { experiments, selectedId, hoveredId }

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return
    const state = stateRef.current
    state.dpr = window.devicePixelRatio || 1

    const resize = () => {
      const { width, height } = container.getBoundingClientRect()
      state.W = width; state.H = height
      canvas.width  = width  * state.dpr
      canvas.height = height * state.dpr
      canvas.style.width  = width  + 'px'
      canvas.style.height = height + 'px'
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    const draw = (time: number) => {
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const { W, H, dpr, yaw, pitch } = state
      if (W === 0 || H === 0) return

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)

      const cx = W / 2, cy = H / 2
      const R = Math.min(W, H) * 0.34
      const D = R * 2.4

      const { experiments: exps, selectedId: selId, hoveredId: hovId } = dataRef.current
      const n = Math.max(exps.length, 1)

      const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * 1.3)
      core.addColorStop(0,   'rgba(0,180,120,0.10)')
      core.addColorStop(0.5, 'rgba(0,140,90,0.04)')
      core.addColorStop(1,   'transparent')
      ctx.beginPath(); ctx.arc(cx, cy, R * 1.3, 0, Math.PI * 2); ctx.fillStyle = core; ctx.fill()

      const drawRing = (tiltX: number, tiltY: number, alpha: number) => {
        ctx.beginPath()
        for (let i = 0; i <= 64; i++) {
          const a = (i / 64) * Math.PI * 2
          let p: Pt3 = { x: Math.cos(a) * R, y: 0, z: Math.sin(a) * R }
          p = rotateX(p, tiltX); p = rotateY(p, tiltY)
          p = rotateY(p, yaw); p = rotateX(p, pitch)
          const denom = D + p.z
          if (denom <= 1) continue
          const k = D / denom
          const sx = cx + p.x * k, sy = cy + p.y * k
          if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy)
        }
        ctx.strokeStyle = `rgba(0,200,140,${alpha})`
        ctx.lineWidth = 0.6
        ctx.stroke()
      }
      drawRing(0, 0, 0.08)
      drawRing(Math.PI / 2, 0, 0.06)
      drawRing(Math.PI / 4, Math.PI / 3, 0.05)

      const pulse = (Math.sin(time * 0.0025) + 1) / 2

      const projected: ProjectedPoint[] = exps.map((exp, i) => {
        const yFrac = n === 1 ? 0 : 1 - (i / (n - 1)) * 2
        const radiusAtY = Math.sqrt(Math.max(0, 1 - yFrac * yFrac))
        const theta = GOLDEN_ANGLE * i
        const dirX = Math.cos(theta) * radiusAtY
        const dirY = yFrac
        const dirZ = Math.sin(theta) * radiusAtY

        const recencyFrac = n === 1 ? 1 : i / (n - 1)
        const shellR = R * (0.35 + recencyFrac * 0.65)

        let p: Pt3 = { x: dirX * shellR, y: dirY * shellR, z: dirZ * shellR }
        p = rotateY(p, yaw)
        p = rotateX(p, pitch)

        const denom = D + p.z
        const k = denom > 1 ? D / denom : 0.0001
        const sx = cx + p.x * k
        const sy = cy + p.y * k

        const hasMetric = exp.best_metric != null
        const epochFrac = exp.best_epoch != null ? Math.min(exp.best_epoch / 200, 1) : 0
        const baseR = (hasMetric ? 3 + epochFrac * 3.5 : 2.2) * k
        const color = STATUS_COLOR[exp.status] ?? 'rgba(255,255,255,0.3)'

        return { exp, sx, sy, k, r: Math.max(baseR, 1), color, hasMetric }
      })

      projected.sort((a, b) => a.k - b.k)
      state.lastProjected = projected

      for (const p of projected) {
        const isSel = p.exp.experiment_id === selId
        const isHov = p.exp.experiment_id === hovId
        const isRunning = p.exp.status === 'running'
        const fade = Math.max(0.25, Math.min(1, p.k))

        if (isRunning) {
          const ringR = p.r + 3 + pulse * 5
          const ring = ctx.createRadialGradient(p.sx, p.sy, 0, p.sx, p.sy, ringR)
          ring.addColorStop(0, `rgba(0,212,160,${0.3 * fade})`)
          ring.addColorStop(1, 'transparent')
          ctx.beginPath(); ctx.arc(p.sx, p.sy, ringR, 0, Math.PI * 2); ctx.fillStyle = ring; ctx.fill()
        }

        if (isSel) {
          ctx.beginPath()
          ctx.arc(p.sx, p.sy, p.r + 5, 0, Math.PI * 2)
          ctx.strokeStyle = `rgba(255,255,255,${0.5 * fade})`
          ctx.lineWidth = 1
          ctx.stroke()
        }

        ctx.beginPath()
        ctx.arc(p.sx, p.sy, isHov ? p.r + 1.2 : p.r, 0, Math.PI * 2)
        ctx.fillStyle = p.color
        ctx.globalAlpha = (p.hasMetric ? 0.9 : 0.45) * fade
        ctx.fill()
        ctx.globalAlpha = 1

        if (isHov || isSel) {
          ctx.font = '9px monospace'
          ctx.fillStyle = `rgba(220,230,225,${0.85 * fade})`
          ctx.fillText(p.exp.experiment_id, p.sx + p.r + 6, p.sy - p.r - 2)
        }
      }
    }

    let lastTime = 0
    const loop = (time: number) => {
      const dt = Math.min(time - lastTime, 50)
      lastTime = time
      if (!state.dragging) state.yaw += dt * 0.00012
      draw(time)
      state.animId = requestAnimationFrame(loop)
    }
    state.animId = requestAnimationFrame(loop)

    return () => { cancelAnimationFrame(state.animId); ro.disconnect() }
  }, [])

  const pickPoint = (mx: number, my: number): ProjectedPoint | null => {
    const pts = stateRef.current.lastProjected
    let best: ProjectedPoint | null = null
    let bestDist = Infinity
    for (const p of pts) {
      const d = Math.hypot(p.sx - mx, p.sy - my)
      const threshold = Math.max(p.r + 6, 8)
      if (d < threshold && d < bestDist) { best = p; bestDist = d }
    }
    return best
  }

  const onMouseDown = (e: React.MouseEvent) => {
    const s = stateRef.current
    s.dragging = true; s.dragged = false
    s.lastMouse = [e.clientX, e.clientY]
  }
  const onMouseMove = (e: React.MouseEvent) => {
    const s = stateRef.current
    const rect = (e.target as HTMLElement).getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top

    if (s.dragging) {
      const dx = e.clientX - s.lastMouse[0]
      const dy = e.clientY - s.lastMouse[1]
      if (Math.abs(dx) > 2 || Math.abs(dy) > 2) s.dragged = true
      s.yaw += dx * 0.006
      s.pitch = Math.max(-1.2, Math.min(1.2, s.pitch - dy * 0.006))
      s.lastMouse = [e.clientX, e.clientY]
      return
    }
    const hit = pickPoint(mx, my)
    setHoveredId(hit ? hit.exp.experiment_id : null)
  }
  const onMouseUp = (e: React.MouseEvent) => {
    const s = stateRef.current
    const wasDragged = s.dragged
    s.dragging = false
    if (!wasDragged) {
      const rect = (e.target as HTMLElement).getBoundingClientRect()
      const hit = pickPoint(e.clientX - rect.left, e.clientY - rect.top)
      if (hit) onSelect(hit.exp)
    }
  }

  const hovered = hoveredId ? experiments.find(e => e.experiment_id === hoveredId) : null

  return (
    <div ref={containerRef} data-testid="experiment-orbit" style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%', cursor: hoveredId ? 'pointer' : 'grab' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={() => { stateRef.current.dragging = false; setHoveredId(null) }}
      />
      {experiments.length === 0 && (
        <div style={{
          position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '11px', color: 'rgba(255,255,255,0.15)', fontFamily: 'monospace', pointerEvents: 'none',
        }}>
          no experiments yet
        </div>
      )}
      {hovered && (
        <div style={{
          position: 'absolute', bottom: 10, left: 10, pointerEvents: 'none',
          backdropFilter: 'blur(16px) saturate(1.4)', background: 'rgba(4,10,7,0.94)',
          border: '1px solid rgba(255,255,255,0.08)', borderRadius: 5, padding: '6px 10px', zIndex: 40,
        }}>
          <div style={{ fontSize: '9.5px', fontWeight: 700, color: '#dde2e0', fontFamily: 'monospace', marginBottom: 4 }}>
            {hovered.experiment_id}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '8.5px', color: STATUS_COLOR[hovered.status] ?? '#fff', fontFamily: 'monospace' }}>
              {hovered.status}
            </span>
            {hovered.best_metric != null && (
              <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>
                mAP {(hovered.best_metric * 100).toFixed(1)}%
              </span>
            )}
            {hovered.best_epoch != null && (
              <span style={{ fontSize: '8.5px', color: 'rgba(255,255,255,0.3)', fontFamily: 'monospace' }}>
                ep {hovered.best_epoch}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
