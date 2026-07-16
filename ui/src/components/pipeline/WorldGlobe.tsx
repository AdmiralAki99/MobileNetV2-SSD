import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import * as topojson from 'topojson-client'

const WORLD_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'

export const AWS_REGIONS: { id: string; name: string; coords: [number, number] }[] = [
  { id: 'us-east-1',      name: 'N. Virginia',   coords: [-77.4,  39.0] },
  { id: 'us-east-2',      name: 'Ohio',           coords: [-82.9,  40.4] },
  { id: 'us-west-1',      name: 'N. California',  coords: [-121.9, 37.4] },
  { id: 'us-west-2',      name: 'Oregon',         coords: [-122.7, 45.5] },
  { id: 'ca-central-1',   name: 'Montreal',       coords: [-73.6,  45.5] },
  { id: 'eu-west-1',      name: 'Ireland',        coords: [-8.0,   53.3] },
  { id: 'eu-west-2',      name: 'London',         coords: [-0.1,   51.5] },
  { id: 'eu-central-1',   name: 'Frankfurt',      coords: [8.7,    50.1] },
  { id: 'eu-north-1',     name: 'Stockholm',      coords: [18.1,   59.3] },
  { id: 'ap-northeast-1', name: 'Tokyo',          coords: [139.7,  35.7] },
  { id: 'ap-southeast-1', name: 'Singapore',      coords: [103.8,   1.3] },
  { id: 'ap-southeast-2', name: 'Sydney',         coords: [151.2, -33.9] },
  { id: 'ap-south-1',     name: 'Mumbai',         coords: [72.9,   19.1] },
  { id: 'sa-east-1',      name: 'São Paulo',      coords: [-46.6, -23.5] },
  { id: 'me-south-1',     name: 'Bahrain',        coords: [50.6,   26.2] },
  { id: 'af-south-1',     name: 'Cape Town',      coords: [18.4,  -33.9] },
]

interface Props {
  activeRegion?: string | null
}

export const WorldGlobe = ({ activeRegion }: Props) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const stateRef     = useRef({
    rotation:  [-20, -25, 0] as [number, number, number],
    dragging:  false,
    lastMouse: [0, 0] as [number, number],
    world:     null as any,
    animId:    0,
    W: 0, H: 0,
    dpr: 1,
  })

  const drawRef = useRef<((time: number) => void) | null>(null)

  useEffect(() => {
    const canvas    = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const state = stateRef.current
    state.dpr = window.devicePixelRatio || 1

    const resize = () => {
      const { width, height } = container.getBoundingClientRect()
      state.W = width
      state.H = height
      canvas.width  = width  * state.dpr
      canvas.height = height * state.dpr
      canvas.style.width  = width  + 'px'
      canvas.style.height = height + 'px'
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(container)

    fetch(WORLD_URL)
      .then(r => r.json())
      .then(world => { state.world = world })

    drawRef.current = (time: number) => {
      const ctx = canvas.getContext('2d')
      if (!ctx || !state.world) return

      const { W, H, dpr, rotation } = state
      const radius = Math.min(W, H) / 2 - 14
      const cx = W / 2, cy = H / 2

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)

      const projection = d3.geoOrthographic()
        .scale(radius)
        .translate([cx, cy])
        .clipAngle(90)
        .rotate(rotation)

      const path      = d3.geoPath(projection, ctx)
      const graticule = d3.geoGraticule()
      const sphere    = { type: 'Sphere' } as GeoJSON.GeoJsonObject

      const land    = topojson.feature(state.world, state.world.objects.land)
      const borders = topojson.mesh(state.world, state.world.objects.countries,
        (a: any, b: any) => a !== b)

      const atmo = ctx.createRadialGradient(cx, cy, radius * 0.88, cx, cy, radius * 1.18)
      atmo.addColorStop(0,   'rgba(0,180,120,0.10)')
      atmo.addColorStop(0.5, 'rgba(0,140,90,0.04)')
      atmo.addColorStop(1,   'transparent')
      ctx.beginPath()
      ctx.arc(cx, cy, radius * 1.18, 0, Math.PI * 2)
      ctx.fillStyle = atmo
      ctx.fill()

      ctx.beginPath()
      path(sphere)
      const sphereFill = ctx.createRadialGradient(cx - radius * 0.25, cy - radius * 0.25, 0, cx, cy, radius)
      sphereFill.addColorStop(0, 'rgba(4,18,12,0.95)')
      sphereFill.addColorStop(1, 'rgba(2,10,7,0.98)')
      ctx.fillStyle = sphereFill
      ctx.fill()

      ctx.beginPath()
      path(graticule())
      ctx.strokeStyle = 'rgba(0,200,140,0.06)'
      ctx.lineWidth   = 0.5
      ctx.stroke()

      ctx.beginPath()
      path(land as GeoJSON.GeoJsonObject)
      ctx.fillStyle = 'rgba(0,70,44,0.30)'
      ctx.fill()

      ctx.beginPath()
      path(borders as GeoJSON.GeoJsonObject)
      ctx.strokeStyle = 'rgba(0,200,140,0.18)'
      ctx.lineWidth   = 0.5
      ctx.stroke()

      ctx.beginPath()
      path(sphere)
      ctx.strokeStyle = 'rgba(0,200,140,0.22)'
      ctx.lineWidth   = 1
      ctx.stroke()

      const limb = ctx.createRadialGradient(
        cx - radius * 0.6, cy - radius * 0.5, 0,
        cx - radius * 0.3, cy - radius * 0.3, radius * 1.1,
      )
      limb.addColorStop(0,   'rgba(0,200,140,0.04)')
      limb.addColorStop(0.7, 'transparent')
      ctx.beginPath()
      path(sphere)
      ctx.fillStyle = limb
      ctx.fill()

      const pulse = (Math.sin(time * 0.0025) + 1) / 2

      AWS_REGIONS.forEach(region => {
        const angularDist = d3.geoDistance(
          region.coords,
          [-rotation[0], -rotation[1]] as [number, number],
        )
        if (angularDist > Math.PI / 2) return

        const projected = projection(region.coords)
        if (!projected) return
        const [px, py] = projected

        const isActive = region.id === activeRegion
        const fadeEdge = Math.max(0, 1 - (angularDist / (Math.PI / 2)) * 1.2)

        const outerR = isActive ? 12 + pulse * 6 : 7 + pulse * 3
        const ring = ctx.createRadialGradient(px, py, 0, px, py, outerR)
        ring.addColorStop(0,   isActive
          ? `rgba(0,212,160,${0.35 * fadeEdge})`
          : `rgba(0,200,140,${0.18 * fadeEdge})`)
        ring.addColorStop(1,   'transparent')
        ctx.beginPath()
        ctx.arc(px, py, outerR, 0, Math.PI * 2)
        ctx.fillStyle = ring
        ctx.fill()

        const dotR = isActive ? 3.5 : 2
        ctx.beginPath()
        ctx.arc(px, py, dotR, 0, Math.PI * 2)
        ctx.fillStyle = isActive
          ? `rgba(0,212,160,${0.9 * fadeEdge})`
          : `rgba(0,200,140,${0.7 * fadeEdge})`
        ctx.fill()

        if (isActive) {
          ctx.font      = '9px monospace'
          ctx.fillStyle = `rgba(0,212,160,${0.7 * fadeEdge})`
          ctx.fillText(region.name, px + 6, py - 5)
        }
      })
    }

    let lastTime = 0
    const loop = (time: number) => {
      const dt = Math.min(time - lastTime, 50)
      lastTime = time
      if (!state.dragging) {
        state.rotation[0] += dt * 0.008
      }
      drawRef.current?.(time)
      state.animId = requestAnimationFrame(loop)
    }
    state.animId = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(state.animId)
      ro.disconnect()
    }
  }, [])


  const onMouseDown = (e: React.MouseEvent) => {
    const s = stateRef.current
    s.dragging  = true
    s.lastMouse = [e.clientX, e.clientY]
  }
  const onMouseMove = (e: React.MouseEvent) => {
    const s = stateRef.current
    if (!s.dragging) return
    const dx = e.clientX - s.lastMouse[0]
    const dy = e.clientY - s.lastMouse[1]
    s.rotation[0] += dx * 0.35
    s.rotation[1]  = Math.max(-60, Math.min(60, s.rotation[1] - dy * 0.35))
    s.lastMouse = [e.clientX, e.clientY]
  }
  const onMouseUp = () => { stateRef.current.dragging = false }

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ display: 'block', width: '100%', height: '100%', cursor: 'grab' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      />
    </div>
  )
}
