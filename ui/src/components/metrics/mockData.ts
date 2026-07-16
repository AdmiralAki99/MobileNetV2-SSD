const _rng = (() => { let s = 42; return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; }; })()
const _curve = (fn: (e: number) => number, noise: number, n = 200) =>
  Array.from({ length: n }, (_, i) => Math.max(0, fn(i + 1) + (_rng() - 0.5) * noise))

export const MOCK_TRAIN_LOSS = _curve(e => 4.5 * Math.exp(-e / 45) + 0.82, 0.18)
export const MOCK_VAL_LOSS   = _curve(e => 4.8 * Math.exp(-e / 50) + 0.94, 0.28)
export const MOCK_MAP_CURVE  = _curve(e => 0.766 / (1 + Math.exp(-(e - 80) / 22)), 0.018)

export const MOCK_LR_CURVE = Array.from({ length: 200 }, (_, i) => {
  const e = i + 1
  if (e <= 10) return (e / 10) * 0.01
  const t = (e - 10) / 190
  return 1e-5 + (0.01 - 1e-5) * 0.5 * (1 + Math.cos(Math.PI * t))
})

export const MOCK_NMS_MEAN_SCORE   = _curve(e => 0.72 - 0.18 * Math.exp(-e / 60), 0.025)
export const MOCK_NMS_AVG_DET      = _curve(e => 3.2  + 4.1  * (1 - Math.exp(-e / 55)), 0.4)
export const MOCK_NMS_ZERO_DET     = _curve(e => Math.max(0, 0.35 * Math.exp(-e / 40)), 0.03)

export const VOC_CLASSES = [
  'aeroplane','bicycle','bird','boat','bottle','bus','car','cat',
  'chair','cow','diningtable','dog','horse','motorbike','person',
  'pottedplant','sheep','sofa','train','tvmonitor',
]

export const MOCK_CLASS_AP: Record<string, number> = {
  aeroplane:0.812, bicycle:0.791, bird:0.734,  boat:0.623,    bottle:0.423,
  bus:0.847,       car:0.812,     cat:0.883,   chair:0.512,   cow:0.741,
  diningtable:0.688, dog:0.871,   horse:0.856, motorbike:0.803, person:0.847,
  pottedplant:0.489, sheep:0.762, sofa:0.692,  train:0.851,   tvmonitor:0.735,
}

export const MOCK_CONF = (() => {
  const N = 21, m = Array.from({ length: N }, () => Array(N).fill(0))
  for (let r = 0; r < N; r++) {
    m[r][r] = 80 + (_rng() * 40 | 0)
    for (let c = 0; c < N; c++) if (c !== r && _rng() < 0.16) m[r][c] = (_rng() * 12 | 0)
  }
  return m
})()

export const BOX_COLORS = ['#00d4a0','#e88548','#7c9ef5','#e84855','#65c16a','#c97cf5','#f5ea7c']

export const MOCK_IMAGES = Array.from({ length: 8 }, (_, i) => ({
  id: i,
  label: VOC_CLASSES[i % 20],
  boxes: Array.from({ length: 1 + i % 3 }, (_, b) => ({
    x: 0.06 + _rng() * 0.40, y: 0.10 + _rng() * 0.30,
    w: 0.20 + _rng() * 0.26, h: 0.20 + _rng() * 0.24,
    cls: VOC_CLASSES[(i + b) % 20], score: 0.56 + _rng() * 0.43,
  })),
}))
