const originalError = console.error.bind(console)
console.error = (...args: unknown[]) => {
  const msg = typeof args[0] === 'string' ? args[0] : ''
  if (msg.includes('not wrapped in act(')) return
  originalError(...args)
}

class MockResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
;(globalThis as any).ResizeObserver = MockResizeObserver

const mockCtx2D = {
  setTransform: () => {},
  clearRect: () => {},
  beginPath: () => {},
  moveTo: () => {},
  lineTo: () => {},
  arc: () => {},
  fill: () => {},
  stroke: () => {},
  fillText: () => {},
  createRadialGradient: () => ({ addColorStop: () => {} }),
  strokeStyle: '', fillStyle: '', lineWidth: 0, font: '', globalAlpha: 1,
}
HTMLCanvasElement.prototype.getContext = (() => mockCtx2D) as any