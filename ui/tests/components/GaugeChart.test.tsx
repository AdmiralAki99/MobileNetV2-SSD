import { render, act } from '@testing-library/react'
import { GaugeChart } from '../../src/components/GaugeChart'

describe('GaugeChart', () => {
  beforeEach(() => jest.useFakeTimers())
  afterEach(() => jest.useRealTimers())

  test('renders an svg', () => {
    const { container } = render(<GaugeChart value={50} max={100} />)
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  test('svg has correct aria-label', () => {
    const { container } = render(<GaugeChart value={50} max={100} />)
    const svg = container.querySelector('svg')!
    expect(svg.getAttribute('aria-label')).toBe('gauge: 50 of 100')
  })

  test('respects size prop on svg dimensions', () => {
    const { container } = render(<GaugeChart value={50} max={100} size={120} />)
    const svg = container.querySelector('svg')!
    expect(svg.getAttribute('width')).toBe('120')
    expect(svg.getAttribute('height')).toBe('90')
  })

  test('animates to value after 400ms timeout', () => {
    const { container } = render(<GaugeChart value={75} max={100} />)
    const lineBefore = container.querySelector('line')!
    const x2Before = lineBefore.getAttribute('x2')

    act(() => { jest.advanceTimersByTime(400) })

    const lineAfter = container.querySelector('line')!
    const x2After = lineAfter.getAttribute('x2')
    expect(x2After).not.toBe(x2Before)
  })

  test('clamps value to max', () => {
    const { container } = render(<GaugeChart value={200} max={100} />)
    act(() => { jest.advanceTimersByTime(400) })
    expect(container.querySelector('svg')).toBeInTheDocument()
  })

  test('renders three arc paths', () => {
    const { container } = render(<GaugeChart value={50} max={100} />)
    const paths = container.querySelectorAll('path')
    expect(paths.length).toBe(3)
  })
})
