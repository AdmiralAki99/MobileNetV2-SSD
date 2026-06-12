import { render, screen, act, fireEvent } from '@testing-library/react'
import { ClassAPChart } from '../../../src/components/metrics/ClassAPChart'
import { MOCK_CLASS_AP, VOC_CLASSES } from '../../../src/components/metrics/mockData'

describe('ClassAPChart', () => {
  beforeEach(() => jest.useFakeTimers())
  afterEach(() => jest.useRealTimers())

  test('renders the chart', () => {
    render(<ClassAPChart data={MOCK_CLASS_AP} />)
    expect(screen.getByTestId('class-ap-chart')).toBeInTheDocument()
  })

  test('renders a row for each VOC class', () => {
    render(<ClassAPChart data={MOCK_CLASS_AP} />)
    VOC_CLASSES.forEach(cls => expect(screen.getByTestId(`class-row-${cls}`)).toBeInTheDocument())
  })

  test('displays mAP summary', () => {
    render(<ClassAPChart data={MOCK_CLASS_AP} />)
    expect(screen.getByTestId('map-summary')).toHaveTextContent('mAP')
    expect(screen.getByTestId('map-summary')).toHaveTextContent('%')
  })

  test('renders score for each class', () => {
    render(<ClassAPChart data={MOCK_CLASS_AP} />)
    expect(screen.getByTestId('score-car')).toHaveTextContent('81.2')
  })

  test('bars start at 0% width before timer fires', () => {
    render(<ClassAPChart data={MOCK_CLASS_AP} />)
    expect(screen.getByTestId('bar-car').style.width).toBe('0%')
  })

  test('bars expand to correct width after 80ms', () => {
    render(<ClassAPChart data={MOCK_CLASS_AP} />)
    act(() => { jest.advanceTimersByTime(80) })
    expect(screen.getByTestId('bar-car').style.width).toBe('81.2%')
  })

  test('classes are sorted by AP descending', () => {
    const { container } = render(<ClassAPChart data={MOCK_CLASS_AP} />)
    const rows = Array.from(container.querySelectorAll('[data-testid^="class-row-"]'))
    const scores = rows.map(r => {
      const cls = r.getAttribute('data-testid')!.replace('class-row-', '')
      return MOCK_CLASS_AP[cls] ?? 0
    })
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i]).toBeLessThanOrEqual(scores[i - 1])
    }
  })

  test('hovering a class row highlights the label', () => {
    render(<ClassAPChart data={MOCK_CLASS_AP} />)
    const row = screen.getByTestId('class-row-car')
    fireEvent.mouseEnter(row)
    expect(row.querySelector('span')!.style.color).toBe('var(--text-primary)')
  })

  test('uses 0 for missing classes', () => {
    render(<ClassAPChart data={{}} />)
    expect(screen.getByTestId('score-car')).toHaveTextContent('0.0')
  })
})
