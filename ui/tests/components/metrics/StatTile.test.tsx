import { render, screen } from '@testing-library/react'
import { StatTile } from '../../../src/components/metrics/StatTile'

describe('StatTile', () => {
  test('renders label', () => {
    render(<StatTile label="Best mAP" value="76.6%" />)
    expect(screen.getByTestId('stat-label')).toHaveTextContent('Best mAP')
  })

  test('renders value', () => {
    render(<StatTile label="Best mAP" value="76.6%" />)
    expect(screen.getByTestId('stat-value')).toHaveTextContent('76.6%')
  })

  test('renders sub when provided', () => {
    render(<StatTile label="Best mAP" value="76.6%" sub="epoch 200" />)
    expect(screen.getByTestId('stat-sub')).toHaveTextContent('epoch 200')
  })

  test('does not render sub when omitted', () => {
    render(<StatTile label="Best mAP" value="76.6%" />)
    expect(screen.queryByTestId('stat-sub')).not.toBeInTheDocument()
  })

  test('renders sparkline when sparkData has more than 1 point', () => {
    const data = Array.from({ length: 30 }, (_, i) => i * 0.01)
    render(<StatTile label="mAP" value="76%" sparkData={data} />)
    expect(screen.getByTestId('sparkline')).toBeInTheDocument()
  })

  test('does not render sparkline when sparkData is absent', () => {
    render(<StatTile label="mAP" value="76%" />)
    expect(screen.queryByTestId('sparkline')).not.toBeInTheDocument()
  })

  test('does not render sparkline when sparkData has 1 point', () => {
    render(<StatTile label="mAP" value="76%" sparkData={[0.5]} />)
    expect(screen.queryByTestId('sparkline')).not.toBeInTheDocument()
  })

  test('applies accentColor to value text', () => {
    render(<StatTile label="mAP" value="76%" accentColor="#00d4a0" />)
    const val = screen.getByTestId('stat-value')
    expect(val.style.color).toBe('rgb(0, 212, 160)')
  })
})
