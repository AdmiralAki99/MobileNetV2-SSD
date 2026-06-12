import { render, screen, fireEvent } from '@testing-library/react'
import { ConfusionMatrix } from '../../../src/components/metrics/ConfusionMatrix'

const makeMatrix = (n: number, fill = 0) =>
  Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => r === c ? 80 : fill))

describe('ConfusionMatrix', () => {
  test('renders the component', () => {
    render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    expect(screen.getByTestId('confusion-matrix')).toBeInTheDocument()
  })

  test('renders the SVG', () => {
    render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    expect(screen.getByTestId('matrix-svg')).toBeInTheDocument()
  })

  test('renders N*N cells', () => {
    const { container } = render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    expect(container.querySelectorAll('[data-testid^="cell-"]').length).toBe(21 * 21)
  })

  test('renders legend min and max', () => {
    render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    expect(screen.getByTestId('legend-min')).toHaveTextContent('0')
    expect(screen.getByTestId('legend-max')).toHaveTextContent('80')
  })

  test('tooltip is not shown by default', () => {
    render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    expect(screen.queryByTestId('cell-tooltip')).not.toBeInTheDocument()
  })

  test('hovering a non-zero cell shows tooltip', () => {
    render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    fireEvent.mouseEnter(screen.getByTestId('cell-0-0'))
    expect(screen.getByTestId('cell-tooltip')).toBeInTheDocument()
    expect(screen.getByTestId('cell-tooltip')).toHaveTextContent('80 samples')
  })

  test('tooltip hides on mouse leave', () => {
    render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    fireEvent.mouseEnter(screen.getByTestId('cell-0-0'))
    fireEvent.mouseLeave(screen.getByTestId('cell-0-0'))
    expect(screen.queryByTestId('cell-tooltip')).not.toBeInTheDocument()
  })

  test('hovering a zero cell does not show tooltip', () => {
    const m = makeMatrix(21, 0)
    render(<ConfusionMatrix matrix={m} />)
    fireEvent.mouseEnter(screen.getByTestId('cell-0-1'))
    expect(screen.queryByTestId('cell-tooltip')).not.toBeInTheDocument()
  })

  test('tooltip shows correct row→col label', () => {
    render(<ConfusionMatrix matrix={makeMatrix(21)} />)
    fireEvent.mouseEnter(screen.getByTestId('cell-0-0'))
    expect(screen.getByTestId('cell-tooltip')).toHaveTextContent('bg → bg')
  })
})
