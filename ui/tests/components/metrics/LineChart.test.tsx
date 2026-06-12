import { render, screen, fireEvent } from '@testing-library/react'
import { LineChart } from '../../../src/components/metrics/LineChart'
import { MOCK_TRAIN_LOSS, MOCK_VAL_LOSS, MOCK_MAP_CURVE } from '../../../src/components/metrics/mockData'

const lossSeries = [
  { label: 'Train', data: MOCK_TRAIN_LOSS },
  { label: 'Val',   data: MOCK_VAL_LOSS   },
]
const mapSeries = [{ label: 'mAP', data: MOCK_MAP_CURVE }]

describe('LineChart', () => {
  test('renders the chart container', () => {
    render(<LineChart title="Loss Curves" series={lossSeries} />)
    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  test('renders the title', () => {
    render(<LineChart title="Loss Curves" series={lossSeries} />)
    expect(screen.getByTestId('chart-title')).toHaveTextContent('Loss Curves')
  })

  test('renders sub text when provided', () => {
    render(<LineChart title="Loss Curves" sub="train & validation" series={lossSeries} />)
    expect(screen.getByTestId('chart-sub')).toHaveTextContent('train & validation')
  })

  test('does not render sub when omitted', () => {
    render(<LineChart title="Loss Curves" series={lossSeries} />)
    expect(screen.queryByTestId('chart-sub')).not.toBeInTheDocument()
  })

  test('renders a legend entry for each series', () => {
    render(<LineChart title="Loss Curves" series={lossSeries} />)
    expect(screen.getByTestId('legend-Train')).toBeInTheDocument()
    expect(screen.getByTestId('legend-Val')).toBeInTheDocument()
  })

  test('renders the svg', () => {
    render(<LineChart title="Loss Curves" series={lossSeries} />)
    expect(screen.getByTestId('chart-svg')).toBeInTheDocument()
  })

  test('renders correct number of path elements for series', () => {
    const { container } = render(<LineChart title="Loss Curves" series={lossSeries} />)
    const paths = container.querySelectorAll('path')
    expect(paths.length).toBeGreaterThanOrEqual(lossSeries.length * 2)
  })

  test('renders with single series', () => {
    render(<LineChart title="mAP" series={mapSeries} />)
    expect(screen.getByTestId('legend-mAP')).toBeInTheDocument()
  })

  test('crosshair and popover hidden by default', () => {
    render(<LineChart title="Loss Curves" series={lossSeries} />)
    expect(screen.queryByTestId('crosshair')).not.toBeInTheDocument()
    expect(screen.queryByTestId('hover-popover')).not.toBeInTheDocument()
  })

  test('hides crosshair on mouse leave', () => {
    render(<LineChart title="Loss Curves" series={lossSeries} />)
    const svg = screen.getByTestId('chart-svg')
    fireEvent.mouseLeave(svg)
    expect(screen.queryByTestId('crosshair')).not.toBeInTheDocument()
  })
})
