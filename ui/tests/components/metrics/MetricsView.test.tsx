import { render, screen } from '@testing-library/react'
import { MetricsView } from '../../../src/components/metrics/MetricsView'
import { MOCK_TRAIN_LOSS, MOCK_VAL_LOSS, MOCK_MAP_CURVE, MOCK_CLASS_AP } from '../../../src/components/metrics/mockData'
import { type Experiment } from '../../../src/types/experiment'

const exp: Experiment = {
  experiment_id: 'exp002_cloud_run',
  status: 'success',
  best_metric: 0.766,
  total_steps: 45000,
}

describe('MetricsView', () => {
  test('renders the view', () => {
    render(<MetricsView />)
    expect(screen.getByTestId('metrics-view')).toBeInTheDocument()
  })

  test('renders the stat strip with 5 tiles', () => {
    render(<MetricsView />)
    expect(screen.getByTestId('stat-strip')).toBeInTheDocument()
    expect(screen.getAllByTestId('stat-tile')).toHaveLength(5)
  })

  test('renders best mAP tile using mock data', () => {
    render(<MetricsView />)
    const tiles = screen.getAllByTestId('stat-label')
    const labels = tiles.map(t => t.textContent)
    expect(labels).toContain('Best mAP@0.5')
  })

  test('renders two line charts', () => {
    render(<MetricsView />)
    expect(screen.getAllByTestId('line-chart')).toHaveLength(2)
  })

  test('renders class AP chart', () => {
    render(<MetricsView />)
    expect(screen.getByTestId('class-ap-chart')).toBeInTheDocument()
  })

  test('uses provided metricsData over mock', () => {
    const customLoss = Array(200).fill(0.5)
    render(<MetricsView metricsData={{ train_loss: customLoss, val_loss: customLoss, map_curve: customLoss, class_ap: MOCK_CLASS_AP }} />)
    const tiles = screen.getAllByTestId('stat-value')
    const trainLossTile = tiles.find(t => t.textContent === '0.500')
    expect(trainLossTile).toBeInTheDocument()
  })

  test('shows total_steps from selectedExp', () => {
    render(<MetricsView selectedExp={exp} />)
    const tiles = screen.getAllByTestId('stat-value')
    const stepsTile = tiles.find(t => t.textContent === '45,000')
    expect(stepsTile).toBeInTheDocument()
  })

  test('shows — for total_steps when no experiment selected', () => {
    render(<MetricsView />)
    const tiles = screen.getAllByTestId('stat-value')
    const dashTile = tiles.find(t => t.textContent === '—')
    expect(dashTile).toBeInTheDocument()
  })

  test('shows experiment id as sub in total steps tile', () => {
    render(<MetricsView selectedExp={exp} />)
    const subs = screen.getAllByTestId('stat-sub')
    expect(subs.some(s => s.textContent === 'exp002_cloud_run')).toBe(true)
  })

  test('renders loss curves chart with correct title', () => {
    render(<MetricsView />)
    expect(screen.getByText('Loss Curves')).toBeInTheDocument()
  })

  test('renders mAP chart with correct title', () => {
    render(<MetricsView />)
    expect(screen.getByText('mAP@0.5')).toBeInTheDocument()
  })
})
