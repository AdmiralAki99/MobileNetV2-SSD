import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DeployView } from '../../../src/components/deploy/DeployView'

const cicdData = {
  current_run: {
    id: 'run_001', branch: 'dev', commit: '84514d5',
    commit_message: 'Adds multi-model object detection',
    trigger: 'push', started_at: '2026-05-23T14:22:10Z', status: 'running',
    stages: [
      { id: 'lint',        name: 'Lint & Types', status: 'success', duration: 18.4 },
      { id: 'unit',        name: 'Unit Tests',   status: 'success', duration: 42.1 },
      { id: 'integration', name: 'Integration',  status: 'running', duration: null },
      { id: 'build',       name: 'Docker Build', status: 'pending', duration: null },
    ],
  },
  recent_runs: [
    { id: 'run_prev_1', branch: 'main', commit: '561e158', status: 'success', started_at: '2026-05-22T09:15:00Z', duration: 892 },
    { id: 'run_prev_2', branch: 'main', commit: '0478399', status: 'failed',  started_at: '2026-05-21T16:40:00Z', duration: 234 },
  ],
}

const releasesData = [
  { version: 'v1.2.0', model: 'mobilenetv2_ssd_voc', experiment: 'exp002_cloud_run', map_score: 0.766, status: 'current',    released_at: '2026-03-15T10:00:00Z', targets: ['jetson'], artifacts: { saved_model: true,  onnx: true,  tensorrt: true  } },
  { version: 'v1.1.0', model: 'mobilenetv2_ssd_voc', experiment: 'exp001_baseline',  map_score: 0.721, status: 'deprecated', released_at: '2026-01-20T14:30:00Z', targets: ['api'],    artifacts: { saved_model: true,  onnx: true,  tensorrt: false } },
]

describe('DeployView', () => {
  test('renders the view', () => {
    render(<DeployView />)
    expect(screen.getByTestId('deploy-view')).toBeInTheDocument()
  })

  test('renders the current run card', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByTestId('current-run-card')).toBeInTheDocument()
  })

  test('renders branch name', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByTestId('run-branch')).toHaveTextContent('dev')
  })

  test('renders commit message', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByTestId('run-commit-message')).toHaveTextContent('Adds multi-model')
  })

  test('renders trigger', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByTestId('run-trigger')).toHaveTextContent('push')
  })

  test('renders stage pipeline', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByTestId('stage-pipeline')).toBeInTheDocument()
  })

  test('renders each stage', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByTestId('stage-lint')).toBeInTheDocument()
    expect(screen.getByTestId('stage-unit')).toBeInTheDocument()
    expect(screen.getByTestId('stage-integration')).toBeInTheDocument()
  })

  test('shows log placeholder before stage selected', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByText('Click a stage above to view logs')).toBeInTheDocument()
  })

  test('clicking a stage shows its log', async () => {
    render(<DeployView cicdData={cicdData} />)
    await userEvent.click(screen.getByTestId('stage-lint'))
    expect(screen.getByTestId('log-stage-name')).toHaveTextContent('Lint & Types')
    expect(screen.getByTestId('log-lines')).toBeInTheDocument()
  })

  test('clicking selected stage deselects it', async () => {
    render(<DeployView cicdData={cicdData} />)
    await userEvent.click(screen.getByTestId('stage-lint'))
    await userEvent.click(screen.getByTestId('stage-lint'))
    expect(screen.getByText('Click a stage above to view logs')).toBeInTheDocument()
  })

  test('renders recent runs', () => {
    render(<DeployView cicdData={cicdData} />)
    expect(screen.getByTestId('recent-run-run_prev_1')).toBeInTheDocument()
    expect(screen.getByTestId('recent-run-run_prev_2')).toBeInTheDocument()
  })

  test('renders release history', () => {
    render(<DeployView releasesData={releasesData} />)
    expect(screen.getByTestId('release-v1.2.0')).toBeInTheDocument()
    expect(screen.getByTestId('release-v1.1.0')).toBeInTheDocument()
  })

  test('release shows mAP score', () => {
    render(<DeployView releasesData={releasesData} />)
    expect(screen.getByTestId('release-v1.2.0')).toHaveTextContent('mAP 76.6%')
  })

  test('shows empty state when no releases', () => {
    render(<DeployView releasesData={[]} />)
    expect(screen.getByText('No releases.')).toBeInTheDocument()
  })
})
