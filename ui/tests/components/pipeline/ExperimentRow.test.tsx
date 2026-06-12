import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ExperimentRow } from '../../../src/components/pipeline/ExperimentRow'
import { type Experiment } from '../../../src/types/experiment'

const baseExp: Experiment = {
  experiment_id: 'exp002_cloud_run',
  fingerprint: 'a1b2c3d4',
  status: 'success',
  region: 'us-east-1',
  best_metric: 0.766,
  best_epoch: 200,
  total_steps: 45000,
}

describe('ExperimentRow', () => {
  test('renders the experiment id', () => {
    render(<ExperimentRow exp={baseExp} />)
    expect(screen.getByText('exp002_cloud_run')).toBeInTheDocument()
  })

  test('renders the status badge', () => {
    render(<ExperimentRow exp={baseExp} />)
    expect(screen.getByText('success')).toBeInTheDocument()
  })

  test('renders the fingerprint', () => {
    render(<ExperimentRow exp={baseExp} />)
    expect(screen.getByText('a1b2c3d4')).toBeInTheDocument()
  })

  test('renders mAP when best_metric is present', () => {
    render(<ExperimentRow exp={baseExp} />)
    expect(screen.getByText('mAP 76.6%')).toBeInTheDocument()
  })

  test('does not render mAP when best_metric is absent', () => {
    render(<ExperimentRow exp={{ ...baseExp, best_metric: undefined }} />)
    expect(screen.queryByText(/mAP/)).not.toBeInTheDocument()
  })

  test('renders region when present', () => {
    render(<ExperimentRow exp={baseExp} />)
    expect(screen.getByText('us-east-1')).toBeInTheDocument()
  })

  test('renders — when fingerprint is absent', () => {
    render(<ExperimentRow exp={{ ...baseExp, fingerprint: undefined }} />)
    expect(screen.getByText('—')).toBeInTheDocument()
  })

  test('calls onClick when clicked', async () => {
    const handler = jest.fn()
    render(<ExperimentRow exp={baseExp} onClick={handler} />)
    await userEvent.click(screen.getByRole('button'))
    expect(handler).toHaveBeenCalledTimes(1)
  })

  test('selected state sets aria-selected', () => {
    render(<ExperimentRow exp={baseExp} selected={true} />)
    expect(screen.getByRole('button')).toHaveAttribute('aria-selected', 'true')
  })

  test('unselected state sets aria-selected to false', () => {
    render(<ExperimentRow exp={baseExp} selected={false} />)
    expect(screen.getByRole('button')).toHaveAttribute('aria-selected', 'false')
  })
})
