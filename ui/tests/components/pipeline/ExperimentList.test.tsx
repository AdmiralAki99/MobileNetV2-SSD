import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ExperimentList } from '../../../src/components/pipeline/ExperimentList'
import { type Experiment } from '../../../src/types/experiment'

const experiments: Experiment[] = [
  { experiment_id: 'exp001', status: 'failed',  fingerprint: 'aaa' },
  { experiment_id: 'exp002', status: 'success', fingerprint: 'bbb', best_metric: 0.766 },
  { experiment_id: 'exp003', status: 'running', fingerprint: 'ccc' },
  { experiment_id: 'exp004', status: 'pending', fingerprint: 'ddd' },
]

describe('ExperimentList', () => {
  test('renders all experiments when filter is "all"', () => {
    render(<ExperimentList experiments={experiments} statusFilter="all" onSelect={() => {}} />)
    experiments.forEach(e => expect(screen.getByText(e.experiment_id)).toBeInTheDocument())
  })

  test('filters to only matching status', () => {
    render(<ExperimentList experiments={experiments} statusFilter="success" onSelect={() => {}} />)
    expect(screen.getByText('exp002')).toBeInTheDocument()
    expect(screen.queryByText('exp001')).not.toBeInTheDocument()
    expect(screen.queryByText('exp003')).not.toBeInTheDocument()
  })

  test('shows empty message when no experiments match filter', () => {
    render(<ExperimentList experiments={experiments} statusFilter="queued" onSelect={() => {}} />)
    expect(screen.getByText('No experiments match this filter.')).toBeInTheDocument()
  })

  test('shows empty message when experiments array is empty', () => {
    render(<ExperimentList experiments={[]} statusFilter="all" onSelect={() => {}} />)
    expect(screen.getByText('No experiments match this filter.')).toBeInTheDocument()
  })

  test('calls onSelect with the clicked experiment', async () => {
    const onSelect = jest.fn()
    render(<ExperimentList experiments={experiments} statusFilter="all" onSelect={onSelect} />)
    await userEvent.click(screen.getByText('exp002'))
    expect(onSelect).toHaveBeenCalledWith(experiments[1])
  })

  test('marks the selected experiment as selected', () => {
    render(<ExperimentList experiments={experiments} selectedId="exp003" statusFilter="all" onSelect={() => {}} />)
    const rows = screen.getAllByRole('button')
    const selectedRow = rows.find(r => r.textContent?.includes('exp003'))!
    expect(selectedRow).toHaveAttribute('aria-selected', 'true')
  })

  test('non-selected experiments have aria-selected false', () => {
    render(<ExperimentList experiments={experiments} selectedId="exp003" statusFilter="all" onSelect={() => {}} />)
    const rows = screen.getAllByRole('button')
    const unselected = rows.find(r => r.textContent?.includes('exp001'))!
    expect(unselected).toHaveAttribute('aria-selected', 'false')
  })
})
