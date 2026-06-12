import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PipelineView } from '../../../src/components/pipeline/PipelineView'
import * as client from '../../../src/api/client'
import { type Experiment } from '../../../src/types/experiment'

jest.mock('../../../src/api/client')
const mockClient = client as jest.Mocked<typeof client>

const experiments: Experiment[] = [
  { experiment_id: 'exp001', status: 'failed',  fingerprint: 'aaa' },
  { experiment_id: 'exp002', status: 'success', fingerprint: 'bbb', best_metric: 0.766, checkpoint_s3_path: 's3://bucket/best' },
  { experiment_id: 'exp003', status: 'running', fingerprint: 'ccc', ec2_instance: 'i-0abc' },
]

beforeEach(() => {
  mockClient.fetchExperiments.mockResolvedValue(experiments)
  mockClient.fetchArtifacts.mockResolvedValue({ artifat_status: { saved_model: false, onnx: false, priors: false } })
  mockClient.fetchInstanceStatus.mockResolvedValue({})
  mockClient.fetchJobStatus.mockResolvedValue({ job_status: { status: 'success' } })
})
afterEach(() => jest.clearAllMocks())

describe('PipelineView', () => {
  test('shows loading state initially', () => {
    mockClient.fetchExperiments.mockReturnValue(new Promise(() => {}))
    render(<PipelineView statusFilter="all" />)
    expect(screen.getByText('Loading…')).toBeInTheDocument()
  })

  test('renders experiment list after fetch', async () => {
    render(<PipelineView statusFilter="all" />)
    await waitFor(() => expect(screen.getByText('exp001')).toBeInTheDocument())
    expect(screen.getByText('exp002')).toBeInTheDocument()
    expect(screen.getByText('exp003')).toBeInTheDocument()
  })

  test('shows experiment count after load', async () => {
    render(<PipelineView statusFilter="all" />)
    await waitFor(() => expect(screen.getByText('3 experiments')).toBeInTheDocument())
  })

  test('filters experiments by statusFilter prop', async () => {
    render(<PipelineView statusFilter="success" />)
    await waitFor(() => expect(screen.getByText('exp002')).toBeInTheDocument())
    expect(screen.queryByText('exp001')).not.toBeInTheDocument()
  })

  test('shows placeholder when no experiment selected', async () => {
    render(<PipelineView statusFilter="all" />)
    await waitFor(() => expect(screen.getByText('Select an experiment')).toBeInTheDocument())
  })

  test('shows detail panel when experiment is clicked', async () => {
    render(<PipelineView statusFilter="all" />)
    await waitFor(() => screen.getByText('exp002'))
    await userEvent.click(screen.getByText('exp002'))
    await waitFor(() => expect(screen.getByTestId('detail-panel')).toBeInTheDocument())
  })

  test('renders Refresh button', async () => {
    render(<PipelineView statusFilter="all" />)
    expect(screen.getByText('Refresh')).toBeInTheDocument()
  })

  test('Refresh button calls fetchExperiments again', async () => {
    render(<PipelineView statusFilter="all" />)
    await waitFor(() => screen.getByText('exp001'))
    await userEvent.click(screen.getByText('Refresh'))
    await waitFor(() => expect(mockClient.fetchExperiments).toHaveBeenCalledTimes(2))
  })
})
