import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DetailPanel } from '../../../src/components/pipeline/DetailPanel'
import { type Experiment } from '../../../src/types/experiment'
import * as client from '../../../src/api/client'

jest.mock('../../../src/api/client')
const mockClient = client as jest.Mocked<typeof client>

const successExp: Experiment = {
  experiment_id: 'exp002_cloud_run',
  fingerprint: 'a1b2c3d4',
  status: 'success',
  best_metric: 0.766,
  best_epoch: 200,
  checkpoint_s3_path: 's3://bucket/exp002/best',
  config_filename: 'exp002.yaml',
}

const runningExp: Experiment = {
  experiment_id: 'exp003',
  fingerprint: 'ccc',
  status: 'running',
  ec2_instance: 'i-0abc123',
}

const failedExp: Experiment = {
  experiment_id: 'exp001',
  fingerprint: 'aaa',
  status: 'failed',
  failure_reason: 'CUDA OOM: batch size too large',
}

beforeEach(() => {
  mockClient.fetchInstanceStatus.mockResolvedValue({})
  mockClient.fetchArtifacts.mockResolvedValue({ artifat_status: { saved_model: true, onnx: true, priors: true } })
  mockClient.fetchJobStatus.mockResolvedValue({ job_status: { status: 'success' } })
  mockClient.launchTraining.mockResolvedValue({ status: 'ok' })
  mockClient.stopTraining.mockResolvedValue({ status: 'ok' })
  mockClient.exportSavedModel.mockResolvedValue({ job_id: 'job_001', status: 'running' })
  mockClient.exportOnnx.mockResolvedValue({ job_id: 'job_002', status: 'running' })
})
afterEach(() => jest.clearAllMocks())

describe('DetailPanel', () => {
  test('shows placeholder when no experiment selected', () => {
    render(<DetailPanel selectedExp={null} onRefresh={() => {}} />)
    expect(screen.getByText('Select an experiment')).toBeInTheDocument()
  })

  test('renders experiment id when selected', async () => {
    render(<DetailPanel selectedExp={successExp} onRefresh={() => {}} />)
    expect(screen.getByText('exp002_cloud_run')).toBeInTheDocument()
  })

  test('shows mAP score when best_metric present', async () => {
    render(<DetailPanel selectedExp={successExp} onRefresh={() => {}} />)
    expect(screen.getByTestId('map-score')).toHaveTextContent('mAP 77%')
  })

  test('does not show mAP when best_metric absent', () => {
    render(<DetailPanel selectedExp={runningExp} onRefresh={() => {}} />)
    expect(screen.queryByTestId('map-score')).not.toBeInTheDocument()
  })

  test('shows failure reason for failed experiments', () => {
    render(<DetailPanel selectedExp={failedExp} onRefresh={() => {}} />)
    expect(screen.getByTestId('failure-reason')).toHaveTextContent('CUDA OOM')
  })

  test('Launch button enabled for pending/failed', () => {
    render(<DetailPanel selectedExp={failedExp} onRefresh={() => {}} />)
    expect(screen.getByText('Launch')).not.toBeDisabled()
  })

  test('Launch button disabled for running experiment', () => {
    render(<DetailPanel selectedExp={runningExp} onRefresh={() => {}} />)
    expect(screen.getByText('Launch')).toBeDisabled()
  })

  test('Stop button enabled for running experiment with ec2_instance', () => {
    render(<DetailPanel selectedExp={runningExp} onRefresh={() => {}} />)
    expect(screen.getByText('Stop')).not.toBeDisabled()
  })

  test('Stop button disabled for non-running experiment', () => {
    render(<DetailPanel selectedExp={successExp} onRefresh={() => {}} />)
    expect(screen.getByText('Stop')).toBeDisabled()
  })

  test('fetches artifacts on mount', async () => {
    render(<DetailPanel selectedExp={successExp} onRefresh={() => {}} />)
    await waitFor(() => expect(mockClient.fetchArtifacts).toHaveBeenCalledWith('exp002_cloud_run'))
  })

  test('renders artifact rows after fetch', async () => {
    render(<DetailPanel selectedExp={successExp} onRefresh={() => {}} />)
    await waitFor(() => expect(screen.getByTestId('artifacts')).toBeInTheDocument())
    expect(screen.getByText('SavedModel')).toBeInTheDocument()
    expect(screen.getByText('ONNX')).toBeInTheDocument()
  })

  test('calls launchTraining and onRefresh on Launch click', async () => {
    const onRefresh = jest.fn()
    render(<DetailPanel selectedExp={failedExp} onRefresh={onRefresh} />)
    await userEvent.click(screen.getByText('Launch'))
    await waitFor(() => expect(mockClient.launchTraining).toHaveBeenCalled())
    expect(onRefresh).toHaveBeenCalled()
  })

  test('calls stopTraining and onRefresh on Stop click', async () => {
    const onRefresh = jest.fn()
    render(<DetailPanel selectedExp={runningExp} onRefresh={onRefresh} />)
    await userEvent.click(screen.getByText('Stop'))
    await waitFor(() => expect(mockClient.stopTraining).toHaveBeenCalled())
    expect(onRefresh).toHaveBeenCalled()
  })
})
