import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { OpsView } from '../../../src/components/ops/OpsView'
import * as client from '../../../src/api/client'

jest.mock('../../../src/api/client')
const mockClient = client as jest.Mocked<typeof client>

const airflowData = {
  dag_id: 'etl_pipeline',
  schedule: '0 2 * * *',
  last_run: { run_id: 'run_001', state: 'success', duration: 503, start_date: '2026-05-29T02:00:02Z' },
  tasks: [
    { task_id: 'provision_ec2', state: 'success', duration: 0.8 },
    { task_id: 'run_etl_job',   state: 'success', duration: 483.2 },
  ],
}

const rayData = {
  status: 'running',
  dashboard_url: 'http://localhost:8265',
  resources: { cpu_used: 2.1, cpu_total: 4, memory_used_gb: 8.3, memory_total_gb: 16 },
  nodes: [{ id: 'node-001', ip: '127.0.0.1', status: 'alive', cpu_pct: 52 }],
}

const runsData = [
  { run_id: 'run_001', state: 'success', run_type: 'scheduled', duration: 503, start_date: '2026-05-29T02:00:02Z' },
  { run_id: 'run_002', state: 'failed',  run_type: 'manual',    duration: 47,  start_date: '2026-05-28T14:22:00Z' },
]

beforeEach(() => {
  mockClient.fetchAirflowRunTasks.mockResolvedValue([])
})
afterEach(() => jest.clearAllMocks())

describe('OpsView', () => {
  test('renders the view', () => {
    render(<OpsView />)
    expect(screen.getByTestId('ops-view')).toBeInTheDocument()
  })

  test('renders dag id', () => {
    render(<OpsView airflowData={airflowData} />)
    expect(screen.getByTestId('dag-id')).toHaveTextContent('etl_pipeline')
  })

  test('renders dag schedule', () => {
    render(<OpsView airflowData={airflowData} />)
    expect(screen.getByTestId('dag-schedule')).toHaveTextContent('0 2 * * *')
  })

  test('renders the dag graph', () => {
    render(<OpsView airflowData={airflowData} />)
    expect(screen.getByTestId('dag-graph')).toBeInTheDocument()
  })

  test('renders task rows in table', () => {
    render(<OpsView airflowData={airflowData} />)
    expect(screen.getByTestId('task-row-provision_ec2')).toBeInTheDocument()
    expect(screen.getByTestId('task-row-run_etl_job')).toBeInTheDocument()
  })

  test('renders ray panel', () => {
    render(<OpsView rayData={rayData} />)
    expect(screen.getByTestId('ray-panel')).toBeInTheDocument()
  })

  test('renders cpu bar when resources available', () => {
    render(<OpsView rayData={rayData} />)
    expect(screen.getByTestId('cpu-bar')).toBeInTheDocument()
  })

  test('renders ray nodes', () => {
    render(<OpsView rayData={rayData} />)
    expect(screen.getByTestId('node-node-001')).toBeInTheDocument()
  })

  test('renders run history rows', () => {
    render(<OpsView runsData={runsData} />)
    expect(screen.getByTestId('run-row-run_001')).toBeInTheDocument()
    expect(screen.getByTestId('run-row-run_002')).toBeInTheDocument()
  })

  test('clicking a run row fetches its tasks', async () => {
    render(<OpsView airflowData={airflowData} runsData={runsData} />)
    await userEvent.click(screen.getByTestId('run-row-run_001'))
    await waitFor(() => expect(mockClient.fetchAirflowRunTasks).toHaveBeenCalledWith('run_001'))
  })

  test('shows back to latest button when a run is selected', async () => {
    render(<OpsView airflowData={airflowData} runsData={runsData} />)
    await userEvent.click(screen.getByTestId('run-row-run_001'))
    expect(screen.getByTestId('back-to-latest')).toBeInTheDocument()
  })

  test('back to latest button clears selection', async () => {
    render(<OpsView airflowData={airflowData} runsData={runsData} />)
    await userEvent.click(screen.getByTestId('run-row-run_001'))
    await userEvent.click(screen.getByTestId('back-to-latest'))
    expect(screen.queryByTestId('back-to-latest')).not.toBeInTheDocument()
  })

  test('shows empty state when no tasks', () => {
    render(<OpsView airflowData={{ ...airflowData, tasks: [] }} />)
    expect(screen.getByText('No task runs.')).toBeInTheDocument()
  })
})
