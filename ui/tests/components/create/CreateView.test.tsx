import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CreateView } from '../../../src/components/create/CreateView'
import * as client from '../../../src/api/client'

jest.mock('../../../src/api/client')
const mockClient = client as jest.Mocked<typeof client>

const mockLibrary = {
  backbones: [
    { name: 'mobilenetv2', path: 'base/backbones/mobilenetv2.yaml', content: { backbone: { name: 'mobilenetv2' } } },
    { name: 'mobilenetv3', path: 'base/backbones/mobilenetv3.yaml', content: { backbone: { name: 'mobilenetv3' } } },
  ],
  losses: [
    { name: 'ssd_loss', path: 'base/losses/ssd_loss.yaml', content: { loss: { type: 'ssd' } } },
  ],
  optimizers: [
    { name: 'adamw_cosine', path: 'base/optimizers/adamw_cosine.yaml', content: { optimizer: { type: 'adamw' } } },
  ],
}

beforeEach(() => {
  mockClient.fetchConfigLibrary.mockResolvedValue(mockLibrary)
  mockClient.refreshConfigLibrary.mockResolvedValue({ status: 'synced' })
  mockClient.registerExperiment.mockResolvedValue({
    experiment_id: 'exp005', fingerprint: 'abc123', config_ref: 'experiments/exp005_abc123.json', created: true,
  })
})
afterEach(() => jest.clearAllMocks())

describe('CreateView', () => {
  test('renders the view', async () => {
    render(<CreateView />)
    await waitFor(() => expect(screen.getByTestId('create-view')).toBeInTheDocument())
  })

  test('shows loading state initially', () => {
    mockClient.fetchConfigLibrary.mockReturnValue(new Promise(() => {}))
    render(<CreateView />)
    expect(screen.getByText('Loading config library…')).toBeInTheDocument()
  })

  test('fetches config library on mount', async () => {
    render(<CreateView />)
    await waitFor(() => expect(mockClient.fetchConfigLibrary).toHaveBeenCalledTimes(1))
  })

  test('renders a selector for each category', async () => {
    render(<CreateView />)
    await waitFor(() => expect(screen.getByTestId('select-backbones')).toBeInTheDocument())
    expect(screen.getByTestId('select-losses')).toBeInTheDocument()
    expect(screen.getByTestId('select-optimizers')).toBeInTheDocument()
  })

  test('renders options for each category', async () => {
    render(<CreateView />)
    await waitFor(() => screen.getByTestId('select-backbones'))
    expect(screen.getByRole('option', { name: 'mobilenetv2' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'mobilenetv3' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'ssd_loss' })).toBeInTheDocument()
  })

  test('shows preview when a component is selected', async () => {
    render(<CreateView />)
    await waitFor(() => screen.getByTestId('select-backbones'))
    await userEvent.selectOptions(screen.getByTestId('select-backbones'), 'base/backbones/mobilenetv2.yaml')
    expect(screen.getByTestId('component-preview')).toBeInTheDocument()
    expect(screen.getByTestId('component-preview')).toHaveTextContent('mobilenetv2')
  })

  test('register button is present', async () => {
    render(<CreateView />)
    await waitFor(() => screen.getByTestId('register-btn'))
    expect(screen.getByTestId('register-btn')).toBeInTheDocument()
  })

  test('submitting calls registerExperiment', async () => {
    render(<CreateView />)
    await waitFor(() => screen.getByTestId('register-btn'))
    await userEvent.click(screen.getByTestId('register-btn'))
    await waitFor(() => expect(mockClient.registerExperiment).toHaveBeenCalledTimes(1))
  })

  test('shows success message on created: true', async () => {
    render(<CreateView />)
    await waitFor(() => screen.getByTestId('register-btn'))
    await userEvent.click(screen.getByTestId('register-btn'))
    await waitFor(() => expect(screen.getByTestId('register-result')).toBeInTheDocument())
    expect(screen.getByTestId('register-result')).toHaveTextContent('exp005')
    expect(screen.getByTestId('register-result')).toHaveTextContent('abc123')
  })

  test('shows already exists message on created: false', async () => {
    mockClient.registerExperiment.mockResolvedValue({
      experiment_id: 'exp005', fingerprint: 'abc123', config_ref: 'x', created: false,
    })
    render(<CreateView />)
    await waitFor(() => screen.getByTestId('register-btn'))
    await userEvent.click(screen.getByTestId('register-btn'))
    await waitFor(() => expect(screen.getByTestId('register-result')).toHaveTextContent('already exists'))
  })

  test('shows error on invalid overrides JSON', async () => {
    render(<CreateView />)
    await waitFor(() => screen.getByTestId('overrides-input'))
    await userEvent.clear(screen.getByTestId('overrides-input'))
    await userEvent.type(screen.getByTestId('overrides-input'), 'not valid json')
    await userEvent.click(screen.getByTestId('register-btn'))
    await waitFor(() => expect(screen.getByTestId('register-result')).toHaveTextContent('not valid JSON'))
  })

  test('refresh library button calls refreshConfigLibrary then fetchConfigLibrary', async () => {
    render(<CreateView />)
    await waitFor(() => screen.getByText('Refresh Library'))
    await userEvent.click(screen.getByText('Refresh Library'))
    await waitFor(() => expect(mockClient.refreshConfigLibrary).toHaveBeenCalled())
    expect(mockClient.fetchConfigLibrary).toHaveBeenCalledTimes(2)
  })

  test('shows empty state when library has no categories', async () => {
    mockClient.fetchConfigLibrary.mockResolvedValue({})
    render(<CreateView />)
    await waitFor(() => expect(screen.getByText(/No config library found/)).toBeInTheDocument())
  })
})
